import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from resemotenet_model import ResEmoteNet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


CANONICAL_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
RESEMOTENET_LABELS = ["happy", "surprise", "sad", "angry", "disgust", "fear", "neutral"]
REORDER_TO_CANONICAL = [RESEMOTENET_LABELS.index(label) for label in CANONICAL_LABELS]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype="float32")[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype="float32")[:, None, None]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a ResEmoteNet PyTorch checkpoint.")
    parser.add_argument("--dataset", default="dataset_mutfer2024", help="Dataset root with train/validation/test folders.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Split to evaluate.")
    parser.add_argument("--checkpoint", default="models/resemotenet/ResEmoteNetBS32.pth", help="PyTorch checkpoint path.")
    parser.add_argument("--reports", default="reports/resemotenet_bs32_test", help="Output report directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images for quick smoke tests.")
    parser.add_argument(
        "--detector",
        choices=["none", "yunet"],
        default="yunet",
        help="Optional face detector crop before emotion inference.",
    )
    parser.add_argument(
        "--yunet-model",
        default="models/yunet/face_detection_yunet_2023mar.onnx",
        help="Path to OpenCV YuNet face detection model.",
    )
    parser.add_argument("--face-margin", type=float, default=0.0, help="Extra crop margin around detected faces.")
    return parser.parse_args()


def find_split(dataset_root, split):
    split_dir = dataset_root / split
    if split_dir.is_dir():
        return split_dir
    raise FileNotFoundError(f"Could not find split folder: {split_dir}")


def collect_samples(split_dir):
    samples = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = class_dir.name.lower()
        if label not in CANONICAL_LABELS:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))
    if not samples:
        raise FileNotFoundError(f"No supported images found in {split_dir}")
    return samples


class YuNetFaceDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YuNet model not found: {path}")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3, 5000)

    def detect_largest(self, image):
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return None
        boxes = []
        for face in faces:
            raw_box = np.asarray(face[:4], dtype="float64")
            if not np.all(np.isfinite(raw_box)):
                continue
            box = np.rint(np.clip(raw_box, -1_000_000, 1_000_000)).astype("int64")
            if box[2] > 0 and box[3] > 0:
                boxes.append(box)
        if not boxes:
            return None
        return max(boxes, key=lambda box: int(box[2]) * int(box[3]))


def expand_box(box, image_shape, margin):
    x, y, w, h = [int(value) for value in box]
    height, width = image_shape[:2]
    extra_w = int(w * margin)
    extra_h = int(h * margin)
    x1 = max(0, x - extra_w)
    y1 = max(0, y - extra_h)
    x2 = min(width, x + w + extra_w)
    y2 = min(height, y + h + extra_h)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def load_model(checkpoint_path):
    model = ResEmoteNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    image_3ch = np.stack([gray, gray, gray], axis=0)
    image_3ch = (image_3ch - IMAGENET_MEAN) / IMAGENET_STD
    return image_3ch.astype("float32")


def read_batch(batch, detector, margin):
    tensors = []
    labels = []
    skipped = []
    for path, label in batch:
        image = cv2.imread(str(path))
        if image is None:
            skipped.append(str(path))
            continue
        if detector is not None:
            box = detector.detect_largest(image)
            if box is None:
                skipped.append(str(path))
                continue
            x, y, w, h = expand_box(box, image.shape, margin)
            image = image[y : y + h, x : x + w]
        tensors.append(preprocess_image(image))
        labels.append(label)
    if not tensors:
        return None, [], skipped
    return torch.from_numpy(np.stack(tensors, axis=0)), labels, skipped


def plot_class_distribution(samples, reports_dir):
    counts = {label: 0 for label in CANONICAL_LABELS}
    for _, label in samples:
        counts[label] += 1
    plt.figure(figsize=(10, 5))
    plt.bar(CANONICAL_LABELS, [counts[label] for label in CANONICAL_LABELS], color="#2878b5")
    plt.title("Evaluation Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution_bar_chart.png", dpi=160)
    plt.close()


def plot_confusion(y_true, y_pred, reports_dir, split_name):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CANONICAL_LABELS))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"ResEmoteNet {split_name.title()} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CANONICAL_LABELS))
    plt.xticks(tick_marks, CANONICAL_LABELS, rotation=45, ha="right")
    plt.yticks(tick_marks, CANONICAL_LABELS)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{split_name}_confusion_matrix.png", dpi=160)
    plt.close()


def plot_auc(y_true, probabilities, reports_dir, split_name):
    y_one_hot = np.eye(len(CANONICAL_LABELS), dtype="float32")[y_true]
    auc_scores = {}
    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(CANONICAL_LABELS):
        if len(np.unique(y_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probabilities[:, idx])
        auc_value = roc_auc_score(y_one_hot[:, idx], probabilities[:, idx])
        auc_scores[label] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title(f"ResEmoteNet {split_name.title()} One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if auc_scores:
        plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / f"{split_name}_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def plot_efficiency(batch_seconds, batch_sizes, reports_dir):
    batches = np.arange(1, len(batch_seconds) + 1)
    images_per_second = [
        size / seconds if seconds > 0 else 0 for size, seconds in zip(batch_sizes, batch_seconds)
    ]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(batches, batch_seconds, marker="o", color="#c44536")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Seconds per batch", color="#c44536")
    ax1.tick_params(axis="y", labelcolor="#c44536")
    ax2 = ax1.twinx()
    ax2.bar(batches, images_per_second, alpha=0.25, color="#2878b5")
    ax2.set_ylabel("Images per second", color="#2878b5")
    ax2.tick_params(axis="y", labelcolor="#2878b5")
    plt.title("ResEmoteNet Evaluation Efficiency")
    fig.tight_layout()
    plt.savefig(reports_dir / "efficiency_graph.png", dpi=160)
    plt.close()


def evaluate(args):
    dataset_root = Path(args.dataset)
    split_dir = find_split(dataset_root, args.split)
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(split_dir)
    if args.limit > 0:
        samples = samples[: args.limit]

    model = load_model(args.checkpoint)
    detector = YuNetFaceDetector(args.yunet_model) if args.detector == "yunet" else None

    y_true = []
    probabilities = []
    skipped = []
    batch_seconds = []
    batch_sizes = []

    started_at = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        tensor, labels, batch_skipped = read_batch(batch, detector, args.face_margin)
        skipped.extend(batch_skipped)
        if tensor is None:
            continue

        batch_started_at = time.perf_counter()
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()
        batch_seconds.append(time.perf_counter() - batch_started_at)
        batch_sizes.append(len(labels))

        probabilities.append(probs[:, REORDER_TO_CANONICAL])
        y_true.extend(CANONICAL_LABELS.index(label) for label in labels)

        done = min(start + len(batch), len(samples))
        print(f"Evaluated {done}/{len(samples)} images", flush=True)

    probabilities = np.concatenate(probabilities, axis=0)
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.argmax(probabilities, axis=1)
    total_seconds = time.perf_counter() - started_at

    plot_class_distribution(samples, reports_dir)
    plot_confusion(y_true, y_pred, reports_dir, args.split)
    auc_scores = plot_auc(y_true, probabilities, reports_dir, args.split)
    plot_efficiency(batch_seconds, batch_sizes, reports_dir)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CANONICAL_LABELS))),
        target_names=CANONICAL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "dataset": str(dataset_root.resolve()),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": CANONICAL_LABELS,
        "raw_model_labels": RESEMOTENET_LABELS,
        "detector": args.detector,
        "face_margin": float(args.face_margin),
        "samples": int(len(y_true)),
        "skipped": skipped,
        "total_seconds": float(total_seconds),
        "images_per_second": float(len(y_true) / total_seconds) if total_seconds > 0 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "auc_one_vs_rest": auc_scores,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}, indent=2))
    print(f"Saved reports: {reports_dir.resolve()}")


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
