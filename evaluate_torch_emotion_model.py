import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PyTorch facial emotion model.")
    parser.add_argument("--model", required=True, help="Saved .pt checkpoint.")
    parser.add_argument("--dataset", default="dataset_mutfer2024/test", help="ImageFolder split to evaluate.")
    parser.add_argument("--reports", default="reports/torch_eval_latest", help="Output report folder.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tta-horizontal-flip", action="store_true", help="Average original and flipped predictions.")
    return parser.parse_args()


def make_model(arch, num_classes):
    efficientnet_factories = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
    }
    if arch in efficientnet_factories:
        model = efficientnet_factories[arch](weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    labels = list(checkpoint["labels"])
    if labels != LABELS:
        raise ValueError(f"Checkpoint labels are {labels}; expected {LABELS}")
    model = make_model(checkpoint["arch"], len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return checkpoint, model


def evaluate(model, loader, device, tta_horizontal_flip):
    y_true = []
    probabilities = []
    started_at = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if tta_horizontal_flip:
                flipped_logits = model(torch.flip(images, dims=[3]))
                probs = (torch.softmax(logits, dim=1) + torch.softmax(flipped_logits, dim=1)) / 2.0
            else:
                probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.detach().cpu().numpy())
            y_true.extend(labels.numpy().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started_at
    probabilities = np.concatenate(probabilities, axis=0)
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.argmax(probabilities, axis=1)
    return {
        "y_true": y_true,
        "probabilities": probabilities,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "total_seconds": float(total_seconds),
        "images_per_second": float(len(y_true) / total_seconds) if total_seconds > 0 else 0.0,
    }


def plot_confusion(y_true, probabilities, reports_dir):
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("PyTorch Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, LABELS, rotation=45, ha="right")
    plt.yticks(ticks, LABELS)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(reports_dir / "test_confusion_matrix.png", dpi=160)
    plt.close()


def plot_auc(y_true, probabilities, reports_dir):
    y_one_hot = np.eye(len(LABELS), dtype="float64")[y_true]
    auc_scores = {}
    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(LABELS):
        if len(np.unique(y_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probabilities[:, idx])
        auc_value = roc_auc_score(y_one_hot[:, idx], probabilities[:, idx])
        auc_scores[label] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title("PyTorch Test One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / "test_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def plot_efficiency(metrics, reports_dir):
    plt.figure(figsize=(7, 5))
    plt.bar(["images/sec"], [metrics["images_per_second"]], color="#2878b5")
    plt.title("PyTorch Evaluation Efficiency")
    plt.ylabel("Images per second")
    plt.tight_layout()
    plt.savefig(reports_dir / "efficiency_graph.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    checkpoint, model = load_checkpoint(Path(args.model), device)
    dataset = datasets.ImageFolder(args.dataset, transform=eval_transform(int(checkpoint.get("image_size", 224))))
    if dataset.classes != LABELS:
        raise ValueError(f"{args.dataset} classes are {dataset.classes}; expected {LABELS}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, loader, device, args.tta_horizontal_flip)
    auc_scores = plot_auc(metrics["y_true"], metrics["probabilities"], reports_dir)
    plot_confusion(metrics["y_true"], metrics["probabilities"], reports_dir)
    plot_efficiency(metrics, reports_dir)
    np.savez_compressed(
        reports_dir / "test_predictions.npz",
        y_true=metrics["y_true"],
        probabilities=metrics["probabilities"],
        labels=np.asarray(LABELS),
    )
    y_pred = np.argmax(metrics["probabilities"], axis=1)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "images_per_second": metrics["images_per_second"],
        "auc_one_vs_rest": auc_scores,
        "classification_report": classification_report(
            metrics["y_true"],
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": payload["accuracy"], "macro_f1": payload["macro_f1"], "reports": str(reports_dir)}, indent=2))


if __name__ == "__main__":
    main()
