from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve

from audio_emotion import make_audio_recognizer
from emotion_schema import EMOTION_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate audio emotion recognition on folder-per-class WAV data.")
    parser.add_argument("--dataset", default="dataset_ravdess_audio/test")
    parser.add_argument("--backend", choices=["acoustic", "hf"], default="acoustic")
    parser.add_argument("--reports", default="reports/audio_ravdess_latest")
    parser.add_argument("--hf-local-files-only", action="store_true")
    return parser.parse_args()


def iter_audio_files(dataset):
    root = Path(dataset)
    for label_index, label in enumerate(EMOTION_LABELS):
        folder = root / label
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.wav")):
            yield label_index, path


def plot_confusion(y_true, y_pred, reports_dir):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Audio Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(EMOTION_LABELS))
    plt.xticks(ticks, EMOTION_LABELS, rotation=45, ha="right")
    plt.yticks(ticks, EMOTION_LABELS)
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
    plt.savefig(reports_dir / "audio_confusion_matrix.png", dpi=160)
    plt.close()


def plot_auc(y_true, probabilities, reports_dir):
    y_one_hot = np.eye(len(EMOTION_LABELS), dtype="float64")[y_true]
    auc_scores = {}
    plt.figure(figsize=(9, 7))
    for index, label in enumerate(EMOTION_LABELS):
        if len(np.unique(y_one_hot[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, index], probabilities[:, index])
        auc_value = roc_auc_score(y_one_hot[:, index], probabilities[:, index])
        auc_scores[label] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title("Audio One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if auc_scores:
        plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / "audio_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def plot_class_distribution(y_true, reports_dir):
    counts = np.bincount(np.asarray(y_true, dtype="int64"), minlength=len(EMOTION_LABELS))
    plt.figure(figsize=(10, 5))
    plt.bar(EMOTION_LABELS, counts, color="#2878b5")
    plt.title("Audio Test Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Audio clips")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "audio_class_distribution_bar_chart.png", dpi=160)
    plt.close()


def plot_efficiency(total_seconds, total_files, reports_dir):
    clips_per_second = total_files / total_seconds if total_seconds > 0 else 0.0
    plt.figure(figsize=(8, 5))
    plt.bar(["audio inference"], [clips_per_second], color="#2878b5")
    plt.title("Audio Inference Efficiency")
    plt.ylabel("Clips per second")
    plt.tight_layout()
    plt.savefig(reports_dir / "audio_efficiency_graph.png", dpi=160)
    plt.close()
    return clips_per_second


def main():
    args = parse_args()
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    recognizer = make_audio_recognizer(args.backend, local_files_only=args.hf_local_files_only)

    y_true = []
    probabilities = []
    files = []
    started_at = time.perf_counter()
    for label_index, path in iter_audio_files(args.dataset):
        result = recognizer.predict_file(path)
        y_true.append(label_index)
        probabilities.append([result.probabilities[label] for label in EMOTION_LABELS])
        files.append(str(path))

    total_seconds = time.perf_counter() - started_at
    if not y_true:
        raise ValueError(f"No WAV files found in {args.dataset}")

    y_true = np.asarray(y_true, dtype="int64")
    probabilities = np.asarray(probabilities, dtype="float32")
    y_pred = np.argmax(probabilities, axis=1)

    plot_confusion(y_true, y_pred, reports_dir)
    auc_scores = plot_auc(y_true, probabilities, reports_dir)
    plot_class_distribution(y_true, reports_dir)
    clips_per_second = plot_efficiency(total_seconds, len(files), reports_dir)
    np.savez_compressed(
        reports_dir / "audio_predictions.npz",
        y_true=y_true,
        probabilities=probabilities,
        labels=np.asarray(EMOTION_LABELS),
        files=np.asarray(files),
    )

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "backend": args.backend,
        "labels": EMOTION_LABELS,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "clips_per_second": float(clips_per_second),
        "auc_one_vs_rest": auc_scores,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    (reports_dir / "audio_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "reports": str(reports_dir)}, indent=2))


if __name__ == "__main__":
    main()
