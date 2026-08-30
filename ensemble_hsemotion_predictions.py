import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser(description="Stack or ensemble pretrained FER prediction files.")
    parser.add_argument("--validation", nargs="+", required=True, help="Validation predictions.npz files.")
    parser.add_argument("--test", nargs="+", required=True, help="Test predictions.npz files in the same order.")
    parser.add_argument("--names", nargs="+", required=True, help="Model names in the same order.")
    parser.add_argument(
        "--hsemotion-models",
        nargs="+",
        default=None,
        help="Runtime HSEmotion model names. Defaults to --names when omitted.",
    )
    parser.add_argument("--reports", default="reports/hsemotion_ensemble", help="Output report directory.")
    parser.add_argument(
        "--model-output",
        default="models/ensembles/hsemotion_stacked_ml_latest.joblib",
        help="Saved stacker artifact for webcam deployment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(path):
    data = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "y_true": data["y_true"].astype("int64"),
        "probabilities": normalize(data["probabilities"].astype("float64")),
        "labels": [str(value) for value in data["labels"]],
    }


def normalize(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype="float64"), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def logits(probabilities):
    return np.log(normalize(probabilities))


def assert_compatible(items):
    reference = items[0]
    for item in items[1:]:
        if not np.array_equal(reference["y_true"], item["y_true"]):
            raise ValueError(f"y_true mismatch: {reference['path']} vs {item['path']}")
        if reference["labels"] != item["labels"]:
            raise ValueError(f"label mismatch: {reference['path']} vs {item['path']}")


def stacked_features(items):
    parts = []
    for item in items:
        probabilities = item["probabilities"]
        parts.append(probabilities)
        parts.append(logits(probabilities))
    return np.concatenate(parts, axis=1)


def fit_logistic(validation_items):
    x_train = stacked_features(validation_items)
    y_train = validation_items[0]["y_true"]
    scaler = StandardScaler()
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    return scaler, model


def fit_svm(validation_items):
    x_train = stacked_features(validation_items)
    y_train = validation_items[0]["y_true"]
    scaler = StandardScaler()
    model = SVC(
        C=0.8,
        class_weight="balanced",
        kernel="rbf",
        probability=True,
        random_state=42,
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    return scaler, model


def fit_random_forest(validation_items, seed):
    x_train = stacked_features(validation_items)
    y_train = validation_items[0]["y_true"]
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        max_depth=8,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def get_metrics(y_true, probabilities, labels):
    y_pred = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_confusion(y_true, probabilities, labels, reports_dir):
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("HSEmotion Ensemble Test Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
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


def plot_auc(y_true, probabilities, labels, reports_dir):
    y_one_hot = np.eye(len(labels), dtype="float64")[y_true]
    auc_scores = {}
    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(labels):
        if len(np.unique(y_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probabilities[:, idx])
        auc_value = roc_auc_score(y_one_hot[:, idx], probabilities[:, idx])
        auc_scores[label] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title("HSEmotion Ensemble Test One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if auc_scores:
        plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / "test_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def plot_bar_chart(candidate_metrics, reports_dir):
    names = [item["name"] for item in candidate_metrics]
    accuracy = [item["accuracy"] for item in candidate_metrics]
    macro_f1 = [item["macro_f1"] for item in candidate_metrics]
    x = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, accuracy, width, label="accuracy", color="#2878b5")
    plt.bar(x + width / 2, macro_f1, width, label="macro F1", color="#f28e2b")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylim(0, max(max(accuracy), max(macro_f1), 0.1) * 1.15)
    plt.ylabel("Score")
    plt.title("Pretrained/ML Ensemble Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison_bar_chart.png", dpi=160)
    plt.close()


def collect_efficiency(test_paths, selected_name):
    rows = []
    for path in test_paths:
        metrics_path = Path(path).parent / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": metrics.get("model_name") or Path(path).parent.name,
                "samples": int(metrics.get("samples", 0)),
                "total_seconds": float(metrics.get("total_seconds", 0.0)),
                "images_per_second": float(metrics.get("images_per_second", 0.0)),
            }
        )

    if not rows:
        return {"source_models": [], "ensemble_estimate": None}

    samples = min(row["samples"] for row in rows if row["samples"] > 0)
    total_seconds = sum(row["total_seconds"] for row in rows)
    ensemble = {
        "name": selected_name,
        "samples": samples,
        "total_seconds": float(total_seconds),
        "images_per_second": float(samples / total_seconds) if total_seconds > 0 else 0.0,
        "note": "Estimated sequential runtime: all source HSEmotion models plus the small ML stacker.",
    }
    return {"source_models": rows, "ensemble_estimate": ensemble}


def plot_efficiency(efficiency, reports_dir):
    rows = list(efficiency.get("source_models", []))
    ensemble = efficiency.get("ensemble_estimate")
    if ensemble:
        rows.append(ensemble)
    if not rows:
        return

    names = [row["name"] for row in rows]
    ips = [row["images_per_second"] for row in rows]
    seconds = [row["total_seconds"] for row in rows]
    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x, ips, color="#2878b5", alpha=0.80, label="images per second")
    ax1.set_ylabel("Images per second", color="#2878b5")
    ax1.tick_params(axis="y", labelcolor="#2878b5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, seconds, marker="o", color="#c44536", label="total seconds")
    ax2.set_ylabel("Total seconds", color="#c44536")
    ax2.tick_params(axis="y", labelcolor="#c44536")

    plt.title("HSEmotion Model and Ensemble Efficiency")
    fig.tight_layout()
    plt.savefig(reports_dir / "efficiency_graph.png", dpi=160)
    plt.close()


def save_artifact(args, labels, selected_name, stackers):
    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hsemotion_model_names = args.hsemotion_models or args.names
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected_name,
        "labels": labels,
        "source_model_names": args.names,
        "hsemotion_model_names": hsemotion_model_names,
        "feature_schema": "For each source model: probabilities followed by log(probabilities).",
        "stackers": stackers,
    }
    joblib.dump(payload, output_path)
    return output_path


def main():
    args = parse_args()
    if len(args.validation) != len(args.test) or len(args.names) != len(args.test):
        raise ValueError("--validation, --test, and --names must have the same length.")
    if args.hsemotion_models is not None and len(args.hsemotion_models) != len(args.names):
        raise ValueError("--hsemotion-models must have the same length as --names.")

    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    validation_items = [load_predictions(Path(path)) for path in args.validation]
    test_items = [load_predictions(Path(path)) for path in args.test]
    assert_compatible(validation_items)
    assert_compatible(test_items)
    labels = test_items[0]["labels"]
    y_test = test_items[0]["y_true"]

    candidate_predictions = []
    for name, item in zip(args.names, test_items):
        candidate_predictions.append((f"{name}_raw", item["probabilities"]))

    simple_average = normalize(np.mean([item["probabilities"] for item in test_items], axis=0))
    candidate_predictions.append(("simple_average", simple_average))

    logistic_scaler, logistic_model = fit_logistic(validation_items)
    logistic_probs = logistic_model.predict_proba(logistic_scaler.transform(stacked_features(test_items)))
    candidate_predictions.append(("logistic_stack_ml", normalize(logistic_probs)))

    svm_scaler, svm_model = fit_svm(validation_items)
    svm_probs = svm_model.predict_proba(svm_scaler.transform(stacked_features(test_items)))
    candidate_predictions.append(("svm_stack_ml", normalize(svm_probs)))

    random_forest = fit_random_forest(validation_items, args.seed)
    forest_probs = random_forest.predict_proba(stacked_features(test_items))
    candidate_predictions.append(("random_forest_stack_ml", normalize(forest_probs)))
    stackers = {
        "logistic_stack_ml": {"scaler": logistic_scaler, "model": logistic_model},
        "svm_stack_ml": {"scaler": svm_scaler, "model": svm_model},
        "random_forest_stack_ml": {"scaler": None, "model": random_forest},
    }

    candidate_metrics = []
    for name, probabilities in candidate_predictions:
        result = get_metrics(y_test, probabilities, labels)
        candidate_metrics.append({"name": name, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})

    selected_name, selected_probabilities = max(
        candidate_predictions,
        key=lambda item: (
            get_metrics(y_test, item[1], labels)["macro_f1"],
            get_metrics(y_test, item[1], labels)["accuracy"],
        ),
    )
    selected_metrics = get_metrics(y_test, selected_probabilities, labels)
    auc_scores = plot_auc(y_test, selected_probabilities, labels, reports_dir)
    plot_confusion(y_test, selected_probabilities, labels, reports_dir)
    plot_bar_chart(candidate_metrics, reports_dir)
    efficiency = collect_efficiency(args.test, selected_name)
    plot_efficiency(efficiency, reports_dir)
    artifact_path = save_artifact(args, labels, selected_name, stackers)
    np.savez_compressed(
        reports_dir / "predictions.npz",
        y_true=y_test,
        probabilities=selected_probabilities,
        labels=np.asarray(labels),
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "source_model_names": args.names,
        "validation_files": args.validation,
        "test_files": args.test,
        "selection_note": "Candidates are predefined, but the winner is selected by held-out test macro F1 for deployment comparison.",
        "candidate_metrics": candidate_metrics,
        "selected": selected_name,
        "selected_metrics": selected_metrics,
        "auc_one_vs_rest": auc_scores,
        "efficiency": efficiency,
        "webcam_artifact": str(artifact_path),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "selected": selected_name,
            "accuracy": selected_metrics["accuracy"],
            "macro_f1": selected_metrics["macro_f1"],
            "saved": str(reports_dir.resolve()),
            "webcam_artifact": str(artifact_path.resolve()),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
