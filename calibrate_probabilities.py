import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate emotion probabilities using a validation split.")
    parser.add_argument("--validation", required=True, help="Validation predictions.npz.")
    parser.add_argument("--test", required=True, help="Test predictions.npz.")
    parser.add_argument("--output", default="models/calibration/hsemotion_enet_b2_7_yunet000.json")
    parser.add_argument("--report", default="reports/calibration_hsemotion_enet_b2_7_yunet000.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(path):
    data = np.load(path, allow_pickle=True)
    return {
        "y_true": data["y_true"].astype("int64"),
        "probabilities": data["probabilities"].astype("float64"),
        "labels": [str(value) for value in data["labels"]],
    }


def normalize(probabilities):
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def logits(probabilities):
    return np.log(normalize(probabilities))


def metrics(y_true, probabilities):
    y_pred = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fit_prior_bias(probabilities, y_true, num_classes):
    predicted_prior = normalize(probabilities).mean(axis=0)
    true_prior = np.bincount(y_true, minlength=num_classes).astype("float64")
    true_prior = true_prior / true_prior.sum()
    return np.log(np.clip(true_prior, 1e-12, None)) - np.log(np.clip(predicted_prior, 1e-12, None))


def apply_prior_bias(probabilities, bias):
    adjusted = logits(probabilities) + bias
    adjusted = adjusted - adjusted.max(axis=1, keepdims=True)
    exp_values = np.exp(adjusted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def fit_logistic(features, y_true):
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(features, y_true)
    return model


def logistic_to_payload(model, feature_type):
    return {
        "method": "logistic_regression",
        "feature_type": feature_type,
        "classes": [int(value) for value in model.classes_],
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def apply_logistic_payload(probabilities, payload):
    features = logits(probabilities) if payload["feature_type"] == "logits" else normalize(probabilities)
    coef = np.asarray(payload["coef"], dtype="float64")
    intercept = np.asarray(payload["intercept"], dtype="float64")
    scores = features @ coef.T + intercept
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_values = np.exp(scores)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def build_candidates(validation, seed):
    y = validation["y_true"]
    p = validation["probabilities"]
    num_classes = p.shape[1]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(p, y))

    candidates = []

    bias = fit_prior_bias(p[train_idx], y[train_idx], num_classes)
    holdout_probs = apply_prior_bias(p[holdout_idx], bias)
    candidates.append(
        {
            "name": "prior_bias",
            "holdout": metrics(y[holdout_idx], holdout_probs),
            "payload": {"method": "prior_bias", "bias": bias.tolist()},
        }
    )

    for feature_type, feature_fn in [("probabilities", normalize), ("logits", logits)]:
        model = fit_logistic(feature_fn(p[train_idx]), y[train_idx])
        payload = logistic_to_payload(model, feature_type)
        holdout_probs = apply_logistic_payload(p[holdout_idx], payload)
        candidates.append(
            {
                "name": f"logistic_{feature_type}",
                "holdout": metrics(y[holdout_idx], holdout_probs),
                "payload": payload,
            }
        )

    return candidates


def refit_payload(name, validation):
    y = validation["y_true"]
    p = validation["probabilities"]
    if name == "prior_bias":
        bias = fit_prior_bias(p, y, p.shape[1])
        return {"method": "prior_bias", "bias": bias.tolist()}
    if name == "logistic_probabilities":
        return logistic_to_payload(fit_logistic(normalize(p), y), "probabilities")
    if name == "logistic_logits":
        return logistic_to_payload(fit_logistic(logits(p), y), "logits")
    raise ValueError(f"Unknown calibration method: {name}")


def apply_payload(probabilities, payload):
    if payload["method"] == "prior_bias":
        return apply_prior_bias(probabilities, np.asarray(payload["bias"], dtype="float64"))
    if payload["method"] == "logistic_regression":
        return apply_logistic_payload(probabilities, payload)
    raise ValueError(f"Unknown calibration method: {payload['method']}")


def main():
    args = parse_args()
    validation = load_predictions(args.validation)
    test = load_predictions(args.test)

    raw_validation = metrics(validation["y_true"], validation["probabilities"])
    raw_test = metrics(test["y_true"], test["probabilities"])
    candidates = build_candidates(validation, args.seed)
    selected = max(candidates, key=lambda item: (item["holdout"]["macro_f1"], item["holdout"]["accuracy"]))
    final_payload = refit_payload(selected["name"], validation)
    calibrated_test_probs = apply_payload(test["probabilities"], final_payload)
    calibrated_test = metrics(test["y_true"], calibrated_test_probs)

    output_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": validation["labels"],
        "selected_method": selected["name"],
        "raw_validation": raw_validation,
        "raw_test": raw_test,
        "candidate_holdout_metrics": candidates,
        "calibrated_test": calibrated_test,
        "calibration": final_payload,
    }

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(json.dumps({
        "selected_method": selected["name"],
        "raw_test": raw_test,
        "calibrated_test": calibrated_test,
        "saved": str(output_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
