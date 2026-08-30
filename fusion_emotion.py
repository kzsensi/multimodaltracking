from __future__ import annotations

from dataclasses import asdict

import numpy as np

from emotion_schema import EMOTION_LABELS, EmotionResult, make_result


DEFAULT_WEIGHTS = {
    "visual": 0.50,
    "audio": 0.30,
    "text": 0.20,
}


def result_to_vector(result: EmotionResult) -> np.ndarray:
    return np.asarray([result.probabilities[label] for label in EMOTION_LABELS], dtype="float32")


def fuse_results(results, weights=None):
    weights = dict(weights or DEFAULT_WEIGHTS)
    usable = [result for result in results if result is not None]
    if not usable:
        return make_result("fusion", "weighted_late_fusion", [0, 0, 0, 0, 1, 0, 0], {"missing_all": True})

    active_weight_sum = sum(float(weights.get(result.modality, 0.0)) for result in usable)
    if active_weight_sum <= 0.0:
        active_weight_sum = float(len(usable))
        weights = {result.modality: 1.0 for result in usable}

    fused = np.zeros(len(EMOTION_LABELS), dtype="float32")
    used_weights = {}
    for result in usable:
        raw_weight = float(weights.get(result.modality, 0.0))
        normalized_weight = raw_weight / active_weight_sum
        used_weights[result.modality] = normalized_weight
        fused += normalized_weight * result_to_vector(result)

    return make_result(
        "fusion",
        "weighted_late_fusion",
        fused,
        {
            "weights": used_weights,
            "inputs": [asdict(result) for result in usable],
        },
    )
