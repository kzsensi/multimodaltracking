from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(EMOTION_LABELS)}

TEXT_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "annoyance": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "nervousness": "fear",
    "joy": "happy",
    "happy": "happy",
    "happiness": "happy",
    "amusement": "happy",
    "approval": "happy",
    "optimism": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "remorse": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
}

AUDIO_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


@dataclass(frozen=True)
class EmotionResult:
    modality: str
    backend: str
    probabilities: dict[str, float]
    label: str
    confidence: float
    details: dict


def normalize_probabilities(values) -> np.ndarray:
    probabilities = np.asarray(values, dtype="float32")
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities = np.zeros(len(EMOTION_LABELS), dtype="float32")
        probabilities[LABEL_TO_INDEX["neutral"]] = 1.0
        return probabilities
    return probabilities / total


def probabilities_to_dict(values) -> dict[str, float]:
    probabilities = normalize_probabilities(values)
    return {label: float(probabilities[index]) for index, label in enumerate(EMOTION_LABELS)}


def top_emotion(values) -> tuple[str, float]:
    probabilities = normalize_probabilities(values)
    index = int(np.argmax(probabilities))
    return EMOTION_LABELS[index], float(probabilities[index])


def remap_scores(scores: Mapping[str, float], label_map: Mapping[str, str]) -> np.ndarray:
    output = np.zeros(len(EMOTION_LABELS), dtype="float32")
    for raw_label, value in scores.items():
        canonical = label_map.get(str(raw_label).strip().lower())
        if canonical in LABEL_TO_INDEX:
            output[LABEL_TO_INDEX[canonical]] += float(value)
    return normalize_probabilities(output)


def make_result(modality: str, backend: str, probabilities, details: dict | None = None) -> EmotionResult:
    normalized = normalize_probabilities(probabilities)
    label, confidence = top_emotion(normalized)
    return EmotionResult(
        modality=modality,
        backend=backend,
        probabilities=probabilities_to_dict(normalized),
        label=label,
        confidence=confidence,
        details=details or {},
    )
