from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from emotion_schema import AUDIO_LABEL_MAP, EMOTION_LABELS, LABEL_TO_INDEX, make_result, remap_scores


def read_wav_mono(path):
    sample_rate, samples = wavfile.read(str(path))
    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.dtype.kind in {"i", "u"}:
        max_value = np.iinfo(samples.dtype).max
        samples = samples.astype("float32") / max(1, max_value)
    else:
        samples = samples.astype("float32")
    samples = np.nan_to_num(samples)
    return int(sample_rate), samples


def acoustic_features(sample_rate, samples):
    if samples.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "peak": 0.0,
        }

    centered = samples - float(np.mean(samples))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    peak = float(np.max(np.abs(centered)))
    zero_crossings = np.count_nonzero(np.diff(np.signbit(centered)))
    zcr = float(zero_crossings / max(1, centered.size - 1))

    windowed = centered[: min(centered.size, sample_rate * 4)]
    if windowed.size < 8:
        centroid = 0.0
    else:
        spectrum = np.abs(np.fft.rfft(windowed * np.hanning(windowed.size)))
        frequencies = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
        centroid = float((frequencies * spectrum).sum() / max(1e-9, spectrum.sum()))

    return {
        "duration_seconds": float(samples.size / max(1, sample_rate)),
        "rms": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "peak": peak,
    }


class AcousticHeuristicAudioEmotionRecognizer:
    backend = "acoustic_heuristic"

    def predict_file(self, audio_path):
        sample_rate, samples = read_wav_mono(Path(audio_path))
        features = acoustic_features(sample_rate, samples)
        scores = np.full(len(EMOTION_LABELS), 0.06, dtype="float32")

        rms = features["rms"]
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid_hz"]
        duration = features["duration_seconds"]

        if duration < 0.2 or rms < 0.004:
            scores[LABEL_TO_INDEX["neutral"]] += 2.0
            return make_result("audio", self.backend, scores, features)

        energy = min(1.0, rms / 0.12)
        noisiness = min(1.0, zcr / 0.18)
        brightness = min(1.0, centroid / 3500.0)

        scores[LABEL_TO_INDEX["angry"]] += 1.3 * energy + 0.4 * brightness
        scores[LABEL_TO_INDEX["fear"]] += 0.8 * brightness + 0.7 * noisiness
        scores[LABEL_TO_INDEX["happy"]] += 0.8 * energy + 0.3 * brightness
        scores[LABEL_TO_INDEX["surprise"]] += 0.7 * energy + 0.7 * brightness
        scores[LABEL_TO_INDEX["sad"]] += 1.1 * (1.0 - energy) + 0.4 * (1.0 - brightness)
        scores[LABEL_TO_INDEX["neutral"]] += 0.8 * (1.0 - abs(energy - 0.35))
        scores[LABEL_TO_INDEX["disgust"]] += 0.35 * noisiness + 0.2 * (1.0 - brightness)

        return make_result("audio", self.backend, scores, features)


class HuggingFaceAudioEmotionRecognizer:
    backend = "huggingface:audio-classification"

    def __init__(self, model_name="Dpngtm/wav2vec2-emotion-recognition", device=-1, local_files_only=False):
        from transformers import pipeline

        self.model_name = model_name
        self.pipeline = pipeline(
            "audio-classification",
            model=model_name,
            top_k=None,
            device=device,
            model_kwargs={"local_files_only": local_files_only},
        )

    def predict_file(self, audio_path):
        predictions = self.pipeline(str(audio_path))
        scores = {item["label"]: float(item["score"]) for item in predictions}
        probabilities = remap_scores(scores, AUDIO_LABEL_MAP)
        return make_result("audio", f"huggingface:{self.model_name}", probabilities, {"raw_scores": scores})


def make_audio_recognizer(backend="acoustic", **kwargs):
    if backend in {"acoustic", "heuristic"}:
        return AcousticHeuristicAudioEmotionRecognizer()
    if backend == "hf":
        return HuggingFaceAudioEmotionRecognizer(**kwargs)
    raise ValueError(f"Unsupported audio backend: {backend}")
