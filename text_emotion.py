from __future__ import annotations

import re
from collections import Counter

import numpy as np

from emotion_schema import EMOTION_LABELS, LABEL_TO_INDEX, TEXT_LABEL_MAP, make_result, remap_scores


LEXICON = {
    "angry": {
        "angry",
        "anger",
        "annoyed",
        "mad",
        "furious",
        "irritated",
        "hate",
        "hated",
        "rage",
        "frustrated",
    },
    "disgust": {
        "disgust",
        "disgusted",
        "gross",
        "nasty",
        "sick",
        "repulsive",
        "awful",
        "terrible",
    },
    "fear": {
        "fear",
        "afraid",
        "scared",
        "terrified",
        "nervous",
        "worried",
        "anxious",
        "panic",
        "unsafe",
    },
    "happy": {
        "happy",
        "joy",
        "joyful",
        "glad",
        "excited",
        "great",
        "love",
        "loved",
        "awesome",
        "amazing",
        "wonderful",
    },
    "neutral": {
        "ok",
        "okay",
        "fine",
        "normal",
        "average",
        "usual",
        "regular",
    },
    "sad": {
        "sad",
        "sadness",
        "unhappy",
        "cry",
        "crying",
        "depressed",
        "hurt",
        "lonely",
        "upset",
        "sorry",
    },
    "surprise": {
        "surprise",
        "surprised",
        "shocked",
        "wow",
        "unexpected",
        "suddenly",
        "amazed",
        "confused",
        "what",
    },
}


class LexiconTextEmotionRecognizer:
    backend = "lexicon"

    def predict(self, text: str):
        words = re.findall(r"[a-z']+", text.lower())
        counts = Counter(words)
        scores = np.full(len(EMOTION_LABELS), 0.04, dtype="float32")
        scores[LABEL_TO_INDEX["neutral"]] = 0.16

        for label, keywords in LEXICON.items():
            hit_count = sum(counts[word] for word in keywords)
            if hit_count:
                scores[LABEL_TO_INDEX[label]] += float(hit_count)

        punctuation_boost = min(0.5, text.count("!") * 0.08)
        if punctuation_boost:
            scores[LABEL_TO_INDEX["happy"]] += punctuation_boost * 0.45
            scores[LABEL_TO_INDEX["angry"]] += punctuation_boost * 0.35
            scores[LABEL_TO_INDEX["surprise"]] += punctuation_boost * 0.20

        if not text.strip():
            scores = np.zeros(len(EMOTION_LABELS), dtype="float32")
            scores[LABEL_TO_INDEX["neutral"]] = 1.0

        return make_result("text", self.backend, scores, {"word_count": len(words)})


class HuggingFaceTextEmotionRecognizer:
    backend = "huggingface:j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base", device=-1, local_files_only=False):
        from transformers import pipeline

        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            model_kwargs={"local_files_only": local_files_only},
            tokenizer_kwargs={"local_files_only": local_files_only},
        )

    def predict(self, text: str):
        if not text.strip():
            return make_result("text", self.backend, [0, 0, 0, 0, 1, 0, 0], {"empty_text": True})

        predictions = self.pipeline(text)[0]
        scores = {item["label"]: float(item["score"]) for item in predictions}
        probabilities = remap_scores(scores, TEXT_LABEL_MAP)
        return make_result("text", self.backend, probabilities, {"raw_scores": scores})


def make_text_recognizer(backend="lexicon", **kwargs):
    if backend == "lexicon":
        return LexiconTextEmotionRecognizer()
    if backend == "hf":
        return HuggingFaceTextEmotionRecognizer(**kwargs)
    raise ValueError(f"Unsupported text backend: {backend}")
