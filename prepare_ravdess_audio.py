from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


EMOTION_ID_TO_LABEL = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RAVDESS audio speech into train/validation/test folders.")
    parser.add_argument(
        "--zip",
        default="data_sources/RAVDESS/Audio_Speech_Actors_01-24.zip",
        help="Path to Audio_Speech_Actors_01-24.zip.",
    )
    parser.add_argument("--output", default="dataset_ravdess_audio", help="Output dataset folder.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_for_actor(actor_id):
    actor = int(actor_id)
    if actor <= 18:
        return "train"
    if actor <= 21:
        return "validation"
    return "test"


def parse_ravdess_filename(path):
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    modality, vocal_channel, emotion_id, intensity, statement, repetition, actor_id = parts
    if modality != "03" or vocal_channel != "01":
        return None
    label = EMOTION_ID_TO_LABEL.get(emotion_id)
    if label is None:
        return None
    return {
        "label": label,
        "split": split_for_actor(actor_id),
        "actor_id": actor_id,
        "emotion_id": emotion_id,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
    }


def main():
    args = parse_args()
    zip_path = Path(args.zip)
    output = Path(args.output)
    if not zip_path.exists():
        raise FileNotFoundError(f"RAVDESS zip not found: {zip_path}")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".wav"):
                continue
            parsed = parse_ravdess_filename(member)
            if parsed is None:
                skipped += 1
                continue
            destination = output / parsed["split"] / parsed["label"] / Path(member).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            counts[f"{parsed['split']}/{parsed['label']}"] += 1

    for split in ["train", "validation", "test"]:
        for label in LABELS:
            (output / split / label).mkdir(parents=True, exist_ok=True)

    summary = {
        "source_zip": str(zip_path),
        "output": str(output),
        "labels": LABELS,
        "split_rule": "actors 01-18 train, 19-21 validation, 22-24 test",
        "counts": dict(sorted(counts.items())),
        "skipped_files": skipped,
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
