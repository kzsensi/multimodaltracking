import argparse
import json
import random
import shutil
from pathlib import Path


LABEL_MAP = {
    "Angry": "angry",
    "Disgusted": "disgust",
    "Fearful": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprise",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MUTFER2024 for folder-based training.")
    parser.add_argument("--source", default="MUTFER2024", help="Downloaded MUTFER2024 folder.")
    parser.add_argument("--output", default="dataset_mutfer2024", help="Prepared dataset output folder.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    return parser.parse_args()


def image_files(path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in extensions)


def validate_args(args):
    total = args.train_ratio + args.validation_ratio
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1.")
    if total >= 1:
        raise ValueError("--train-ratio + --validation-ratio must be less than 1.")


def copy_split(files, output_root, label, train_end, validation_end):
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for index, source_file in enumerate(files):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"

        target_dir = output_root / split / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / source_file.name
        shutil.copy2(source_file, target_file)
        split_counts[split] += 1
    return split_counts


def main():
    args = parse_args()
    validate_args(args)

    source_root = Path(args.source)
    output_root = Path(args.output)

    if not source_root.exists():
        raise FileNotFoundError(f"Source folder not found: {source_root}")
    if output_root.exists():
        raise FileExistsError(
            f"Output folder already exists: {output_root}. "
            "Rename it or remove it manually before preparing again."
        )

    rng = random.Random(args.seed)
    summary = {
        "source": str(source_root.resolve()),
        "output": str(output_root.resolve()),
        "seed": args.seed,
        "splits": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": round(1.0 - args.train_ratio - args.validation_ratio, 6),
        },
        "classes": {},
    }

    for source_label, target_label in LABEL_MAP.items():
        source_dir = source_root / source_label
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {source_dir}")

        files = image_files(source_dir)
        if not files:
            raise FileNotFoundError(f"No images found in {source_dir}")

        rng.shuffle(files)
        train_end = int(len(files) * args.train_ratio)
        validation_end = train_end + int(len(files) * args.validation_ratio)
        split_counts = copy_split(files, output_root, target_label, train_end, validation_end)
        summary["classes"][target_label] = {
            "source_label": source_label,
            "total": len(files),
            **split_counts,
        }

    summary_path = output_root / "mutfer2024_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared dataset: {output_root.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    for label, counts in summary["classes"].items():
        print(
            f"{label}: total={counts['total']} "
            f"train={counts['train']} validation={counts['validation']} test={counts['test']}"
        )


if __name__ == "__main__":
    main()
