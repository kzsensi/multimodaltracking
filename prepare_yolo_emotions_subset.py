import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create balanced cropped FER folders from AdamCodd/yolo-emotions.")
    parser.add_argument(
        "--zip",
        default="data_sources/yolo-emotions/emotions_dataset.zip",
        help="Path to emotions_dataset.zip.",
    )
    parser.add_argument("--output", default="dataset_yolo_emotions_balanced", help="Output ImageFolder root.")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.12, help="Extra margin around YOLO face boxes.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_key(split, stem):
    return f"{split}/images/{stem}"


def build_image_index(names):
    index = {}
    for name in names:
        lowered = name.lower()
        if not any(lowered.endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue
        parts = name.split("/")
        if len(parts) == 3 and parts[1] == "images":
            split, _, filename = parts
            stem = Path(filename).stem
            index[image_key(split, stem)] = name
    return index


def collect_candidates(zip_file, rng):
    names = zip_file.namelist()
    image_index = build_image_index(names)
    candidates = defaultdict(list)
    missing_images = 0
    bad_lines = 0

    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or parts[1] != "labels" or not name.lower().endswith(".txt"):
            continue
        split, _, filename = parts
        if split not in {"train", "val", "test"}:
            continue
        stem = Path(filename).stem
        image_name = image_index.get(image_key(split, stem))
        if image_name is None:
            missing_images += 1
            continue

        text = zip_file.read(name).decode("utf-8", errors="ignore")
        for line_index, raw_line in enumerate(text.splitlines()):
            values = raw_line.strip().split()
            if len(values) < 5:
                continue
            try:
                class_id = int(float(values[0]))
                x_center, y_center, width, height = [float(value) for value in values[1:5]]
            except ValueError:
                bad_lines += 1
                continue
            if class_id < 0 or class_id >= len(LABELS):
                bad_lines += 1
                continue
            if width <= 0 or height <= 0:
                bad_lines += 1
                continue
            candidates[(split, class_id)].append(
                {
                    "image": image_name,
                    "label": name,
                    "line_index": line_index,
                    "bbox": [x_center, y_center, width, height],
                }
            )

    for items in candidates.values():
        rng.shuffle(items)

    return candidates, {"missing_images": missing_images, "bad_lines": bad_lines}


def crop_box(image, bbox, margin):
    x_center, y_center, width, height = bbox
    image_width, image_height = image.size
    box_width = width * image_width
    box_height = height * image_height
    extra_w = box_width * margin
    extra_h = box_height * margin
    left = max(0, int(round((x_center * image_width) - (box_width / 2) - extra_w)))
    top = max(0, int(round((y_center * image_height) - (box_height / 2) - extra_h)))
    right = min(image_width, int(round((x_center * image_width) + (box_width / 2) + extra_w)))
    bottom = min(image_height, int(round((y_center * image_height) + (box_height / 2) + extra_h)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def save_crop(zip_file, item, destination, image_size, margin):
    with zip_file.open(item["image"]) as image_file:
        image = Image.open(image_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        cropped = crop_box(image, item["bbox"], margin)
        if cropped is None:
            return False
        cropped = ImageOps.fit(cropped, (image_size, image_size), method=Image.Resampling.BILINEAR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(destination, format="JPEG", quality=92, optimize=True)
        return True


def export_split(zip_file, candidates, output_root, source_split, target_split, per_class, image_size, margin):
    counts = Counter()
    skipped = Counter()
    if per_class <= 0:
        return counts, skipped

    for class_id, label in enumerate(LABELS):
        items = candidates.get((source_split, class_id), [])
        for item in items:
            if counts[label] >= per_class:
                break
            source_stem = Path(item["image"]).stem
            filename = f"{source_stem}_{item['line_index']:02d}.jpg"
            destination = output_root / target_split / label / filename
            try:
                ok = save_crop(zip_file, item, destination, image_size, margin)
            except Exception:
                ok = False
            if ok:
                counts[label] += 1
            else:
                skipped[label] += 1
    return counts, skipped


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    zip_path = Path(args.zip)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zip_file:
        candidates, scan_issues = collect_candidates(zip_file, rng)
        train_counts, train_skipped = export_split(
            zip_file,
            candidates,
            output_root,
            "train",
            "train",
            args.train_per_class,
            args.image_size,
            args.margin,
        )
        val_counts, val_skipped = export_split(
            zip_file,
            candidates,
            output_root,
            "val",
            "validation",
            args.val_per_class,
            args.image_size,
            args.margin,
        )
        test_counts, test_skipped = export_split(
            zip_file,
            candidates,
            output_root,
            "test",
            "test",
            args.test_per_class,
            args.image_size,
            args.margin,
        )

    payload = {
        "source_zip": str(zip_path),
        "labels": LABELS,
        "image_size": args.image_size,
        "margin": args.margin,
        "requested_per_class": {
            "train": args.train_per_class,
            "validation": args.val_per_class,
            "test": args.test_per_class,
        },
        "counts": {
            "train": dict(train_counts),
            "validation": dict(val_counts),
            "test": dict(test_counts),
        },
        "skipped": {
            "train": dict(train_skipped),
            "validation": dict(val_skipped),
            "test": dict(test_skipped),
        },
        "scan_issues": scan_issues,
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
