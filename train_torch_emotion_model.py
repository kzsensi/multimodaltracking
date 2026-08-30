import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="GPU PyTorch trainer for seven-class facial emotion recognition.")
    parser.add_argument("--train-dir", required=True, help="ImageFolder training directory.")
    parser.add_argument("--validation-dir", required=True, help="ImageFolder validation directory.")
    parser.add_argument("--test-dir", default="dataset_mutfer2024/test", help="ImageFolder test directory.")
    parser.add_argument("--extra-train-dir", nargs="*", default=[], help="Optional extra ImageFolder train directories.")
    parser.add_argument("--reports", default="reports/torch_yolo_resnet18_latest")
    parser.add_argument("--model-output", default="models/torch/yolo_resnet18_latest.pt")
    parser.add_argument("--initial-checkpoint", help="Optional checkpoint to continue fine-tuning from.")
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weight-multiplier",
        nargs="*",
        default=[],
        metavar="LABEL=VALUE",
        help="Optional per-class loss multipliers, e.g. sad=1.2 fear=1.2.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for reliability.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use weighted train sampling.")
    parser.add_argument("--augmentation", choices=["light", "rich"], default="light")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print train progress every N batches.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transform(image_size, augmentation):
    if augmentation == "light":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_folder(path, transform):
    dataset = datasets.ImageFolder(path, transform=transform)
    if dataset.classes != LABELS:
        raise ValueError(f"{path} classes are {dataset.classes}; expected {LABELS}")
    return dataset


def make_train_dataset(args):
    roots = [args.train_dir] + list(args.extra_train_dir)
    parts = [load_image_folder(root, train_transform(args.image_size, args.augmentation)) for root in roots]
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)


def targets_of(dataset):
    if isinstance(dataset, ConcatDataset):
        targets = []
        for part in dataset.datasets:
            targets.extend(part.targets)
        return targets
    return list(dataset.targets)


def make_loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_sampler(dataset):
    targets = targets_of(dataset)
    counts = Counter(targets)
    weights = [1.0 / counts[target] for target in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_model(arch, num_classes):
    efficientnet_factories = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
    }
    if arch in efficientnet_factories:
        factory, weights = efficientnet_factories[arch]
        model = factory(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model, arch):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if arch.startswith("efficientnet_"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.fc.parameters():
            parameter.requires_grad = True


def parse_class_weight_multipliers(values):
    multipliers = {label: 1.0 for label in LABELS}
    for raw_value in values:
        if "=" not in raw_value:
            raise ValueError(f"Invalid class weight multiplier {raw_value!r}; expected LABEL=VALUE.")
        label, value = raw_value.split("=", 1)
        if label not in multipliers:
            raise ValueError(f"Unknown class label {label!r}; expected one of {LABELS}.")
        multipliers[label] = float(value)
    return multipliers


def class_weights(dataset, device, multipliers):
    targets = targets_of(dataset)
    counts = Counter(targets)
    total = sum(counts.values())
    weights = []
    for idx in range(len(LABELS)):
        label = LABELS[idx]
        weights.append((total / (len(LABELS) * max(1, counts[idx]))) * multipliers[label])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_initial_checkpoint(model, checkpoint_path, arch, device):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    labels = list(checkpoint["labels"])
    checkpoint_arch = checkpoint.get("arch")
    if labels != LABELS:
        raise ValueError(f"Initial checkpoint labels are {labels}; expected {LABELS}")
    if checkpoint_arch != arch:
        raise ValueError(f"Initial checkpoint arch is {checkpoint_arch}; requested {arch}")
    model.load_state_dict(checkpoint["model_state"])


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    probabilities = []
    total_seconds = 0.0
    total_images = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            started_at = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - started_at
            total_images += int(images.shape[0])
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities.append(probs)
            y_true.extend(labels.numpy().tolist())
    probabilities = np.concatenate(probabilities, axis=0)
    y_true = np.asarray(y_true, dtype="int64")
    y_pred = np.argmax(probabilities, axis=1)
    return {
        "y_true": y_true,
        "probabilities": probabilities,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "seconds": float(total_seconds),
        "images_per_second": float(total_images / total_seconds) if total_seconds > 0 else 0.0,
    }


def plot_class_distribution(dataset, reports_dir):
    counts = Counter(targets_of(dataset))
    plt.figure(figsize=(10, 5))
    plt.bar(LABELS, [counts[idx] for idx in range(len(LABELS))], color="#2878b5")
    plt.title("Training Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution_bar_chart.png", dpi=160)
    plt.close()


def plot_training(history, reports_dir):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["validation_loss"] for row in history], label="validation loss")
    plt.plot(epochs, [row["validation_accuracy"] for row in history], label="validation accuracy")
    plt.plot(epochs, [row["validation_macro_f1"] for row in history], label="validation macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "training_curves.png", dpi=160)
    plt.close()


def plot_efficiency(history, test_metrics, reports_dir):
    epochs = [row["epoch"] for row in history]
    seconds = [row["epoch_seconds"] for row in history]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, seconds, marker="o", color="#c44536", label="seconds per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Seconds per epoch", color="#c44536")
    ax1.tick_params(axis="y", labelcolor="#c44536")
    ax2 = ax1.twinx()
    ax2.axhline(test_metrics["images_per_second"], color="#2878b5", label="test images/sec")
    ax2.set_ylabel("Inference images per second", color="#2878b5")
    ax2.tick_params(axis="y", labelcolor="#2878b5")
    plt.title("Training and Inference Efficiency")
    fig.tight_layout()
    plt.savefig(reports_dir / "efficiency_graph.png", dpi=160)
    plt.close()


def plot_confusion(y_true, probabilities, reports_dir, split_name):
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{split_name.title()} Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, LABELS, rotation=45, ha="right")
    plt.yticks(ticks, LABELS)
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
    plt.savefig(reports_dir / f"{split_name}_confusion_matrix.png", dpi=160)
    plt.close()


def plot_auc(y_true, probabilities, reports_dir, split_name):
    y_one_hot = np.eye(len(LABELS), dtype="float64")[y_true]
    auc_scores = {}
    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(LABELS):
        if len(np.unique(y_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probabilities[:, idx])
        auc_value = roc_auc_score(y_one_hot[:, idx], probabilities[:, idx])
        auc_scores[label] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title(f"{split_name.title()} One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if auc_scores:
        plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / f"{split_name}_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def save_history(history, reports_dir):
    with (reports_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(args):
    set_seed(args.seed)
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = make_train_dataset(args)
    validation_dataset = load_image_folder(args.validation_dir, eval_transform(args.image_size))
    test_dataset = load_image_folder(args.test_dir, eval_transform(args.image_size))

    sampler = make_sampler(train_dataset) if args.balanced_sampler else None
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)
    validation_loader = make_loader(validation_dataset, args.batch_size, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)

    model = make_model(args.arch, len(LABELS)).to(device)
    load_initial_checkpoint(model, args.initial_checkpoint, args.arch, device)
    if args.freeze_backbone:
        freeze_backbone(model, args.arch)
    weight_multipliers = parse_class_weight_multipliers(args.class_weight_multiplier)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, device, weight_multipliers),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    plot_class_distribution(train_dataset, reports_dir)
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        started_at = time.perf_counter()
        model.train()
        losses = []
        total_batches = len(train_loader)
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if args.progress_every > 0 and batch_index % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": total_batches,
                            "train_loss": float(np.mean(losses[-args.progress_every :])),
                        }
                    ),
                    flush=True,
                )

        validation_metrics = evaluate(model, validation_loader, device)
        validation_loss = 0.0
        model.eval()
        with torch.inference_mode():
            loss_values = []
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss_values.append(float(loss_fn(logits, labels).detach().cpu()))
            validation_loss = float(np.mean(loss_values)) if loss_values else 0.0

        epoch_seconds = time.perf_counter() - started_at
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "epoch_seconds": float(epoch_seconds),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "labels": LABELS,
                    "image_size": args.image_size,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "validation_metrics": validation_metrics,
                },
                model_output,
            )

    checkpoint = torch.load(model_output, map_location=device, weights_only=False)
    model = make_model(checkpoint["arch"], len(checkpoint["labels"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    validation_metrics = evaluate(model, validation_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    save_history(history, reports_dir)
    plot_training(history, reports_dir)
    plot_efficiency(history, test_metrics, reports_dir)
    validation_auc = plot_auc(validation_metrics["y_true"], validation_metrics["probabilities"], reports_dir, "validation")
    test_auc = plot_auc(test_metrics["y_true"], test_metrics["probabilities"], reports_dir, "test")
    plot_confusion(validation_metrics["y_true"], validation_metrics["probabilities"], reports_dir, "validation")
    plot_confusion(test_metrics["y_true"], test_metrics["probabilities"], reports_dir, "test")

    np.savez_compressed(
        reports_dir / "test_predictions.npz",
        y_true=test_metrics["y_true"],
        probabilities=test_metrics["probabilities"],
        labels=np.asarray(LABELS),
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch": args.arch,
        "labels": LABELS,
        "train_dir": args.train_dir,
        "extra_train_dir": args.extra_train_dir,
        "validation_dir": args.validation_dir,
        "test_dir": args.test_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augmentation": args.augmentation,
        "freeze_backbone": bool(args.freeze_backbone),
        "initial_checkpoint": args.initial_checkpoint,
        "label_smoothing": args.label_smoothing,
        "class_weight_multiplier": weight_multipliers,
        "model_output": str(model_output),
        "best_validation_macro_f1": float(best_macro_f1),
        "validation": {
            "accuracy": validation_metrics["accuracy"],
            "macro_f1": validation_metrics["macro_f1"],
            "auc_one_vs_rest": validation_auc,
            "classification_report": classification_report(
                validation_metrics["y_true"],
                np.argmax(validation_metrics["probabilities"], axis=1),
                labels=list(range(len(LABELS))),
                target_names=LABELS,
                output_dict=True,
                zero_division=0,
            ),
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "images_per_second": test_metrics["images_per_second"],
            "auc_one_vs_rest": test_auc,
            "classification_report": classification_report(
                test_metrics["y_true"],
                np.argmax(test_metrics["probabilities"], axis=1),
                labels=list(range(len(LABELS))),
                target_names=LABELS,
                output_dict=True,
                zero_division=0,
            ),
        },
        "history": history,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "model_output": str(model_output.resolve()),
                "reports": str(reports_dir.resolve()),
            },
            indent=2,
        )
    )


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
