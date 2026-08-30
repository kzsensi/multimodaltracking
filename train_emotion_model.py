import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SPLIT_DIR_NAMES = {
    "train": ["train", "training", "FER2013Train"],
    "validation": ["validation", "valid", "val", "dev", "FER2013Valid"],
    "test": ["test", "testing", "FER2013Test"],
}


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_seconds = []
        self._started_at = None

    def on_epoch_begin(self, epoch, logs=None):
        self._started_at = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._started_at)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a facial emotion CNN.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root folder.")
    parser.add_argument("--model", default="emotion_model.h5", help="Path for the best saved model.")
    parser.add_argument("--labels", default="labels.json", help="Path for model label metadata.")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path for model training metadata.")
    parser.add_argument("--reports", default="reports", help="Directory for metrics and graphs.")
    parser.add_argument("--image-size", type=int, default=48, help="Square image size used by the model.")
    parser.add_argument(
        "--color-mode",
        choices=["grayscale", "rgb"],
        default="grayscale",
        help="Image color mode for training.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of epochs.")
    parser.add_argument(
        "--architecture",
        choices=["simple_cnn", "mobilenet_v2"],
        default="simple_cnn",
        help="Model architecture.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split when no val folder exists.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training image augmentation.")
    return parser.parse_args()


def find_split_dir(dataset_root, split_name):
    for name in SPLIT_DIR_NAMES[split_name]:
        candidate = dataset_root / name
        if candidate.is_dir() and has_class_folders(candidate):
            return candidate
    return None


def has_class_folders(path):
    return any(child.is_dir() for child in path.iterdir()) if path.exists() else False


def detect_dataset_layout(dataset_root):
    train_dir = find_split_dir(dataset_root, "train")
    val_dir = find_split_dir(dataset_root, "validation")
    test_dir = find_split_dir(dataset_root, "test")

    if train_dir:
        return {
            "layout": "split-folders",
            "train": train_dir,
            "validation": val_dir,
            "test": test_dir,
        }

    if has_class_folders(dataset_root):
        return {
            "layout": "single-folder-with-validation-split",
            "train": dataset_root,
            "validation": None,
            "test": None,
        }

    raise FileNotFoundError(
        "No emotion folders found. Expected either dataset/<emotion>/... or "
        "dataset/train/<emotion>/... ."
    )


def build_datagens(args):
    if args.architecture == "mobilenet_v2":
        rescale = None
        preprocessing_function = mobilenet_v2_preprocess
    else:
        rescale = 1.0 / 255
        preprocessing_function = None

    if args.no_augmentation:
        train_datagen = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocessing_function)
    else:
        train_datagen = ImageDataGenerator(
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            rotation_range=12,
            width_shift_range=0.10,
            height_shift_range=0.10,
            zoom_range=0.10,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    eval_datagen = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocessing_function)
    split_train_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        rotation_range=12 if not args.no_augmentation else 0,
        width_shift_range=0.10 if not args.no_augmentation else 0,
        height_shift_range=0.10 if not args.no_augmentation else 0,
        zoom_range=0.10 if not args.no_augmentation else 0,
        horizontal_flip=not args.no_augmentation,
        fill_mode="nearest",
        validation_split=args.validation_split,
    )
    split_eval_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        validation_split=args.validation_split,
    )
    return train_datagen, eval_datagen, split_train_datagen, split_eval_datagen


def make_generator(datagen, directory, args, shuffle, subset=None):
    return datagen.flow_from_directory(
        directory,
        target_size=(args.image_size, args.image_size),
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=shuffle,
        subset=subset,
        seed=args.seed,
    )


def load_generators(args):
    dataset_root = Path(args.dataset)
    layout = detect_dataset_layout(dataset_root)
    train_datagen, eval_datagen, split_train_datagen, split_eval_datagen = build_datagens(args)

    if layout["layout"] == "split-folders":
        train_data = make_generator(train_datagen, layout["train"], args, shuffle=True)
        if layout["validation"]:
            val_data = make_generator(eval_datagen, layout["validation"], args, shuffle=False)
        else:
            val_data = make_generator(split_eval_datagen, layout["train"], args, shuffle=False, subset="validation")
            train_data = make_generator(split_train_datagen, layout["train"], args, shuffle=True, subset="training")
    else:
        train_data = make_generator(split_train_datagen, layout["train"], args, shuffle=True, subset="training")
        val_data = make_generator(split_eval_datagen, layout["train"], args, shuffle=False, subset="validation")

    test_data = None
    if layout["test"]:
        test_data = make_generator(eval_datagen, layout["test"], args, shuffle=False)

    return layout, train_data, val_data, test_data


def build_simple_cnn(image_size, channels, num_classes):
    return Sequential(
        [
            Input(shape=(image_size, image_size, channels)),
            Conv2D(32, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )


def build_mobilenet_v2(image_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    return Sequential(
        [
            Input(shape=(image_size, image_size, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.35),
            Dense(256, activation="relu"),
            Dropout(0.30),
            Dense(num_classes, activation="softmax"),
        ]
    )


def build_model(args, channels, num_classes):
    if args.architecture == "mobilenet_v2":
        if args.color_mode != "rgb":
            raise ValueError("MobileNetV2 requires --color-mode rgb.")
        if args.image_size < 96:
            raise ValueError("Use --image-size 96 or larger for MobileNetV2.")
        return build_mobilenet_v2(args.image_size, num_classes)
    return build_simple_cnn(args.image_size, channels, num_classes)


def default_learning_rate(args):
    if args.learning_rate is not None:
        return args.learning_rate
    if args.architecture == "mobilenet_v2":
        return 0.0003
    return 0.001


def ordered_class_names(class_indices):
    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def assert_matching_classes(reference, candidate, name):
    if candidate is None:
        return
    if reference.class_indices != candidate.class_indices:
        raise ValueError(
            f"{name} classes do not match training classes.\n"
            f"Train: {reference.class_indices}\n{name}: {candidate.class_indices}"
        )


def compute_weights(train_data):
    classes = np.unique(train_data.classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_data.classes)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def predict_generator(model, generator):
    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    y_true = generator.classes[: len(probabilities)]
    y_pred = np.argmax(probabilities, axis=1)
    return y_true, y_pred, probabilities


def plot_class_distribution(generator, class_names, reports_dir):
    counts = np.bincount(generator.classes, minlength=len(class_names))
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts, color="#2878b5")
    plt.title("Training Class Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution_bar_chart.png", dpi=160)
    plt.close()


def plot_learning_curves(history, reports_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("accuracy", []), label="train accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="validation accuracy")
    plt.plot(history.history.get("loss", []), label="train loss")
    plt.plot(history.history.get("val_loss", []), label="validation loss")
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "training_curves.png", dpi=160)
    plt.close()


def plot_confusion(y_true, y_pred, class_names, reports_dir, split_name):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{split_name.title()} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{split_name}_confusion_matrix.png", dpi=160)
    plt.close()


def plot_auc(y_true, probabilities, class_names, reports_dir, split_name):
    y_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    plt.figure(figsize=(9, 7))
    auc_scores = {}
    for idx, class_name in enumerate(class_names):
        if len(np.unique(y_one_hot[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_one_hot[:, idx], probabilities[:, idx])
        auc_value = roc_auc_score(y_one_hot[:, idx], probabilities[:, idx])
        auc_scores[class_name] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.title(f"{split_name.title()} One-vs-Rest AUC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(reports_dir / f"{split_name}_auc_curve.png", dpi=160)
    plt.close()
    return auc_scores


def plot_efficiency(epoch_seconds, train_samples, reports_dir):
    epochs = np.arange(1, len(epoch_seconds) + 1)
    samples_per_second = [train_samples / seconds if seconds > 0 else 0 for seconds in epoch_seconds]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, epoch_seconds, marker="o", color="#c44536", label="seconds per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Seconds per epoch", color="#c44536")
    ax1.tick_params(axis="y", labelcolor="#c44536")

    ax2 = ax1.twinx()
    ax2.bar(epochs, samples_per_second, alpha=0.25, color="#2878b5", label="samples per second")
    ax2.set_ylabel("Samples per second", color="#2878b5")
    ax2.tick_params(axis="y", labelcolor="#2878b5")

    plt.title("Training Efficiency")
    fig.tight_layout()
    plt.savefig(reports_dir / "efficiency_graph.png", dpi=160)
    plt.close()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def evaluate_and_report(model, generator, class_names, reports_dir, split_name):
    y_true, y_pred, probabilities = predict_generator(model, generator)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    auc_scores = plot_auc(y_true, probabilities, class_names, reports_dir, split_name)
    plot_confusion(y_true, y_pred, class_names, reports_dir, split_name)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "auc_one_vs_rest": auc_scores,
    }


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    layout, train_data, val_data, test_data = load_generators(args)
    assert_matching_classes(train_data, val_data, "Validation")
    assert_matching_classes(train_data, test_data, "Test")

    class_names = ordered_class_names(train_data.class_indices)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("At least two emotion class folders are required.")

    channels = 1 if args.color_mode == "grayscale" else 3
    model = build_model(args, channels, num_classes)
    learning_rate = default_learning_rate(args)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    class_weight = compute_weights(train_data)
    epoch_timer = EpochTimer()
    callbacks = [
        epoch_timer,
        ModelCheckpoint(args.model, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        CSVLogger(str(reports_dir / "training_history.csv")),
    ]

    started_at = time.perf_counter()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    total_train_seconds = time.perf_counter() - started_at

    model_path = Path(args.model)
    if model_path.exists():
        best_model = load_model(model_path)
    else:
        model.save(model_path)
        best_model = model

    plot_class_distribution(train_data, class_names, reports_dir)
    plot_learning_curves(history, reports_dir)
    plot_efficiency(epoch_timer.epoch_seconds, train_data.samples, reports_dir)

    metrics = {
        "validation": evaluate_and_report(best_model, val_data, class_names, reports_dir, "validation")
    }
    if test_data is not None:
        metrics["test"] = evaluate_and_report(best_model, test_data, class_names, reports_dir, "test")

    label_metadata = {
        "labels": class_names,
        "class_indices": train_data.class_indices,
        "image_size": args.image_size,
        "color_mode": args.color_mode,
        "architecture": args.architecture,
        "preprocessing": "mobilenet_v2" if args.architecture == "mobilenet_v2" else "rescale_1_over_255",
    }
    write_json(args.labels, label_metadata)

    training_metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": str(Path(args.dataset).resolve()),
        "layout": {
            key: str(value.resolve()) if isinstance(value, Path) else value for key, value in layout.items()
        },
        "model_path": str(Path(args.model).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "image_size": args.image_size,
        "color_mode": args.color_mode,
        "architecture": args.architecture,
        "learning_rate": learning_rate,
        "preprocessing": "mobilenet_v2" if args.architecture == "mobilenet_v2" else "rescale_1_over_255",
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "class_names": class_names,
        "class_weight": class_weight,
        "train_samples": int(train_data.samples),
        "validation_samples": int(val_data.samples),
        "test_samples": int(test_data.samples) if test_data is not None else 0,
        "total_train_seconds": float(total_train_seconds),
        "epoch_seconds": [float(value) for value in epoch_timer.epoch_seconds],
        "metrics": metrics,
    }
    write_json(args.metadata, training_metadata)
    write_json(reports_dir / "metrics.json", metrics)

    print(f"Saved model: {Path(args.model).resolve()}")
    print(f"Saved labels: {Path(args.labels).resolve()}")
    print(f"Saved reports: {reports_dir.resolve()}")
    print(f"Classes ({num_classes}): {', '.join(class_names)}")


if __name__ == "__main__":
    main()
