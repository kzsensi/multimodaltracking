import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


FALLBACK_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
HSEMOTION_LABEL_MAP = {
    "Anger": "angry",
    "Contempt": "contempt",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time webcam facial emotion recognition.")
    parser.add_argument(
        "--backend",
        choices=["keras", "hsemotion", "hsemotion_ensemble", "torch"],
        default="torch",
        help="Inference backend. Use torch for the latest GPU-trained PyTorch model.",
    )
    parser.add_argument("--model", default="emotion_model.h5", help="Trained Keras model path.")
    parser.add_argument(
        "--torch-model",
        default="models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt",
        help="Trained PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--torch-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch inference device.",
    )
    parser.add_argument("--labels", default="labels.json", help="Label metadata JSON from training.")
    parser.add_argument(
        "--hsemotion-model",
        default="enet_b2_7",
        help="HSEmotion ONNX model name. enet_b2_7 is the recommended 7-class model.",
    )
    parser.add_argument(
        "--hsemotion-bgr-input",
        action="store_true",
        help="Use raw OpenCV BGR crops for HSEmotion instead of the measured-better RGB conversion.",
    )
    parser.add_argument(
        "--ensemble-artifact",
        default="models/ensembles/hsemotion_stacked_ml_latest.joblib",
        help="Saved ensemble stacker artifact from ensemble_hsemotion_predictions.py.",
    )
    parser.add_argument(
        "--detector",
        choices=["yunet", "haar"],
        default="yunet",
        help="Face detector. YuNet is recommended; Haar is kept as a fallback.",
    )
    parser.add_argument(
        "--yunet-model",
        default="models/yunet/face_detection_yunet_2023mar.onnx",
        help="Path to the OpenCV YuNet face detection ONNX model.",
    )
    parser.add_argument("--face-margin", type=float, default=0.0, help="Extra crop margin around detected faces.")
    parser.add_argument("--calibration", default=None, help="Optional probability calibration JSON.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--confidence", type=float, default=0.40, help="Minimum confidence before showing a class.")
    parser.add_argument("--smoothing", type=float, default=0.65, help="EMA smoothing weight for previous prediction.")
    parser.add_argument("--show-probs", action="store_true", help="Draw top prediction probabilities.")
    return parser.parse_args()


def load_labels(labels_path, expected_outputs):
    path = Path(labels_path)
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        labels = metadata.get("labels") or metadata.get("class_names")
    else:
        metadata = {}
        labels = FALLBACK_LABELS

    if not labels:
        raise ValueError(f"No labels found in {path}.")
    if len(labels) != expected_outputs:
        raise ValueError(
            f"Model outputs {expected_outputs} classes, but {path} defines {len(labels)} labels: {labels}"
        )
    return labels, metadata


def load_calibration(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("calibration", payload)


def normalize_probabilities(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype="float32"), 1e-12, None)
    return probabilities / probabilities.sum()


def apply_calibration(probabilities, calibration):
    if calibration is None:
        return probabilities

    probabilities = normalize_probabilities(probabilities)
    method = calibration.get("method")
    if method == "prior_bias":
        scores = np.log(probabilities) + np.asarray(calibration["bias"], dtype="float32")
    elif method == "logistic_regression":
        feature_type = calibration.get("feature_type", "probabilities")
        if feature_type == "logits":
            features = np.log(probabilities)
        else:
            features = probabilities
        coef = np.asarray(calibration["coef"], dtype="float32")
        intercept = np.asarray(calibration["intercept"], dtype="float32")
        scores = features @ coef.T + intercept
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    scores = scores - scores.max()
    exp_values = np.exp(scores)
    return exp_values / exp_values.sum()


def stacked_features(probability_items):
    parts = []
    for probabilities in probability_items:
        probabilities = normalize_probabilities(probabilities)
        parts.append(probabilities)
        parts.append(np.log(probabilities))
    return np.concatenate(parts, axis=0).reshape(1, -1)


def preprocess_face(frame, gray_frame, box, image_size, color_mode, preprocessing):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

    x, y, w, h = box
    if color_mode == "rgb":
        face = frame[y : y + h, x : x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    else:
        face = gray_frame[y : y + h, x : x + w]
    face = cv2.resize(face, (image_size, image_size))
    face = face.astype("float32")
    if preprocessing == "mobilenet_v2":
        face = mobilenet_v2_preprocess(face)
    else:
        face = face / 255.0
    channels = 3 if color_mode == "rgb" else 1
    return face.reshape(1, image_size, image_size, channels)


class KerasEmotionBackend:
    def __init__(self, model_path, labels_path, calibration=None):
        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        output_shape = self.model.output_shape[-1]
        input_shape = self.model.input_shape
        self.image_size = int(input_shape[1]) if input_shape and input_shape[1] else 48
        self.labels, label_metadata = load_labels(labels_path, output_shape)
        model_channels = int(input_shape[-1]) if input_shape and input_shape[-1] else 1
        self.color_mode = label_metadata.get("color_mode") or ("rgb" if model_channels == 3 else "grayscale")
        self.preprocessing = label_metadata.get("preprocessing", "rescale_1_over_255")
        self.calibration = calibration
        self.description = f"Keras {self.image_size}x{self.image_size} {self.color_mode}"

    def predict(self, frame, gray_frame, box):
        face = preprocess_face(frame, gray_frame, box, self.image_size, self.color_mode, self.preprocessing)
        return apply_calibration(self.model.predict(face, verbose=0)[0], self.calibration)


class HSEmotionBackend:
    def __init__(self, model_name, use_rgb=True, calibration=None):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self.model = HSEmotionRecognizer(model_name)
        raw_labels = [HSEMOTION_LABEL_MAP[value] for _, value in sorted(self.model.idx_to_class.items())]
        self.keep_indices = [raw_labels.index(label) for label in FALLBACK_LABELS if label in raw_labels]
        self.labels = [raw_labels[idx] for idx in self.keep_indices]
        if self.labels != FALLBACK_LABELS:
            raise ValueError(f"{model_name} labels are {raw_labels}; cannot map them to {FALLBACK_LABELS}.")
        self.use_rgb = use_rgb
        self.calibration = calibration
        self.description = f"HSEmotion ONNX {model_name}"

    def predict(self, frame, gray_frame, box):
        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        if self.use_rgb:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        _, scores = self.model.predict_emotions(face, logits=False)
        probabilities = np.asarray(scores[self.keep_indices], dtype="float32")
        total = float(probabilities.sum())
        probabilities = probabilities / total if total > 0 else probabilities
        return apply_calibration(probabilities, self.calibration)


class HSEmotionEnsembleBackend:
    def __init__(self, artifact_path, use_rgb=True):
        import joblib
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Ensemble artifact not found: {self.artifact_path}")

        self.artifact = joblib.load(self.artifact_path)
        self.labels = list(self.artifact["labels"])
        if self.labels != FALLBACK_LABELS:
            raise ValueError(f"Ensemble labels are {self.labels}; expected {FALLBACK_LABELS}.")

        selected = self.artifact["selected"]
        if selected not in self.artifact["stackers"]:
            raise ValueError(f"Selected ensemble backend is not deployable: {selected}")

        self.selected = selected
        selected_stacker = self.artifact["stackers"][selected]
        self.scaler = selected_stacker.get("scaler")
        self.stacker = selected_stacker["model"]
        self.use_rgb = use_rgb
        self.recognizers = []
        for model_name in self.artifact["hsemotion_model_names"]:
            recognizer = HSEmotionRecognizer(model_name)
            raw_labels = [HSEMOTION_LABEL_MAP[value] for _, value in sorted(recognizer.idx_to_class.items())]
            keep_indices = [raw_labels.index(label) for label in FALLBACK_LABELS if label in raw_labels]
            if [raw_labels[idx] for idx in keep_indices] != FALLBACK_LABELS:
                raise ValueError(f"{model_name} labels are {raw_labels}; cannot map them to {FALLBACK_LABELS}.")
            self.recognizers.append((model_name, recognizer, keep_indices))

        self.description = f"HSEmotion stacked ML ensemble ({selected})"

    def predict(self, frame, gray_frame, box):
        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        if self.use_rgb:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        probability_items = []
        for _, recognizer, keep_indices in self.recognizers:
            _, scores = recognizer.predict_emotions(face, logits=False)
            probabilities = np.asarray(scores[keep_indices], dtype="float32")
            probability_items.append(normalize_probabilities(probabilities))

        features = stacked_features(probability_items)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return normalize_probabilities(self.stacker.predict_proba(features)[0])


class TorchEmotionBackend:
    def __init__(self, model_path, device_name="auto"):
        import torch
        from torch import nn
        from torchvision import models

        self.torch = torch
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {path}")

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
        self.device = torch.device(device_name)

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.labels = list(checkpoint["labels"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.arch = checkpoint.get("arch", "resnet18")
        if self.labels != FALLBACK_LABELS:
            raise ValueError(f"PyTorch labels are {self.labels}; expected {FALLBACK_LABELS}.")

        efficientnet_factories = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
        }
        if self.arch in efficientnet_factories:
            self.model = efficientnet_factories[self.arch](weights=None)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, len(self.labels))
        elif self.arch == "resnet18":
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.labels))
        else:
            raise ValueError(f"Unsupported PyTorch architecture: {self.arch}")

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype="float32").reshape(3, 1, 1)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype="float32").reshape(3, 1, 1)
        self.description = f"PyTorch {self.arch} on {self.device}"

    def predict(self, frame, gray_frame, box):
        x, y, w, h = box
        face = frame[y : y + h, x : x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        array = face.astype("float32") / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = (array - self.mean) / self.std
        tensor = self.torch.from_numpy(array).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return normalize_probabilities(probabilities)


class HaarFaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.detector.empty():
            raise RuntimeError("Could not load OpenCV Haar face detector.")

    def detect(self, frame, gray_frame):
        return self.detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(45, 45))


class YuNetFaceDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YuNet model not found: {path}")
        self.detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, frame, gray_frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        boxes = []
        for face in faces:
            if not np.all(np.isfinite(face[:4])):
                continue
            x, y, w, h = face[:4].astype(int)
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, w, h))
        return boxes


def build_face_detector(args):
    if args.detector == "yunet":
        return YuNetFaceDetector(args.yunet_model)
    return HaarFaceDetector()


def expand_box(box, frame_shape, margin):
    x, y, w, h = box
    height, width = frame_shape[:2]
    extra_w = int(w * margin)
    extra_h = int(h * margin)
    x1 = max(0, x - extra_w)
    y1 = max(0, y - extra_h)
    x2 = min(width, x + w + extra_w)
    y2 = min(height, y + h + extra_h)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def smooth_prediction(previous, current, smoothing):
    if previous is None or smoothing <= 0:
        return current
    return smoothing * previous + (1.0 - smoothing) * current


def draw_probability_panel(frame, probabilities, labels, origin):
    x0, y0 = origin
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(labels))]
    for row, idx in enumerate(top_indices):
        label = labels[idx]
        score = float(probabilities[idx])
        y = y0 + row * 22
        cv2.putText(frame, f"{label}: {score:.2f}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.rectangle(frame, (x0 + 120, y - 12), (x0 + 220, y - 2), (70, 70, 70), 1)
        cv2.rectangle(frame, (x0 + 120, y - 12), (x0 + 120 + int(score * 100), y - 2), (0, 180, 255), -1)


def main():
    args = parse_args()
    calibration = load_calibration(args.calibration)
    if args.backend == "hsemotion":
        backend = HSEmotionBackend(args.hsemotion_model, use_rgb=not args.hsemotion_bgr_input, calibration=calibration)
    elif args.backend == "hsemotion_ensemble":
        backend = HSEmotionEnsembleBackend(args.ensemble_artifact, use_rgb=not args.hsemotion_bgr_input)
    elif args.backend == "torch":
        backend = TorchEmotionBackend(args.torch_model, device_name=args.torch_device)
    else:
        backend = KerasEmotionBackend(args.model, args.labels, calibration=calibration)

    face_detector = build_face_detector(args)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam {args.camera} is not accessible.")

    smoothed = None
    fps = 0.0
    last_frame_at = time.perf_counter()

    print("Press Q to quit.")
    print(f"Backend: {backend.description}")
    print(f"Detector: {args.detector}")
    print(f"Calibration: {args.calibration or 'none'}")
    print(f"Loaded {len(backend.labels)} labels: {', '.join(backend.labels)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        frame_seconds = now - last_frame_at
        last_frame_at = now
        if frame_seconds > 0:
            fps = 0.90 * fps + 0.10 * (1.0 / frame_seconds) if fps else 1.0 / frame_seconds

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detect(frame, gray)

        if len(faces) == 0:
            smoothed = None
            cv2.putText(frame, "No face", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)

        for face_index, raw_box in enumerate(faces):
            x, y, w, h = expand_box(raw_box, frame.shape, args.face_margin)
            probabilities = backend.predict(frame, gray, (x, y, w, h))

            if face_index == 0:
                smoothed = smooth_prediction(smoothed, probabilities, args.smoothing)
                display_probabilities = smoothed
            else:
                display_probabilities = probabilities

            best_index = int(np.argmax(display_probabilities))
            confidence = float(display_probabilities[best_index])
            raw_label = backend.labels[best_index]
            label = raw_label if confidence >= args.confidence else f"uncertain ({raw_label})"
            color = (0, 220, 0) if confidence >= args.confidence else (0, 180, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, max(24, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            if args.show_probs:
                draw_probability_panel(frame, display_probabilities, backend.labels, (x, y + h + 24))

        cv2.putText(frame, f"FPS {fps:.1f}", (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Facial Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
