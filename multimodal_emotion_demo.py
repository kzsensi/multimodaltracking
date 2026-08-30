from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from audio_emotion import make_audio_recognizer
from emotion_schema import EMOTION_LABELS, make_result
from emotion_webcam import HaarFaceDetector, TorchEmotionBackend, YuNetFaceDetector
from fusion_emotion import DEFAULT_WEIGHTS, fuse_results
from text_emotion import make_text_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run visual, audio, text, and late-fusion emotion inference.")
    parser.add_argument("--image", help="Optional image path for visual emotion inference.")
    parser.add_argument("--audio-file", help="Optional WAV file path for audio emotion inference.")
    parser.add_argument("--text", default="", help="Optional text for textual emotion inference.")
    parser.add_argument("--output", default="reports/multimodal_smoke_latest/result.json")
    parser.add_argument(
        "--visual-model",
        default="models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt",
    )
    parser.add_argument("--visual-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--face-detector", choices=["yunet", "haar", "none"], default="yunet")
    parser.add_argument("--yunet-model", default="models/yunet/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--text-backend", choices=["lexicon", "hf"], default="lexicon")
    parser.add_argument("--audio-backend", choices=["acoustic", "hf"], default="acoustic")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--visual-weight", type=float, default=DEFAULT_WEIGHTS["visual"])
    parser.add_argument("--audio-weight", type=float, default=DEFAULT_WEIGHTS["audio"])
    parser.add_argument("--text-weight", type=float, default=DEFAULT_WEIGHTS["text"])
    return parser.parse_args()


def largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return max(boxes, key=lambda box: int(box[2]) * int(box[3]))


def make_face_detector(name, yunet_model):
    if name == "none":
        return None
    if name == "yunet":
        return YuNetFaceDetector(yunet_model)
    return HaarFaceDetector()


def predict_visual(image_path, model_path, device_name, detector_name, yunet_model):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        detector = make_face_detector(detector_name, yunet_model)
        box = largest_box(detector.detect(frame, gray)) if detector is not None else None
    except (RuntimeError, FileNotFoundError) as error:
        box = None
        detector_error = str(error)
    else:
        detector_error = None
    details = {"image": str(image_path), "face_detector": detector_name}
    if box is None:
        height, width = frame.shape[:2]
        box = (0, 0, width, height)
        details["face_detector"] = "full_image_fallback"
        if detector_error:
            details["detector_error"] = detector_error
    else:
        details["box"] = [int(value) for value in box]

    backend = TorchEmotionBackend(model_path, device_name)
    started_at = time.perf_counter()
    probabilities = backend.predict(frame, gray, box)
    details["seconds"] = time.perf_counter() - started_at
    details["backend_description"] = backend.description
    return make_result("visual", backend.description, probabilities, details)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if args.image:
        results.append(
            predict_visual(
                Path(args.image),
                args.visual_model,
                args.visual_device,
                args.face_detector,
                args.yunet_model,
            )
        )

    if args.audio_file:
        audio = make_audio_recognizer(
            args.audio_backend,
            local_files_only=args.hf_local_files_only,
        )
        results.append(audio.predict_file(args.audio_file))

    if args.text:
        text = make_text_recognizer(
            args.text_backend,
            local_files_only=args.hf_local_files_only,
        )
        results.append(text.predict(args.text))

    weights = {
        "visual": args.visual_weight,
        "audio": args.audio_weight,
        "text": args.text_weight,
    }
    fused = fuse_results(results, weights)
    payload = {
        "labels": EMOTION_LABELS,
        "modalities": [asdict(result) for result in results],
        "fusion": asdict(fused),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"fusion_label": fused.label, "fusion_confidence": fused.confidence, "output": str(output_path)}, indent=2))
    for result in results:
        print(f"{result.modality}: {result.label} ({result.confidence:.3f}) via {result.backend}")


if __name__ == "__main__":
    main()
