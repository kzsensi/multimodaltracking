from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from audio_emotion import make_audio_recognizer
from emotion_schema import EMOTION_LABELS, make_result
from emotion_webcam import TorchEmotionBackend, YuNetFaceDetector
from fusion_emotion import fuse_results
from text_emotion import make_text_recognizer


DEFAULT_MODEL = "models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt"
DEFAULT_YUNET = "models/yunet/face_detection_yunet_2023mar.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Hosted Gradio demo for multimodal emotion recognition.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--visual-model", default=DEFAULT_MODEL)
    parser.add_argument("--visual-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--yunet-model", default=DEFAULT_YUNET)
    return parser.parse_args()


class DemoRuntime:
    def __init__(self, visual_model, visual_device, yunet_model):
        self.visual = TorchEmotionBackend(visual_model, visual_device)
        self.detector = YuNetFaceDetector(yunet_model) if Path(yunet_model).exists() else None
        self.text = make_text_recognizer("lexicon")
        self.audio = make_audio_recognizer("acoustic")

    def predict_visual(self, image):
        if image is None:
            return None, None

        frame_rgb = np.asarray(image)
        if frame_rgb.ndim != 3:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = None
        detector_name = "full_image_fallback"

        if self.detector is not None:
            faces = self.detector.detect(frame, gray)
            if len(faces):
                box = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                detector_name = "yunet"

        if box is None:
            height, width = frame.shape[:2]
            box = (0, 0, width, height)

        started_at = time.perf_counter()
        probabilities = self.visual.predict(frame, gray, box)
        elapsed = time.perf_counter() - started_at
        result = make_result(
            "visual",
            self.visual.description,
            probabilities,
            {
                "face_detector": detector_name,
                "box": [int(value) for value in box],
                "seconds": elapsed,
            },
        )

        x, y, w, h = [int(value) for value in box]
        annotated = frame_rgb.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (35, 170, 70), 2)
        cv2.putText(
            annotated,
            f"{result.label} {result.confidence:.2f}",
            (max(0, x), max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (35, 170, 70),
            2,
            cv2.LINE_AA,
        )
        return result, annotated

    def predict_audio(self, audio_path):
        if not audio_path:
            return None
        path = audio_path if isinstance(audio_path, str) else audio_path[0]
        return self.audio.predict_file(path)

    def predict_text(self, text):
        if not text or not text.strip():
            return None
        return self.text.predict(text)


def probs_for_chart(result):
    if result is None:
        return {label: 0.0 for label in EMOTION_LABELS}
    return {label: round(result.probabilities[label], 4) for label in EMOTION_LABELS}


def result_line(result):
    if result is None:
        return "not provided"
    return f"{result.label} ({result.confidence:.2%})"


def build_interface(runtime):
    def run(image, audio, text, visual_weight, audio_weight, text_weight):
        visual_result, annotated = runtime.predict_visual(image)
        audio_result = runtime.predict_audio(audio)
        text_result = runtime.predict_text(text)
        fusion_result = fuse_results(
            [visual_result, audio_result, text_result],
            {"visual": visual_weight, "audio": audio_weight, "text": text_weight},
        )
        summary = (
            f"Final emotion: {fusion_result.label} ({fusion_result.confidence:.2%})\n"
            f"Visual: {result_line(visual_result)}\n"
            f"Audio: {result_line(audio_result)}\n"
            f"Text: {result_line(text_result)}"
        )
        return (
            annotated,
            summary,
            probs_for_chart(fusion_result),
            probs_for_chart(visual_result),
            probs_for_chart(audio_result),
            probs_for_chart(text_result),
        )

    with gr.Blocks(title="Multimodal Emotion Recognition") as demo:
        gr.Markdown("# Multimodal Emotion Recognition")
        gr.Markdown(
            "Visual model: EfficientNet-B1, 85.22% real-time test accuracy on the local MUTFER2024 test split. "
            "Use the webcam input for live visual predictions, or upload an image/audio file for a one-shot demo."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Live Webcam"):
                        webcam = gr.Image(
                            label="Live Webcam",
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                            mirror_webcam=True,
                        )
                    with gr.Tab("Upload Image"):
                        upload_image = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
                audio = gr.Audio(label="Optional WAV Audio", sources=["microphone", "upload"], type="filepath")
                text = gr.Textbox(label="Optional Text", lines=3, placeholder="Type a sentence here...")
                with gr.Accordion("Fusion Weights", open=False):
                    visual_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Visual")
                    audio_weight = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Audio")
                    text_weight = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Text")
                button = gr.Button("Analyze Uploaded Image / Inputs", variant="primary")
            with gr.Column(scale=1):
                annotated = gr.Image(label="Visual Result", type="numpy")
                summary = gr.Textbox(label="Result Summary", lines=5)
        with gr.Row():
            fusion = gr.Label(label="Fusion Probabilities", num_top_classes=7)
            visual = gr.Label(label="Visual Probabilities", num_top_classes=7)
        with gr.Row():
            audio_probs = gr.Label(label="Audio Probabilities", num_top_classes=7)
            text_probs = gr.Label(label="Text Probabilities", num_top_classes=7)
        button.click(
            run,
            inputs=[upload_image, audio, text, visual_weight, audio_weight, text_weight],
            outputs=[annotated, summary, fusion, visual, audio_probs, text_probs],
        )
        webcam.stream(
            run,
            inputs=[webcam, audio, text, visual_weight, audio_weight, text_weight],
            outputs=[annotated, summary, fusion, visual, audio_probs, text_probs],
            stream_every=0.75,
            trigger_mode="always_last",
            concurrency_limit=1,
            show_progress="hidden",
        )
    return demo


def main():
    args = parse_args()
    runtime = DemoRuntime(args.visual_model, args.visual_device, args.yunet_model)
    demo = build_interface(runtime)
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
