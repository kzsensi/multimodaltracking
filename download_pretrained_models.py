import argparse
import urllib.request
from pathlib import Path


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained models used by the webcam demo.")
    parser.add_argument(
        "--yunet-model",
        default="models/yunet/face_detection_yunet_2023mar.onnx",
        help="Where to save the YuNet face detector ONNX model.",
    )
    parser.add_argument(
        "--hsemotion-model",
        default="enet_b2_7",
        help="HSEmotion model name to warm into the local ~/.hsemotion cache.",
    )
    return parser.parse_args()


def download_yunet(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"YuNet already exists: {path.resolve()}")
        return
    print(f"Downloading YuNet from {YUNET_URL}")
    urllib.request.urlretrieve(YUNET_URL, path)
    print(f"Saved YuNet: {path.resolve()}")


def warm_hsemotion(model_name):
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    recognizer = HSEmotionRecognizer(model_name)
    print(f"HSEmotion ready: {model_name}")
    print(f"Classes: {recognizer.idx_to_class}")


def main():
    args = parse_args()
    download_yunet(args.yunet_model)
    warm_hsemotion(args.hsemotion_model)


if __name__ == "__main__":
    main()
