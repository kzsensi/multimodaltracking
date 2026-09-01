# Facial Emotion Recognition System
A real-time deep learning system that classifies emotions from webcam feed using CNNs.

<details>
  <summary><strong>Table of Contents</strong></summary>

- [Overview](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#overview)
- [Objectives](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#objectives)
- [How the system works](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#how-the-system-works)
- [Project Structure](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#project-structure)
- [Dataset Description](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#dataset-description)
  - [Dataset Source](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#dataset-source)
- [System Architecture](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#system-architecture)
- [Model Architecture](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#model-architecture)
- [Technologies Used](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#technologies-used)
- [Python Version Requirement](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#python-version-requirement)
- [Installation Instructions](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#installation-instructions)
  - [Step 1](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-1-clone-the-repository)
  - [Step 2](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-2-install-dependencies)
  - [Step 3](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-3-install-dependencies)
  - [Step 4](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-4-dataset-setup)
  - [Step 5](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-5-train-emotion-recognition-model-required)
  - [Step 6](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-6-run-the-system)
- [Output](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#output)
- [Performance Notes](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#performance-notes)
- [Limitations](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#limitations)
- [Future Enhancements](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#future-enhancements)
- [Conclusion](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#conclusion)
- [Author](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#author)
</details>

# Overview
This project implements a **Facial Emotion Recognition System** using **Deep Learning** and **Computer Vision**.
The system detects a human face from a webcam feed and classifies the facial expression in real time.

The system currently recognizes seven expressions. We are intentionally not adding the eighth class (`contempt`) yet because the current priority is improving seven-class quality first.

**Recognized emotions:**
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

# Objectives
- To understand and implement the **Convolutional Neural Networks (CNNs)**
- To perform **image-based emotion classification**
- To build a **real-time emotion recognition system**
- To gain hands-on experience with **TensorFlow, Keras and OpenCV**

# How the system works
1. Facial images are used to train/evaluate emotion classifiers
2. Images are preprocessed according to the selected backend
3. The model predicts emotion probabilities
4. During runtime:
    - Webcam captures frames
    - Face is detected
    - Emotion is predicted by either the local Keras CNN or pretrained HSEmotion ONNX model
    - Emotion label is displayed in real time

# Project Structure
```
facial-emotion-recognition/
│
├── dataset/                     # Training dataset
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── train_emotion_model.py       # Model training, metrics, and graph generation
├── evaluate_hsemotion_model.py  # Pretrained HSEmotion evaluation and graphs
├── calibrate_probabilities.py   # Validation-set probability calibration
├── evaluate_resemotenet_model.py# Optional ResEmoteNet checkpoint evaluation
├── emotion_webcam.py            # Real-time emotion detection
├── labels.json                  # Label order used by the saved model
├── reports/                     # Generated metrics and graphs after training
├── emotion_model.h5             # Trained CNN model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

```

# Dataset Description
The project now supports both the older FER2013-style dataset and the newer local MUTFER2024 split:
- Local CNN baseline input: 48 x 48 RGB
- HSEmotion pretrained input: 260 x 260 face crop internally
- One face per image
- Images organized into folders based on emotion labels

## Dataset Source
Original starter dataset:

https://www.kaggle.com/datasets/msambare/fer2013

Current local dataset used for the latest evaluation:

https://data.mendeley.com/datasets/vxtwysdsjw/2

The datasets are **not included** in this repository due to size and licensing constraints.

# System Architecture
The system follows the pipeline below:
1. Webcam frame capture
2. Face detection and crop extraction using YuNet by default
3. Backend-specific preprocessing
4. Emotion classification
5. Temporal smoothing and confidence thresholding
6. Real-time display with label, confidence, FPS, and optional probability bars

# Model Architecture
Two facial inference backends are available:

1. Local Keras CNN baseline
   - Three convolution blocks
   - Batch normalization
   - Max pooling
   - Dropout
   - Dense softmax classifier

2. Pretrained HSEmotion ONNX model
   - Recommended model: `enet_b2_7`
   - Trained for facial emotion recognition on AffectNet-style labels
   - Better current test result on MUTFER2024 than the local CNN baseline

3. Stacked HSEmotion + ML ensemble
   - Runs several pretrained HSEmotion variants
   - Combines their probabilities with a saved random-forest stacker
   - Stronger than the single HSEmotion model, but slower

4. GPU-trained PyTorch EfficientNet-B1
   - Trained with MUTFER2024 plus a balanced subset of AdamCodd/yolo-emotions
   - Continued with weak-class weighting for sad, fear, angry, and neutral
   - Best current visual accuracy result
   - Recommended when running from `venv_gpu`

The model outputs probabilities for each of the seven emotion classes.

# Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- ONNX Runtime
- HSEmotion ONNX

# Python Version Requirement
This project is tested and verified on **Python 3.10.11**.

**Important:**  
Some libraries used in this project (such as `face-recognition` and its dependencies like `dlib`) are **not fully compatible with Python 3.11+** at the time of development.

Using a Python version other than **3.10.11** may result in installation errors or runtime issues.

**Recommended version:**  
- Python **3.10.11**

# Installation Instructions
## Step 1: Clone the repository

```
git clone https://github.com/aniketrepo/facial-recognition-system.git
cd facial-recognition-system
```

## Step 2: Create a Virtual Environment
Windows
```
python -m venv venv
venv\Scripts\activate
```

Linux/MacOS
```
python3.10 -m venv venv
source venv/bin/activate
```
> Make sure the Python version inside venv is 3.10.11

## Step 3: Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Dataset Setup
- Add face images to the `dataset/` directory
- Each emotion should have its own folder
- The trainer supports either direct folders or train/test folders

Supported direct layout:
```
dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Supported split layout:
```
dataset/
├── train/
│   ├── angry/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

### MUTFER2024 Setup

If the downloaded MUTFER2024 folder is present at `MUTFER2024/`, prepare it
with:

```
python prepare_mutfer2024.py --source MUTFER2024 --output dataset_mutfer2024
```

Then train the RGB model:

```
python train_emotion_model.py --dataset dataset_mutfer2024 --epochs 5 --batch-size 64 --color-mode rgb --model emotion_model.h5 --labels labels.json --metadata model_metadata.json --reports reports/mutfer2024_rgb --no-augmentation
```

## Step 5: Download Pretrained Webcam Models

```
python download_pretrained_models.py
```

Optional experimental dependencies for ResEmoteNet/large-model checks:

```
pip install -r requirements-experimental.txt
```

## Step 6: Train Emotion Recognition Model (Optional Baseline)
This step generated the trained model file (`.h5`).
```
python train_emotion_model.py --dataset dataset --epochs 30
```

**Output:**
```
emotion_model.h5
labels.json
model_metadata.json
reports/class_distribution_bar_chart.png
reports/validation_auc_curve.png
reports/validation_confusion_matrix.png
reports/efficiency_graph.png
```
> This step is required only once unless you change the dataset.

## Step 7: Run the System

Recommended pretrained backend with YuNet face detection:

```
python emotion_webcam.py --backend hsemotion --calibration models/calibration/hsemotion_enet_b2_7_yunet000.json --show-probs
```

Best-accuracy ensemble backend:

```
python emotion_webcam.py --backend hsemotion_ensemble --show-probs
```

Best current GPU-trained backend:

```
venv_gpu\Scripts\python.exe emotion_webcam.py --backend torch --torch-device cuda --show-probs
```

This command now defaults to:

```
models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
```

Local Keras CNN baseline:

```
python emotion_webcam.py --backend keras --detector yunet --show-probs
```

Fallback Haar detector:

```
python emotion_webcam.py --backend hsemotion --detector haar --show-probs
```

## Step 7B: Run Multimodal Smoke Demo

The first multimodal layer supports visual, audio, text, and weighted late
fusion. The current text/audio backends are lightweight fallbacks for integration
testing. Use pretrained text/audio models later for final research metrics.

Visual + text smoke test:

```
venv_gpu\Scripts\python.exe multimodal_emotion_demo.py --image dataset_mutfer2024\test\happy\Happy_1724392681279.jpg --text "I am really happy and excited today!" --visual-device cuda --face-detector yunet --output reports\multimodal_smoke_latest\visual_text_result.json
```

Audio + text smoke test:

```
venv_gpu\Scripts\python.exe multimodal_emotion_demo.py --audio-file reports\multimodal_smoke_latest\sample_excited.wav --text "I am shocked and surprised!" --audio-backend acoustic --text-backend lexicon --output reports\multimodal_smoke_latest\audio_text_result.json
```

Optional pretrained multimodal dependencies:

```
venv_gpu\Scripts\python.exe -m pip install -r requirements-multimodal.txt
```

Prepare RAVDESS after `Audio_Speech_Actors_01-24.zip` finishes downloading:

```
venv_gpu\Scripts\python.exe prepare_ravdess_audio.py --zip data_sources\RAVDESS\Audio_Speech_Actors_01-24.zip --output dataset_ravdess_audio --overwrite
```

Evaluate the current no-download audio fallback:

```
venv_gpu\Scripts\python.exe evaluate_audio_emotion_model.py --dataset dataset_ravdess_audio\test --backend acoustic --reports reports\audio_ravdess_acoustic_latest
```

## Step 7C: Browser Demo and Cloudflare Link

Run the local Gradio browser demo:

```
venv_gpu\Scripts\python.exe web_multimodal_demo.py --host 127.0.0.1 --port 7860 --visual-device cuda
```

Open locally:

```
http://127.0.0.1:7860
```

Create a temporary public Cloudflare link:

```
cloudflared tunnel --url http://127.0.0.1:7860
```

Keep both commands running while sharing the public link. The temporary
Cloudflare URL stops working when the local server or tunnel is closed.

## Step 8: Evaluate the Pretrained HSEmotion Model

Fair full-image evaluation:

```
python evaluate_hsemotion_model.py --dataset dataset_mutfer2024 --split test --model-name enet_b2_7 --reports reports/hsemotion_enet_b2_7_test_rgb_full_v2 --batch-size 32 --rgb-input --detector none
```

Webcam-style detected-face evaluation:

```
python evaluate_hsemotion_model.py --dataset dataset_mutfer2024 --split test --model-name enet_b2_7 --reports reports/hsemotion_enet_b2_7_test_rgb_yunet000_v2 --batch-size 32 --rgb-input --detector yunet --face-margin 0.0
```

Calibrate probabilities using validation predictions:

```
python calibrate_probabilities.py --validation reports/hsemotion_enet_b2_7_validation_rgb_full_v2/predictions.npz --test reports/hsemotion_enet_b2_7_test_rgb_full_v2/predictions.npz --output models/calibration/hsemotion_enet_b2_7_full.json --report reports/calibration_hsemotion_enet_b2_7_full.json
```

Build the current stacked ML ensemble from saved prediction files:

```
python ensemble_hsemotion_predictions.py --validation reports/hsemotion_enet_b2_7_validation_rgb_full_v2/predictions.npz reports/hsemotion_enet_b0_8_best_afew_validation_rgb_full_v3/predictions.npz reports/hsemotion_enet_b0_8_best_vgaf_validation_rgb_full_v3/predictions.npz reports/hsemotion_enet_b2_8_validation_rgb_full_v3/predictions.npz --test reports/hsemotion_enet_b2_7_test_rgb_full_v2/predictions.npz reports/hsemotion_enet_b0_8_best_afew_test_rgb_full_v3/predictions.npz reports/hsemotion_enet_b0_8_best_vgaf_test_rgb_full_v3/predictions.npz reports/hsemotion_enet_b2_8_test_rgb_full_v3/predictions.npz --names enet_b2_7 enet_b0_8_afew enet_b0_8_vgaf enet_b2_8 --hsemotion-models enet_b2_7 enet_b0_8_best_afew enet_b0_8_best_vgaf enet_b2_8 --reports reports/hsemotion_stacked_ml_ensemble_latest --model-output models/ensembles/hsemotion_stacked_ml_latest.joblib
```

Latest local result on the MUTFER2024 test split:

```
Keras CNN baseline, full test:                    40.72% accuracy
HSEmotion enet_b2_7, full test:                   49.34% accuracy / 46.19% macro F1
HSEmotion + validation calibration, full test:    51.17% accuracy / 49.58% macro F1
HSEmotion + YuNet crop + calibration, face-found subset: 51.32% accuracy / 49.17% macro F1
HSEmotion stacked ML ensemble, full test:         58.41% accuracy / 57.85% macro F1
PyTorch EfficientNet-B0, full test:               80.63% accuracy / 80.20% macro F1
PyTorch EfficientNet-B0 + flip TTA, full test:    81.40% accuracy / 80.91% macro F1
PyTorch EfficientNet-B1 weakboost, full test:     85.22% accuracy / 84.86% macro F1
PyTorch EfficientNet-B1 weakboost + flip TTA:     86.65% accuracy / 86.30% macro F1
```

Note: the YuNet-cropped score is computed only on images where the detector
found a face. Use the full-test score for dataset reporting and the YuNet score
for webcam-style behavior.

## Step 9: Controls
- Press `Q` to quit the application

# Output
- Detected face is highlighted using a bounding box
- Predicted emotion label is displayed above the face
- Predictions are based on the highest probability from the model output

# Performance Notes
- Accuracy depends on dataset size and class balance
- Similar emotions such as fear and surprise may overlap
- Lighting conditions affect face detection performance
- Emotion recognition is probabilistic and not always exact

# Limitations
- Works best with frontal faces
- Sensitive to lighting and camera quality
- Does not account for head pose variations
- Emotion classification may vary across individuals 

# Future Enhancements
- MediaPipe or RetinaFace detector comparison
- Optional fine-tuning of a stronger architecture
- Graphical User Interface
- Deployment as a web or desktop application

# Conclusion
This project demonstrates a complete deep learning-based solution for facial emotion recognition. It includes dataset preparation, CNN training, model evaluation, and real-time emotion prediction using webcam input.

# Author
Aniket Dixit

B.Tech Data Science
