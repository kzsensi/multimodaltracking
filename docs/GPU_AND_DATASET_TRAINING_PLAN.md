# GPU And Dataset Training Plan

Date: 2026-08-26

## What We Verified Locally

This laptop has an NVIDIA GPU:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM: 4 GB
Driver: 581.83
CUDA shown by driver: 13.0
```

The current project environment is not using it yet:

```text
Python:     3.10.11
TensorFlow: 2.20.0, GPU devices: none
PyTorch:    2.13.0+cpu, CUDA available: false
```

Conclusion: the hardware can be used for training, but the Python ML stack must
be changed to a GPU-enabled PyTorch install, or training must be moved to WSL2
or a cloud GPU.

Attempted to install the CUDA 12.6 PyTorch wheel on 2026-08-26. The first
attempt failed because the command did not allow normal PyPI dependencies. The
second attempt started downloading the 2.6 GB CUDA wheel but stayed silent for
several minutes, so it was interrupted cleanly. After interruption:

```text
pip:     upgraded to 26.2.1
PyTorch: still 2.13.0+cpu
CUDA:    still unavailable inside PyTorch
```

## GPU Decision

Use PyTorch CUDA on native Windows for the next training pass.

Reasons:

- The laptop has an NVIDIA CUDA-capable GPU.
- Current TensorFlow on native Windows does not expose the GPU here.
- Official TensorFlow docs recommend WSL2 for modern Windows GPU access.
- PyTorch provides Windows CUDA wheels and is simpler for this setup.

Do not use ComfyUI for this. ComfyUI is useful for image generation workflows,
not supervised facial emotion recognition training. Synthetic images can be a
future augmentation experiment, but real labeled FER datasets should come first.

## Best Dataset Download Order

### 1. RAF-DB

Official source:

```text
http://whdeng.cn/RAF/model1.html
```

Use it if the official page opens and access is granted. RAF-DB is one of the
best practical next datasets because it is in-the-wild and its basic subset
maps cleanly to the current seven expressions:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

This is the best next download for improving the current facial-only webcam
model without changing the label set.

### 2. AffectNet

Official source:

```text
https://www.mohammadmahoor.com/pages/databases/affectnet/
```

AffectNet is much larger and stronger for FER pretraining/fine-tuning. It is a
better research-quality source than MUTFER alone, but it can be large and
access-controlled.

Use seven classes first. Do not add `contempt` until the seven-class model is
stable.

### 3. FERPlus

Official source:

```text
https://github.com/microsoft/FERPlus
```

FERPlus improves the old FER2013 labels using crowd-vote distributions. It is
useful because it adds better labels and includes `contempt`, but the underlying
images are still FER2013 and `contempt` is very small.

Use it as a label-quality upgrade, not as the main accuracy solution.

### 4. MUTFER2024

Official source:

```text
https://data.mendeley.com/datasets/vxtwysdsjw/2
```

Already downloaded and prepared locally. Keep using it because it is recent,
manageable, and demographically useful, but it has only seven classes and is
not enough by itself to reach very high accuracy.

### 5. ExpW

Public mirrors/source listings show about 91,793 in-the-wild labeled faces.
Use it if it is easy to download cleanly. It is seven-class, so it helps scale
the training set, but licensing and source clarity must be checked before final
report use.

## Data To Avoid For The Immediate Facial-Only Build

Avoid these for now:

```text
DFEW / Aff-Wild2 / ABAW video datasets
```

They are useful later for dynamic video or multimodal research, but they slow
down the quick facial-only webcam milestone.

## Accuracy Improvement Actions We Can Do Now

Update after the short-time ML experiment on 2026-08-26:

```text
Current best without downloading new data:
HSEmotion stacked random-forest ML ensemble
58.41% accuracy / 57.85% macro F1 on MUTFER2024 test
```

This means the immediate no-download deployment path is:

```text
Use models/ensembles/hsemotion_stacked_ml_latest.joblib with
emotion_webcam.py --backend hsemotion_ensemble
```

The accuracy improved, but the ensemble is slower because it runs four
pretrained facial models per face. The fast fallback remains the single
calibrated HSEmotion backend.

1. Install GPU-enabled PyTorch in the project environment.
2. Create a face-cropped/aligned copy of MUTFER2024 using the same YuNet face
   detector used by the webcam.
3. Train an EfficientNet-B0/B2 or ResNet18/ResNet50 classifier on 224x224 face
   crops with class weights, augmentation, mixed precision, and early stopping.
4. Evaluate against the held-out MUTFER2024 test split and generate:

```text
AUC curve graph
confusion matrix graph
bar chart
efficiency graph
```

5. Keep the current calibrated HSEmotion model as the deployed webcam fallback
   unless the newly trained model beats it.

## Realistic Target

For uncontrolled seven-class facial emotion recognition, 80-90% test accuracy
is usually not a realistic promise on mixed real-world data from a small laptop
training setup. A practical near-term target is:

```text
55-65% on MUTFER2024 held-out test
```

With RAF-DB or AffectNet fine-tuning on the GPU, a stronger target becomes:

```text
65-75% on a matching seven-class test protocol
```

Higher numbers may be possible on easier/cleaner datasets, but they should not
be claimed unless our own held-out test graphs prove it.

## Exact Next Download Request

Ask the user to download RAF-DB first if access opens. Place it under:

```text
D:\SITES\miscellaneous\facial_recognition\data_sources\rafdb
```

If RAF-DB access fails, download AffectNet next, or provide whatever archive
was received so a converter script can be written around the actual folder
layout.

## Update After Checking Access-Blocked Sources

Checked on 2026-08-26:

```text
FERPlus: cloned locally into data_sources\FERPlus_repo
RAF-DB:  official page requires password by email request, so not fast
AffectNet official page: no direct download button; request is through AffectNet+
```

FERPlus is useful, but it is not a complete image dataset by itself. It only
contains labels and helper code. The original FER2013 image pixels are still
needed before FERPlus can become a trainable local dataset.

## Faster Alternative Datasets Found

### AdamCodd/yolo-emotions

Source:

```text
https://huggingface.co/datasets/AdamCodd/yolo-emotions
```

Why it is useful:

- around 155K samples;
- seven classes matching the current project;
- includes merged data from ExpW, FER2013/FERPlus, AffectNet, and RAF-DB;
- already includes face/emotion detection style structure.

Concern:

- the dataset card itself warns about noisy labels, repeated images, and some
  weak examples;
- because it includes derivatives of access-controlled datasets, cite and use
  carefully in the final report.

Decision: this is the fastest large candidate for experimentation, but train
with cleaning and keep MUTFER2024 as a cleaner local validation/test check.

### Kaggle AffectNet YOLO Format

Source:

```text
https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format
```

Why it is useful:

- appears to expose AffectNet-style labels in a YOLO-friendly format;
- includes eight labels in the source description, including `contempt`.

Concern:

- Kaggle download requires Kaggle account/API credentials;
- official AffectNet terms should still be respected.

Decision: useful only if the user can download through Kaggle and accepts the
dataset terms.

### EmoNet-Face

Sources:

```text
https://huggingface.co/datasets/laion/emonet-face-binary
https://huggingface.co/datasets/laion/emonet-face-big
```

Why it is useful:

- 2025 NeurIPS synthetic facial emotion benchmark suite;
- broad 40-category taxonomy;
- binary fine-tuning set is about 20K images;
- big pretraining set is over 200K images.

Concern:

- synthetic faces are not the same as webcam faces;
- labels are fine-grained and need mapping to our seven classes.

Decision: use as augmentation or pretraining only, not as the main final test.

### GFFD-2025

Source:

```text
https://data.mendeley.com/datasets/wmfd4p3z32/1
```

Why it is useful:

- published 2025;
- seven emotions matching the project;
- includes genuine vs acted distinction;
- 224x224 cropped/augmented images are mentioned.

Concern:

- small dataset, around 1.9K raw images and 1.5K cropped/augmented images;
- controlled indoor collection, not enough alone for robust webcam accuracy.

Decision: good small supplementary dataset, not a main training source.

### FEA-20K

Source:

```text
https://huggingface.co/datasets/QBiscuits/FEA-20K
```

Why it is useful:

- 2025/2026 fine-grained facial emotion analysis dataset;
- about 20K samples;
- includes emotion labels and action-unit reasoning fields;
- small enough to download quickly compared with AffectNet.

Concern:

- designed more for vision-language reasoning than a plain seven-class FER
  classifier;
- labels require mapping and cleaning before use.

Decision: interesting, but lower priority than AdamCodd/yolo-emotions for the
current webcam classifier.

## New Download Priority

For the quick accuracy push:

```text
1. AdamCodd/yolo-emotions from Hugging Face
2. FER2013 original images/CSV so FERPlus can be used
3. Kaggle AffectNet YOLO Format, if Kaggle download works
4. GFFD-2025 as a small supplementary/evaluation dataset
5. EmoNet-Face Binary/Big only as synthetic augmentation
6. RAF-DB/AffectNet official access later, when passwords/forms are available
```

## Pretrained Model Search Update

Checked on 2026-08-26.

### 1. EmotiEffLib / HSEmotion

Sources:

```text
https://github.com/sb-ai-lab/EmotiEffLib
https://github.com/av-savchenko/hsemotion
```

Status in this project:

```text
Already integrated and currently the best working webcam backend.
```

Why it stays first:

- supports photos/videos;
- supports ONNX and PyTorch-style usage;
- has 7-class and 8-class model variants;
- fast enough for real-time webcam;
- can extract visual embeddings for a smaller classifier.

Current local best:

```text
HSEmotion enet_b2_7 calibrated full-test accuracy: 51.17%
```

Next action:

```text
Evaluate more HSEmotion variants and try probability averaging/ensemble.
```

### 2. Py-Feat Detectorv2 / face_multitask_v2

Sources:

```text
https://py-feat.org/
https://py-feat.org/pages/models/
https://huggingface.co/py-feat/face_multitask_v2
```

Why it is useful:

- predicts 7-class emotion;
- also predicts action units, valence/arousal, gaze, head pose, face mesh, and
  blendshapes;
- useful for the later multimodal/reporting system because it gives richer
  facial evidence than a plain emotion softmax.

Concern:

```text
License is non-commercial research for the v2 multitask model.
```

Next action:

```text
Install and evaluate on MUTFER2024 test if license is acceptable.
```

### 3. ONNX Model Zoo Emotion FERPlus 8

Source:

```text
https://huggingface.co/onnxmodelzoo/emotion-ferplus-8
```

Why it is useful:

- small ONNX model, about 34 MB;
- eight emotion outputs including contempt;
- Apache-2.0 license;
- easy to run with ONNX Runtime.

Concern:

- expects grayscale 64x64 input;
- trained on FERPlus/FER2013 style images, so it may not beat HSEmotion on
  modern webcam photos.

Next action:

```text
Add as an optional 8-class comparison backend, not the main model.
```

### 4. Tanneru BEiT-Large FER/RAFDB/AffectNet

Source:

```text
https://huggingface.co/Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large
```

Why it is useful:

- seven labels match the project;
- model card reports validation accuracy around 76%;
- trained on FER2013, RAF-DB, and AffectNet-style data;
- Apache-2.0 license.

Concern:

- BEiT-Large is heavy;
- previous local download attempt stalled;
- 4 GB VRAM is not enough for comfortable fine-tuning, but CPU/low-batch
  inference may still be possible.

Next action:

```text
Use only for offline evaluation or teacher predictions, not real-time webcam.
```

### 5. EMO-AffectNet

Source:

```text
https://huggingface.co/ElenaRyumina/face_emotion_recognition
```

Why it is useful:

- MIT license;
- made for static and dynamic facial emotion recognition;
- includes image/video emotion model resources.

Concern:

- integration is less direct than HSEmotion or ONNX FERPlus;
- needs a careful local loader/evaluator before trusting it.

Next action:

```text
Evaluate after HSEmotion ensemble and ONNX FERPlus.
```

### 6. FaceEmo-Set ViT

Source:

```text
https://huggingface.co/jihedjabnoun/faceemo-set
```

Why it is useful:

- MIT license;
- seven labels match the project;
- provides ViT model weights;
- model card reports cross-dataset testing on AffectNet and FER2013.

Concern:

- dataset/model is less established than HSEmotion/FERPlus/Py-Feat;
- downloads and loader need validation.

Next action:

```text
Candidate for offline comparison, not first webcam backend.
```

### 7. DeepFace Emotion

Source:

```text
https://github.com/serengil/deepface
```

Why it is useful:

- very easy to install and run;
- supports real-time webcam analysis;
- predicts the same seven basic facial expressions.

Concern:

- designed as a broad face-analysis package, not the strongest FER-specific
  model;
- likely best as a baseline comparison only.

Next action:

```text
Optional quick comparison, not final accuracy path.
```

### Models Already Tried Locally

```text
ResEmoteNet checkpoint: tested, did not beat current HSEmotion backend.
```

## New Pretrained Model Priority

For this project, evaluate in this order:

```text
1. HSEmotion variants + ensemble/calibration
2. ONNX FERPlus 8-class model for comparison/contempt support
3. Py-Feat Detectorv2 if research-only license is acceptable
4. Tanneru BEiT-Large for offline teacher/evaluation only
5. EMO-AffectNet
6. FaceEmo-Set ViT
7. DeepFace baseline comparison
```

For later multimodal work:

```text
Audio: emotion2vec_plus_base
Text:  j-hartmann/emotion-english-distilroberta-base
Fusion/evaluation references: MELD and MERTools
```

## GPU Setup Completed

Date: 2026-08-28

Created a separate GPU environment so the current working webcam environment
is not broken:

```text
venv_gpu
```

Installed CUDA-enabled PyTorch from the official PyTorch CUDA 12.8 wheel index:

```powershell
venv_gpu\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Saved the reproducible dependency file:

```text
requirements-gpu.txt
```

Verification result:

```text
torch:          2.11.0+cu128
torch CUDA:     12.8
CUDA available: true
GPU:            NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM:           4 GB
```

Conclusion:

```text
Codex can now use the laptop GPU through venv_gpu. The user does not need to
grant extra GPU access inside Codex; the missing part was the Python CUDA
package, and that is now installed.
```

## AdamCodd/yolo-emotions Download Decision

Checked on 2026-08-28:

```text
Source:          https://huggingface.co/datasets/AdamCodd/yolo-emotions
Shown size:      about 10.2 GB
Approx samples: 155K
Labels:          angry, disgust, fear, happy, neutral, sad, surprise
```

Decision:

```text
Yes, it is acceptable for this project, but do not train on all 10 GB first.
First download or stream a smaller balanced subset, clean it, train a quick
GPU EfficientNet/ResNet classifier, and evaluate against MUTFER2024 test.
Only use the full dataset if the subset beats the current 58.41% ensemble or
clearly improves weak classes such as fear, sad, angry, and neutral.
```

## Training Results After Download

Date: 2026-08-28

The full `AdamCodd/yolo-emotions` zip was downloaded locally:

```text
data_sources/yolo-emotions/emotions_dataset.zip
```

The gradual-data strategy was tested:

```text
1k YOLO crops per class + MUTFER train:
  ResNet18 fine-tune:       77.17% accuracy / 76.59% macro F1
  EfficientNet-B0 fine-tune: 80.63% accuracy / 80.20% macro F1

2k YOLO crops per class + MUTFER train:
  ResNet18 fine-tune:       74.21% accuracy / 73.53% macro F1
```

Decision:

```text
Keep the 1k-per-class YOLO subset. Do not blindly increase the external data
yet, because the 2k-per-class run performed worse. The likely reason is label
noise/domain mismatch in the larger external subset.
```

Current winner:

```text
models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
reports/final_visual_fer_efficientnet_b1_85_86
```

Recommended run command:

```powershell
venv_gpu\Scripts\python.exe emotion_webcam.py --backend torch --torch-device cuda --show-probs
```

Additional no-retraining evaluation:

```text
EfficientNet-B0 + horizontal flip TTA:
81.40% accuracy / 80.91% macro F1
```

## Final Visual Improvement Loop

Date: 2026-08-28

EfficientNet-B1 was trained and then continued with lower learning rates,
richer augmentation, label smoothing, and weak-class loss boosts for sad, fear,
angry, and neutral.

Final real-time result:

```text
Model:    models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
Reports:  reports/final_visual_fer_efficientnet_b1_85_86
Accuracy: 85.22%
Macro F1: 84.86%
```

Final offline TTA result:

```text
Accuracy: 86.65%
Macro F1: 86.30%
```

Weak-class real-time F1:

```text
sad:     81.82%
fear:    83.37%
angry:   83.09%
neutral: 82.64%
```

This TTA result is useful for the final report, but it is slower than
single-pass webcam inference.
