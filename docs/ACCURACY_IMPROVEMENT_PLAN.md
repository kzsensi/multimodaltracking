# Facial Accuracy Improvement Plan

Date: 2026-08-23

## Current Reality

The current best local facial-only model is functional, but not good enough as
the final model.

```text
Dataset:             MUTFER2024 local split
Classes:             7
Architecture:        simple CNN
Input:               48x48 RGB
Training style:      no augmentation
Epochs:              5
Validation accuracy: 39.21%
Test accuracy:       40.72%
```

This is better than random guessing for 7 classes, but it is still weak for a
real-time webcam demo. The per-class report shows the model is mostly learning
`happy`, while `disgust`, `sad`, `fear`, and `surprise` remain unreliable.

## Experiments Tried After The 40.72% Result

### 1. MobileNetV2 transfer learning

Command used:

```powershell
venv\Scripts\python.exe train_emotion_model.py --dataset dataset_mutfer2024 --epochs 8 --batch-size 64 --image-size 96 --color-mode rgb --architecture mobilenet_v2 --model models\mutfer2024_mobilenet_v2\emotion_model.h5 --labels models\mutfer2024_mobilenet_v2\labels.json --metadata models\mutfer2024_mobilenet_v2\model_metadata.json --reports reports\mutfer2024_mobilenet_v2
```

Observed result before stopping:

```text
Epoch 1 validation accuracy: 26.09%
Epoch 2 validation accuracy: 31.32%
```

Decision: stopped early. Frozen ImageNet MobileNetV2 did not beat the current
CNN quickly enough. It may improve with fine-tuning, but it is not the fastest
path by itself.

### 2. Augmented simple CNN

Command used:

```powershell
venv\Scripts\python.exe train_emotion_model.py --dataset dataset_mutfer2024 --epochs 15 --batch-size 64 --color-mode rgb --architecture simple_cnn --model models\mutfer2024_cnn_aug15\emotion_model.h5 --labels models\mutfer2024_cnn_aug15\labels.json --metadata models\mutfer2024_cnn_aug15\model_metadata.json --reports reports\mutfer2024_cnn_aug15
```

Observed result before stopping:

```text
Epoch 1 validation accuracy: 14.15%
Epoch 2 validation accuracy: 14.76%
Epoch 3 validation accuracy: 17.43%
Epoch 4 validation accuracy: 25.88%
```

Decision: stopped early. The augmentation configuration slowed learning and was
far below the current 40.72% test baseline.

## Code Fix Made During Improvement Pass

The trainer now evaluates the best saved validation checkpoint instead of
overwriting it with the final epoch model after training.

Why this matters:

```text
ModelCheckpoint saves the best validation model.
Final epoch can be worse than the best epoch.
Evaluation should use the best model, not blindly the last model.
```

This change is in:

```text
train_emotion_model.py
```

The current `labels.json` files were also updated to explicitly record:

```text
architecture:  simple_cnn
preprocessing: rescale_1_over_255
```

## Do We Need More Data?

Not as the first fix.

MUTFER2024 has 13,032 images and is fairly balanced across the 7 classes. That
is enough to diagnose the pipeline and train a better baseline. More data can
help later, but adding more datasets too early can also hurt if the labels,
face crops, image style, or emotion definitions do not match.

The bigger problems right now are:

1. The current CNN is too small and too low-resolution for subtle expressions.
2. Training was only 5 epochs for the best result.
3. Webcam input uses Haar face crops, which may not match the training images.
4. Some emotion classes are naturally ambiguous in still images.
5. The current system is 7-class; adding `contempt` as an 8th class before
   accuracy is stable will probably make accuracy worse.

## Best Next Steps

### Step 1: Keep the current model as the baseline

Do not delete or overwrite this until a new model beats it:

```text
models\mutfer2024_rgb\emotion_model.h5
reports\mutfer2024_rgb\metrics.json
```

### Step 2: Longer no-augmentation CNN experiment

The best current run used no augmentation and was still improving at epoch 5,
so a longer rerun was tried:

```powershell
venv\Scripts\python.exe train_emotion_model.py --dataset dataset_mutfer2024 --epochs 25 --batch-size 64 --color-mode rgb --architecture simple_cnn --no-augmentation --model models\mutfer2024_rgb_noaug25\emotion_model.h5 --labels models\mutfer2024_rgb_noaug25\labels.json --metadata models\mutfer2024_rgb_noaug25\model_metadata.json --reports reports\mutfer2024_rgb_noaug25
```

Observed result before stopping:

```text
Epoch 1 validation accuracy: 15.48%
Epoch 2 validation accuracy: 15.94%
```

Decision: stopped early. This rerun was not trending toward the existing
40.72% test baseline.

Do not promote any new local model unless its `metrics.json` beats:

```text
test accuracy: 40.72%
macro F1:       39.61%
```

### Step 3: Try a FER-pretrained model instead of ImageNet-only transfer

The fastest likely quality jump is to use a model already trained for facial
emotion recognition, not just a generic ImageNet model.

Recommended candidate:

```text
EmotiEffLib / AffectNet 7-class pretrained model
```

Why: it is designed for emotion recognition in photos/videos and supports
Python/ONNX-style usage, which fits a webcam system.

Official source checked:

```text
https://github.com/av-savchenko/face-emotion-recognition
```

### Step 4: Improve face detection/cropping

Haar cascade is acceptable for a basic demo, but it is old and misses many
faces under real webcam conditions. Accuracy can look bad even with a better
classifier if the crop is poor.

Better options later:

```text
OpenCV YuNet
MediaPipe Face Detection
RetinaFace
```

This should be done after we choose the final facial model.

### Step 5: Add the 8th emotion only after 7-class quality is acceptable

The common 8th facial emotion is `contempt`. FERPlus and AffectNet-style setups
can include it, but local FERPlus inspection showed `contempt` is tiny compared
with the main classes.

Decision:

```text
Finish a strong 7-class webcam model first.
Add 8-class contempt only in a second model/version.
```

## Definition Of "Facial-Only Complete"

Facial-only should be considered complete when all of these are true:

```text
1. Webcam runs without hard-coded labels.
2. Model, labels, and preprocessing metadata match.
3. Accuracy is meaningfully better than the current 40.72% test baseline.
4. Confusion matrix graph exists.
5. AUC curve graph exists.
6. Class distribution bar chart exists.
7. Efficiency graph exists.
8. README/docs explain the exact command and dataset used.
```

Current status:

```text
Functional: yes
Complete-quality: not yet
```

## Recommendation

Completed next action:

```text
Use a FER-pretrained model next, especially EmotiEffLib/AffectNet 7-class.
Then keep MUTFER2024 training/evaluation graphs as project evidence.
```

Result:

```text
HSEmotion enet_b2_7 raw full-test accuracy:        49.34%
HSEmotion enet_b2_7 calibrated full-test accuracy: 51.17%
Local CNN baseline full-test accuracy:             40.72%
```

Use the pretrained backend for the webcam demo:

```powershell
venv\Scripts\python.exe emotion_webcam.py --backend hsemotion --calibration models\calibration\hsemotion_enet_b2_7_yunet000.json --show-probs
```

This command now uses OpenCV YuNet face detection by default. Haar is still
available with `--detector haar` if YuNet has trouble on a machine.

The local MUTFER CNN and its graphs remain useful as project training evidence,
but the pretrained backend is currently the better real-time facial model.

## Accuracy Ceiling Note

The requested 80-90% target is not realistic for this current setup and
dataset without a much stronger model, a larger/cleaner aligned dataset, and
likely GPU-backed training/fine-tuning. The best working local result after
this pass is:

```text
Fair full-test score: 51.17% accuracy / 49.58% macro F1
Webcam-style detected-face subset: 51.32% accuracy / 49.17% macro F1
```

Next serious improvement would require one of:

```text
1. GPU fine-tuning of a large ViT/BEiT FER model.
2. A better labeled dataset such as AffectNet/RAF-DB with correct license/access.
3. A face-aligned training pipeline that matches the webcam detector crops.
```
