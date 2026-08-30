# Current State Audit

Audit date: 2026-08-23

## Executive Finding

The repository is a working-shaped **facial-expression prototype**, but it is
not yet a multimodal emotion recognition system.

- As a seven-class facial webcam demonstration: approximately **60-70%**
  complete.
- Against the full visual + audio + text + fusion project title: approximately
  **15-20%** complete.

The saved model loads successfully and has the expected input and output
shapes. However, the repository contains no dataset, test report, training
history, confusion matrix, per-class metrics, audio branch, text branch,
multimodal fusion, or integrated interface.

## Post-Audit Facial-Only Update

After this audit, the facial-only code was improved:

- `train_emotion_model.py` now supports either direct emotion folders or
  `train`/`validation`/`test` split folders.
- The output class count is now inferred from dataset folders instead of being
  hard-coded to seven.
- Training now saves `labels.json`, `model_metadata.json`, metrics, and graph
  outputs under `reports/`.
- The webcam script now loads `labels.json`, checks label/model compatibility,
  adds prediction smoothing, adds a confidence threshold, and can show top
  probabilities.

The limitation still stands: no local dataset has been added yet, so meaningful
accuracy, AUC, confusion matrix, and efficiency graphs will appear only after a
training run on local data.

## Repository Inventory

Tracked project files:

```text
.gitignore
LICENSE
README.md
emotion_model.h5
emotion_webcam.py
requirements.txt
train_emotion_model.py
```

The locally added research documents are currently untracked. The dataset is
correctly excluded by `.gitignore`.

## What Is Implemented

### Visual training

`train_emotion_model.py` implements:

- folder-based image loading from `dataset/`;
- conversion to 48 x 48 grayscale images;
- normalization to `[0, 1]`;
- an 80/20 validation split;
- a three-block CNN with batch normalization, max pooling, and dropout;
- a 256-unit dense layer;
- seven-class softmax classification;
- 15 training epochs with Adam and categorical cross-entropy;
- saving to `emotion_model.h5`.

### Saved visual model

The checked-in model loads successfully with TensorFlow 2.20. Its verified
properties are:

```text
Input:       (None, 48, 48, 1)
Output:      (None, 7)
Parameters:  619,911 (about 2.36 MB of parameters)
```

This proves that the file is structurally usable. It does **not** prove its
accuracy because no evaluation results or training history were saved.

### Webcam inference

`emotion_webcam.py` implements:

- webcam capture with OpenCV;
- grayscale conversion;
- Haar-cascade frontal-face detection;
- 48 x 48 face resizing and normalization;
- seven-class prediction;
- face box and winning label overlay;
- `q` to quit.

Current labels are in this order:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

This happens to match alphabetical folder ordering for the current seven
folder names, but the mapping is not saved with the model.

## What Is Missing

| Capability | Current state | Needed for final project |
|---|---:|---|
| Facial-expression model | Basic baseline exists | Better data/model, confidence, smoothing |
| Webcam face detection | Basic Haar detector exists | Robust detector and no-face handling |
| Speech emotion | 0% | Microphone buffering and acoustic model |
| Speech-to-text | 0% | Local ASR for spoken textual content |
| Typed text emotion | 0% | Text classifier and input control |
| Shared label mapping | 0% | Canonical seven-class probability schema |
| Multimodal fusion | 0% | Availability-aware late fusion |
| Temporal alignment | 0% | Timestamped windows and stale-result expiry |
| Integrated user interface | 0% | One camera/mic/text dashboard |
| Evaluation | 0% | Per-modality and fused metrics |
| Reproducibility | Minimal | Seeds, splits, model metadata, label JSON |

## Important Technical Problems

### 1. The Kaggle folder layout is not handled directly

The `msambare/fer2013` Kaggle copy normally contains separate `train/` and
`test/` directories. The script expects emotion folders immediately under
`dataset/`:

```text
dataset/angry
dataset/disgust
...
```

If the Kaggle archive is extracted as `dataset/train/angry`, then
`DATASET_PATH` must point to `dataset/train`, or the data loader must be
rewritten to respect the supplied train/test split. The current script also
creates a new 20% validation split and does not evaluate on Kaggle's test set.

### 2. There is no trustworthy accuracy claim

The model is saved, but the repository has no:

- held-out test evaluation;
- accuracy history;
- macro F1 or weighted F1;
- per-class precision and recall;
- confusion matrix;
- dataset version or class counts;
- record of the best epoch.

Emotion datasets are imbalanced, so validation accuracy alone is insufficient.
FER2013 has only 436 training examples for `disgust`, compared with 7,215 for
`happy` in the common Kaggle split.

### 3. The output layer is fixed independently of the data

`Dense(7)` is hard-coded. Adding an eighth folder such as `contempt` makes the
generator produce eight-element targets while the model still emits seven
values, causing a shape mismatch.

The class list is also hard-coded separately in `emotion_webcam.py`. A safer
training pipeline saves `class_indices` to `labels.json`, and inference loads
that file.

### 4. Training is a minimal baseline

Missing training safeguards include:

- image augmentation;
- class weighting or balanced sampling;
- fixed random seeds;
- best-checkpoint saving;
- early stopping and learning-rate reduction;
- official train/validation/test split support;
- corrupted-image checks;
- transfer learning;
- calibration of output probabilities.

### 5. Real-time predictions will flicker

The top class is selected independently for every frame. There is no rolling
average, exponential moving average, confidence threshold, or minimum face
quality check. This makes labels unstable under normal motion and lighting.

### 6. Face detection is dated

The Haar cascade is fast, but it is fragile for side poses, partial faces,
glasses, low light, and multiple people. It also crops tightly without face
alignment. MediaPipe, YuNet, RetinaFace, or the detector bundled with a modern
FER library is a better later replacement.

### 7. Dependency and documentation cleanup is needed

- Both `opencv-python` and `opencv-contrib-python` are pinned. They install the
  same `cv2` namespace and should not normally be installed together.
- The README mentions `face-recognition` and `dlib`, but neither is required by
  the code or present in `requirements.txt`.
- The README training command should be `python train_emotion_model.py`, not
  `python train_emotion_model.h5`.
- The README run command should be `python emotion_webcam.py`, not
  `python webcam_recognition.py`.

## Seven Expressions or Eight?

There is no universally mandatory number of emotions.

- FER2013 uses seven: six basic emotions plus neutral.
- AffectNet and FERPlus can use eight by adding `contempt`.
- RAVDESS uses eight, but its extra class is `calm`, not `contempt`.
- GoEmotions uses 27 emotion categories plus neutral.
- Valence-arousal models represent emotion continuously rather than as a fixed
  list.

The supplied wheel is a **valence-arousal diagram**. Horizontal position is
positive/negative valence, vertical position is activation/arousal, and radius
indicates intensity. Labels such as `calm`, `relaxed`, `tense`, and `excited`
are examples placed in that continuous space; the diagram is not evidence that
the classifier must have exactly eight output neurons.

For this project, seven shared classes are the best version-1 choice because
they align exactly across:

- the current visual model;
- AffectNet-7 / EmotiEffLib;
- the recommended English text model;
- seven of the nine `emotion2vec+` outputs;
- MELD's aligned audio-video-text labels.

Use these canonical names everywhere:

```text
anger, disgust, fear, joy, neutral, sadness, surprise
```

Map `angry -> anger`, `happy/happiness -> joy`, `sad -> sadness`,
`fearful -> fear`, and `surprised -> surprise` at each adapter boundary.
Treat audio `other` and `unknown` as abstentions, not additional fused emotions.

## Reuse Decision

Keep:

- the existing OpenCV webcam proof of concept;
- the current model as a baseline and fallback;
- its 48 x 48 grayscale preprocessing only when that model is selected;
- the general folder-based training idea after fixing splits and metadata.

Replace or extend:

- the single-file real-time loop with modular visual, audio, text, and fusion
  services;
- hard-coded labels with one canonical schema;
- per-frame argmax with smoothed probability outputs;
- Haar detection when moving to the stronger visual model;
- ad hoc validation with repeatable per-modality and fused evaluation.
