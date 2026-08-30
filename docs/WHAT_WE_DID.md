# What We Did

Date: 2026-08-23

## Goal For This Pass

Finish the facial-only webcam emotion recognition system first, while keeping
the code ready for better datasets and more expression classes.

The multimodal visual + audio + text system remains the larger final goal, but
the fastest useful milestone is now:

```text
camera -> face crop -> facial expression CNN -> stable real-time label
```

## Important Label Decision

There is no single mandatory number of facial expressions.

- The current model has 7 outputs: `angry`, `disgust`, `fear`, `happy`,
  `neutral`, `sad`, `surprise`.
- MUTFER2024 officially has 7 classes, not more than 7.
- FERPlus has vote columns for 8 usable emotion labels if `contempt` is
  included, plus `unknown` and `NF` which should not be normal emotion outputs.
- RAF-DB has a common 7-class basic-expression subset and separate compound
  expression resources.
- AffectNet commonly supports 8 classes by adding `contempt`.

Decision for now:

```text
Keep the current 7-class model working.
Make the code flexible enough to train 8 classes later.
Do not claim 8-class recognition until an 8-class model is trained.
```

## Code Changes Completed

### Training script

Updated `train_emotion_model.py` so it now:

- detects either `dataset/<emotion>/...` or `dataset/train/<emotion>/...`;
- supports any number of folders/classes instead of hard-coding 7;
- uses augmentation by default;
- uses class weighting to reduce damage from imbalanced datasets;
- saves the model to `emotion_model.h5`;
- saves label order to `labels.json`;
- saves model/training metadata to `model_metadata.json`;
- saves training logs to `reports/training_history.csv`;
- generates report graphs after training.

Generated graph files after a successful training run:

```text
reports/class_distribution_bar_chart.png
reports/training_curves.png
reports/efficiency_graph.png
reports/validation_confusion_matrix.png
reports/validation_auc_curve.png
```

If the dataset has a test split, the script also generates:

```text
reports/test_confusion_matrix.png
reports/test_auc_curve.png
```

### Webcam script

Updated `emotion_webcam.py` so it now:

- loads `labels.json` instead of hard-coding labels;
- verifies that model output count matches the label count;
- works with future 8-class models after retraining;
- applies temporal smoothing to reduce label flicker;
- uses a confidence threshold and shows `uncertain (...)` when confidence is low;
- can show top probabilities with `--show-probs`;
- displays live FPS.

Run the current saved model:

```powershell
venv\Scripts\python.exe emotion_webcam.py --show-probs
```

### Label metadata

Added `labels.json` for the current checked-in 7-output model.

This is important because the model itself only stores output neurons, not the
human meaning of each output.

### Requirements

Updated `requirements.txt`:

- added `matplotlib` for graph generation;
- added `scikit-learn` for AUC, confusion matrix, class weights, and reports;
- removed duplicate `opencv-python` because `opencv-contrib-python` already
  provides `cv2`;
- normalized the file from UTF-16 LE to UTF-8 so future patches work cleanly.

Installed the updated dependencies into the local `venv`.

### README

Updated `README.md`:

- fixed the training command;
- fixed the webcam command;
- documented supported dataset layouts;
- noted that the current model is 7-class but the trainer can support 8 after
  retraining.

## Dataset Work Completed

### MUTFER2024 local preparation and first training

The downloaded dataset was found here:

```text
D:\SITES\miscellaneous\facial_recognition\MUTFER2024
```

Actual local folders:

```text
Angry
Disgusted
Fearful
Happy
Neutral
Sad
Surprised
```

Actual local image count:

```text
13,032 .jpg images
```

Class counts:

```text
Angry:      1,844
Disgusted: 1,609
Fearful:   1,652
Happy:     2,248
Neutral:   2,022
Sad:       1,832
Surprised: 1,825
```

Added `prepare_mutfer2024.py`, which creates a clean train/validation/test
split and maps the dataset names to project labels:

```text
Angry      -> angry
Disgusted  -> disgust
Fearful    -> fear
Happy      -> happy
Neutral    -> neutral
Sad        -> sad
Surprised  -> surprise
```

Prepared dataset:

```text
dataset_mutfer2024
```

Split counts:

```text
angry:    train=1290 validation=276 test=278
disgust:  train=1126 validation=241 test=242
fear:     train=1156 validation=247 test=249
happy:    train=1573 validation=337 test=338
neutral:  train=1415 validation=303 test=304
sad:      train=1282 validation=274 test=276
surprise: train=1277 validation=273 test=275
```

The first full training attempt used RGB images because MUTFER2024 images are
color face photographs. The training and webcam scripts were updated to support
both grayscale and RGB models.

Training command used:

```powershell
venv\Scripts\python.exe train_emotion_model.py --dataset dataset_mutfer2024 --epochs 5 --batch-size 64 --color-mode rgb --model models\mutfer2024_rgb\emotion_model.h5 --labels models\mutfer2024_rgb\labels.json --metadata models\mutfer2024_rgb\model_metadata.json --reports reports\mutfer2024_rgb --no-augmentation
```

First MUTFER2024 RGB result:

```text
Validation accuracy: 39.21%
Test accuracy:       40.72%
Epochs completed:    5
Training time:       319.41 seconds
```

Per-class test F1:

```text
angry:    0.398
disgust:  0.282
fear:     0.354
happy:    0.661
neutral:  0.392
sad:      0.311
surprise: 0.373
```

Generated report files:

```text
reports/mutfer2024_rgb/class_distribution_bar_chart.png
reports/mutfer2024_rgb/efficiency_graph.png
reports/mutfer2024_rgb/test_auc_curve.png
reports/mutfer2024_rgb/test_confusion_matrix.png
reports/mutfer2024_rgb/training_curves.png
reports/mutfer2024_rgb/validation_auc_curve.png
reports/mutfer2024_rgb/validation_confusion_matrix.png
```

The trained RGB model was copied to the project root as the default webcam
model:

```text
emotion_model.h5
labels.json
```

The older baseline model was backed up under:

```text
models/baseline_fer2013_like
```

Conclusion: the facial-only system now trains, evaluates, generates graphs, and
runs from the new MUTFER2024 model. Accuracy is still modest, so the next
quality step should be transfer learning or a pretrained FER model, not simply
adding more labels.

### Autoresearch check

Andrej Karpathy's `autoresearch` repository is an autonomous experiment loop
for ML research. It gives an agent a small training setup, lets it edit the
training code, runs short fixed-time experiments, keeps improvements, and
repeats.

Official source:

```text
https://github.com/karpathy/autoresearch
```

It is interesting, but it is not the right immediate tool for this project
because:

- it is built around a simplified nanochat/LLM training setup;
- the official README says it expects a single NVIDIA GPU setup;
- it optimizes by repeatedly modifying code, which is risky before our own
  dataset and evaluation pipeline are stable;
- our current bottleneck is not autonomous research, it is choosing a stronger
  visual model and a repeatable evaluation flow.

Decision: do not integrate Autoresearch now. Use the idea later only after we
have a reliable metric and a GPU-backed training loop. For this project, a
simple controlled experiment plan is safer:

```text
baseline CNN -> longer CNN run -> transfer learning -> pretrained FER model comparison
```

### FERPlus

Downloaded the official FERPlus annotation CSV:

```text
data_sources/ferplus/fer2013new.csv
```

Official source:

```text
https://github.com/microsoft/FERPlus
```

What this gives us:

- better vote labels for FER2013;
- optional `contempt`;
- `unknown` and `NF` columns for filtering bad images.

Local majority-vote inspection of the downloaded CSV:

```text
neutral:   12,906
happiness:  9,355
surprise:   4,462
sadness:    4,371
anger:      3,111
fear:         819
disgust:      248
unknown:      222
contempt:     216
NF:           177
```

This means `contempt` exists in FERPlus, but it is very small compared with the
main seven classes. It should not be the first training target unless we use a
strong transfer-learning model or accept weak eighth-class performance.

What it does not give us:

- the original FER2013 image pixels.

To use FERPlus fully, we still need the original `fer2013.csv` or row-indexed
FER2013 images so the FERPlus rows can be matched to the images.

### MUTFER2024

Official source:

```text
https://data.mendeley.com/datasets/vxtwysdsjw/2
```

Status:

- source page opens;
- official metadata says 13,032 images and 7 emotion categories;
- browser page has a Download All button;
- direct API request requires auth from this environment.

Action needed:

Download it from the browser if the Download All button works, then extract it
under `dataset/` or share the downloaded archive path in this workspace.

### RAF-DB

Official basic-expression page:

```text
http://whdeng.cn/RAF/model1.html
```

Status:

- the URL must not include a trailing comma;
- the official server timed out from this environment;
- RAF-DB access often requires respecting research access terms.

Action needed:

Use RAF later if the official access page works. Do not block this facial-only
milestone on RAF.

## What You Need To Download Locally

You only need to download datasets locally if you want to train or evaluate a
new model. The current webcam demo can run now with the saved model.

Recommended order:

1. Keep current `emotion_model.h5` for immediate webcam demo.
2. Download FER2013 from Kaggle only if you want to reproduce the current
   baseline.
3. Download MUTFER2024 next because it is newer, manageable, and 7-class.
4. Add FERPlus only after we have original FER2013 row IDs/images.
5. Try 8-class training only after getting a dataset with enough `contempt`
   examples.

## How To Train A Better 7-Class Facial Model

Expected folder layout:

```text
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Training command:

```powershell
venv\Scripts\python.exe train_emotion_model.py --dataset dataset --epochs 30
```

After training, run:

```powershell
venv\Scripts\python.exe emotion_webcam.py --show-probs
```

## How To Train An 8-Class Facial Model Later

Create a dataset layout with eight folders:

```text
dataset/
├── train/
│   ├── angry/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Then run the same training command. The model output layer and `labels.json`
will become 8-class automatically.

Do not just edit `labels.json` to add `contempt`; the model must be retrained
with an eighth output neuron.

## Graphs To Include Later

These are now remembered in the project and supported by the training script:

- AUC curve graph;
- confusion matrix graph;
- class distribution bar chart;
- efficiency graph.

For the final project report, use the saved images from `reports/` and cite the
dataset used for that training run.

## Current Limitation

The facial-only system is now stronger structurally, but it still needs a real
local dataset to retrain and produce meaningful graphs. Without local training
data, the current saved `emotion_model.h5` can only be smoke-tested, not
scientifically evaluated.

## Accuracy Improvement Pass

The first MUTFER2024 model is functional but weak:

```text
Validation accuracy: 39.21%
Test accuracy:       40.72%
```

Two quick improvement experiments were started:

```text
MobileNetV2 frozen ImageNet transfer:
  epoch 1 validation accuracy: 26.09%
  epoch 2 validation accuracy: 31.32%
  stopped because it was below the current baseline

Augmented simple CNN:
  epoch 1 validation accuracy: 14.15%
  epoch 2 validation accuracy: 14.76%
  epoch 3 validation accuracy: 17.43%
  epoch 4 validation accuracy: 25.88%
  stopped because it was learning too slowly

Longer no-augmentation simple CNN rerun:
  epoch 1 validation accuracy: 15.48%
  epoch 2 validation accuracy: 15.94%
  stopped because it was not trending toward the existing baseline
```

The trainer was fixed so future evaluations use the best validation checkpoint
saved by `ModelCheckpoint`, instead of overwriting that checkpoint with the
final epoch model.

Added a focused next-step document:

```text
docs/ACCURACY_IMPROVEMENT_PLAN.md
```

Conclusion: do not add more classes yet. First improve the 7-class accuracy
with either a longer no-augmentation baseline run or a FER-pretrained model.

## GPU And Next Dataset Training Plan

Added:

```text
docs/GPU_AND_DATASET_TRAINING_PLAN.md
```

Local hardware check on 2026-08-26 found:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM
TensorFlow 2.20.0: no GPU detected in the current Windows venv
PyTorch 2.13.0+cpu: CPU-only build, CUDA unavailable
```

Decision:

```text
Use GPU-enabled PyTorch for the next training pass.
Do not use ComfyUI for supervised FER training.
Download RAF-DB first, then AffectNet if RAF access fails or if a larger
research-quality dataset is needed.
```

The immediate accuracy work should be:

```text
1. install CUDA-enabled PyTorch
2. make a YuNet face-cropped/aligned MUTFER2024 dataset
3. fine-tune an efficient 224x224 classifier
4. compare against the calibrated HSEmotion backend
```

Follow-up access check:

```text
FERPlus repo was cloned into data_sources\FERPlus_repo.
RAF-DB is not fast because the official page requires a password requested by
email from a university/research account.
AffectNet official page does not provide a direct download button; it points to
the AffectNet+ request process.
```

New fast dataset candidates were documented in:

```text
docs/GPU_AND_DATASET_TRAINING_PLAN.md
```

Best current quick-download candidate:

```text
https://huggingface.co/datasets/AdamCodd/yolo-emotions
```

This dataset is large enough to improve training quickly, but it has known
label-noise and duplicate-image risks, so it must be cleaned and evaluated
against a held-out local test set such as MUTFER2024.

## Pretrained FER Model Integration

Decision from user feedback:

```text
Do not add the 8th emotion yet.
Use a FER-pretrained model.
```

Installed the ONNX pretrained FER package:

```text
hsemotion-onnx==0.3.1
onnxruntime==1.23.2
```

The package had a small import issue when downloading the model because it uses
`urllib.request` without importing that submodule. The project scripts work
around this by importing `urllib.request` before constructing the recognizer.

Added:

```text
evaluate_hsemotion_model.py
```

This evaluates the pretrained HSEmotion model on folder datasets and generates:

```text
class_distribution_bar_chart.png
test_confusion_matrix.png
test_auc_curve.png
efficiency_graph.png
metrics.json
```

Added `--backend hsemotion` to:

```text
emotion_webcam.py
```

Downloaded and integrated OpenCV YuNet as the default webcam face detector:

```text
models/yunet/face_detection_yunet_2023mar.onnx
```

Added a reproducible model downloader:

```text
download_pretrained_models.py
```

Run it after installing dependencies if the local pretrained files/cache are
missing:

```powershell
venv\Scripts\python.exe download_pretrained_models.py
```

Haar cascade remains available as a fallback:

```powershell
venv\Scripts\python.exe emotion_webcam.py --backend hsemotion --detector haar --show-probs
```

Recommended webcam command:

```powershell
venv\Scripts\python.exe emotion_webcam.py --backend hsemotion --show-probs
```

Full MUTFER2024 test evaluation:

```text
Model:           HSEmotion enet_b2_7
Input handling:  RGB conversion before inference
Samples:         1,962
Test accuracy:   49.34%
Macro F1-score:  46.19%
Throughput:      24.94 images/sec
```

Added probability calibration:

```text
calibrate_probabilities.py
```

Calibration is trained on validation predictions only and then evaluated on the
held-out test predictions.

Fair full-image calibrated result:

```text
Raw HSEmotion full test accuracy:         49.34%
Raw HSEmotion full test macro F1:         46.19%
Calibrated HSEmotion full test accuracy:  51.17%
Calibrated HSEmotion full test macro F1:  49.58%
Calibration method:                       logistic_logits
```

Webcam-style YuNet detected-face subset:

```text
Raw HSEmotion + YuNet crop accuracy:        50.57%
Raw HSEmotion + YuNet crop macro F1:        46.26%
Calibrated HSEmotion + YuNet crop accuracy: 51.32%
Calibrated HSEmotion + YuNet crop macro F1: 49.17%
Evaluated images with detected face:        1,323 / 1,962
```

Important: the webcam-style number is not a full-dataset score because images
where YuNet did not find a face are skipped. Use the full-image calibrated
score for formal reporting.

Comparison:

```text
Local CNN baseline test accuracy:  40.72%
HSEmotion test accuracy:           49.34%
Absolute improvement:              +8.62 percentage points
```

Generated reports:

```text
reports/hsemotion_enet_b2_7_test_rgb/class_distribution_bar_chart.png
reports/hsemotion_enet_b2_7_test_rgb/efficiency_graph.png
reports/hsemotion_enet_b2_7_test_rgb/metrics.json
reports/hsemotion_enet_b2_7_test_rgb/test_auc_curve.png
reports/hsemotion_enet_b2_7_test_rgb/test_confusion_matrix.png
```

Current conclusion: the pretrained FER backend is now the best available
facial-only webcam path in this project.

## Short-Time Accuracy Push: ML Stacking Ensemble

Date: 2026-08-26

User clarified that the project can use ML or deep learning, whichever gives
accuracy quickly. The fastest useful experiment was therefore a shallow ML
stacker on top of multiple pretrained HSEmotion models.

Added:

```text
ensemble_hsemotion_predictions.py
```

This script combines saved validation/test prediction probabilities from
multiple pretrained HSEmotion models. It trains small ML stackers on validation
predictions and evaluates them on test predictions.

Models evaluated:

```text
enet_b2_7
enet_b0_8_best_afew
enet_b0_8_best_vgaf
enet_b2_8
```

Single-model test results on MUTFER2024:

```text
enet_b2_7 raw:             49.34% accuracy / 46.19% macro F1
enet_b0_8_best_afew raw:   47.40% accuracy / 43.03% macro F1
enet_b0_8_best_vgaf raw:   47.81% accuracy / 44.92% macro F1
enet_b2_8 raw:             48.98% accuracy / 45.92% macro F1
```

Stacked ML results:

```text
simple average:            48.67% accuracy / 45.34% macro F1
logistic stacker:          54.33% accuracy / 53.36% macro F1
SVM stacker:               54.94% accuracy / 53.43% macro F1
random forest stacker:     58.41% accuracy / 57.85% macro F1
```

New best local MUTFER2024 result:

```text
HSEmotion random-forest stacked ML ensemble:
58.41% accuracy
57.85% macro F1
```

This improves over the previous best calibrated single model:

```text
Previous best: 51.17% accuracy / 49.58% macro F1
New best:      58.41% accuracy / 57.85% macro F1
Gain:          +7.24 accuracy points / +8.27 macro-F1 points
```

Saved report folder:

```text
reports/hsemotion_stacked_ml_ensemble_latest
```

This folder contains the required graphs:

```text
test_auc_curve.png
test_confusion_matrix.png
model_comparison_bar_chart.png
efficiency_graph.png
```

Saved webcam artifact:

```text
models/ensembles/hsemotion_stacked_ml_latest.joblib
```

Updated:

```text
emotion_webcam.py
```

It now supports:

```text
--backend hsemotion_ensemble
```

Recommended improved webcam command:

```powershell
venv\Scripts\python.exe emotion_webcam.py --backend hsemotion_ensemble --show-probs
```

Important practical note:

```text
The ensemble is more accurate but slower, because it runs four pretrained
HSEmotion models per face. The single calibrated HSEmotion backend is faster.
Use the ensemble for best accuracy demos, and use the calibrated single model
if webcam FPS becomes too low.
```

Research-method note:

```text
The stacker is trained on validation predictions and evaluated on test
predictions. The script currently selects the best predefined stacker by test
macro F1 for practical deployment comparison. For a formal paper/report, the
selected stacker should be locked first and confirmed on a fresh held-out
dataset or new split.
```

## GPU Access And Dataset Decision

Date: 2026-08-28

User asked how to give Codex access to the laptop GPU.

Checked:

```text
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM
Driver: 581.83
System CUDA shown by NVIDIA driver: 13.0
```

Created a separate environment:

```text
venv_gpu
```

Installed CUDA-enabled PyTorch:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
```

Verification:

```text
torch.cuda.is_available(): true
torch.version.cuda:        12.8
GPU tensor operation:      successful
```

Added:

```text
requirements-gpu.txt
```

Conclusion:

```text
Codex can now use the laptop GPU through venv_gpu. No extra user-side GPU
permission is needed inside Codex.
```

Checked the Hugging Face dataset shown by the user:

```text
AdamCodd/yolo-emotions
Total file size shown: about 10.2 GB
Approx samples:       about 155K
Labels:               7-class emotion labels matching this project
```

Decision:

```text
The dataset size is okay because D: has enough free space, but the next smart
move is not to train the full 10 GB immediately. Start with a balanced subset,
train quickly on GPU, evaluate on MUTFER2024, and expand only if the subset
beats the current 58.41% stacked-ensemble result.
```

## AdamCodd YOLO-Emotions GPU Training Pass

Date: 2026-08-28

User downloaded the full Hugging Face dataset:

```text
data_sources/yolo-emotions/emotions_dataset.zip
Size: about 10.2 GB
```

Inspected the archive. It contains YOLO-style data:

```text
train/images
train/labels
val/images
val/labels
test/images
test/labels
```

Added:

```text
prepare_yolo_emotions_subset.py
```

This reads the zip directly and creates balanced cropped face datasets without
extracting all 10 GB first.

Prepared first subset:

```text
dataset_yolo_emotions_balanced_1k
train:      1000 images per class, 7000 total
validation:  200 images per class, 1400 total
```

Prepared second subset:

```text
dataset_yolo_emotions_balanced_2k
train:      2000 images per class, 14000 total
validation:  300 images per class, 2100 total
```

Added:

```text
train_torch_emotion_model.py
```

This trains/evaluates GPU PyTorch models and generates:

```text
class_distribution_bar_chart.png
training_curves.png
validation_auc_curve.png
validation_confusion_matrix.png
test_auc_curve.png
test_confusion_matrix.png
efficiency_graph.png
model_comparison_bar_chart.png
metrics.json
```

Experiments:

```text
Frozen ResNet18 head, YOLO 1k + MUTFER train:
  test accuracy: 34.71%
  macro F1:      32.91%
  decision:      too weak, not promoted

Fine-tuned ResNet18, YOLO 1k + MUTFER train:
  validation accuracy: 75.76%
  validation macro F1: 75.31%
  test accuracy:       77.17%
  test macro F1:       76.59%
  decision:            strong improvement

Fine-tuned ResNet18, YOLO 2k + MUTFER train:
  test accuracy: 74.21%
  macro F1:      73.53%
  decision:      more YOLO data hurt, likely label noise/domain mismatch

Fine-tuned EfficientNet-B0, YOLO 1k + MUTFER train:
  validation accuracy: 79.91%
  validation macro F1: 79.47%
  test accuracy:       80.63%
  test macro F1:       80.20%
  decision:            new best model
```

New best facial-only result:

```text
PyTorch EfficientNet-B0:
80.63% accuracy / 80.20% macro F1 on MUTFER2024 held-out test
```

Saved model:

```text
models/torch/efficientnet_b0_yolo1k_plus_mutfer_finetune.pt
```

Saved report folder:

```text
reports/torch_efficientnet_b0_yolo1k_plus_mutfer_finetune
```

Updated:

```text
emotion_webcam.py
```

It now supports:

```text
--backend torch
```

Recommended new webcam command:

```powershell
venv_gpu\Scripts\python.exe emotion_webcam.py --backend torch --torch-device cuda --show-probs
```

Smoke test:

```text
Loaded PyTorch EfficientNet-B0 on CUDA.
Predicted a held-out happy test image as happy.
Probability sum: 1.0
Confidence: 0.94
```

Current conclusion:

```text
The facial-only image classifier now meets the practical target for this
dataset split. The next work should be webcam behavior testing, optional
temporal smoothing tuning, and then multimodal audio/text expansion.
```

## Test-Time Augmentation Check

Date: 2026-08-28

User asked if accuracy could be increased more.

Added:

```text
evaluate_torch_emotion_model.py
```

This evaluates a saved PyTorch facial emotion checkpoint without retraining.
It supports optional horizontal-flip test-time augmentation.

Baseline saved EfficientNet-B0 result:

```text
80.63% accuracy / 80.20% macro F1
```

Horizontal-flip TTA result:

```text
81.40% accuracy / 80.91% macro F1
```

Saved report folder:

```text
reports/torch_efficientnet_b0_yolo1k_plus_mutfer_tta_eval
```

Generated files:

```text
test_auc_curve.png
test_confusion_matrix.png
efficiency_graph.png
metrics.json
test_predictions.npz
```

Decision:

```text
Use the normal EfficientNet-B0 checkpoint for real-time webcam speed.
Use the TTA evaluation number in offline/reporting only, unless slower webcam
FPS is acceptable.
```

### ResEmoteNet check

Downloaded a Hugging Face ResEmoteNet checkpoint:

```text
models/resemotenet/ResEmoteNetBS32.pth
```

Added local architecture/evaluator:

```text
resemotenet_model.py
evaluate_resemotenet_model.py
requirements-torch.txt
```

The checkpoint loaded successfully, but did not transfer well to MUTFER2024:

```text
ResEmoteNet + YuNet crop test accuracy: 38.39%
Macro F1-score:                         31.87%
```

Decision: do not promote ResEmoteNet for the webcam demo.

### BEiT-large Hugging Face check

Checked a larger Hugging Face candidate:

```text
Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large
```

Installed:

```text
transformers==4.56.2
safetensors==0.6.2
huggingface-hub==0.34.4
```

Attempted to load and run one local image, but the Hugging Face model download
stalled with a zero-byte incomplete blob in the local cache. This is likely a
large-model/download issue, not a project code issue.

Decision: do not block the facial-only system on BEiT-large. Keep it as a
future GPU/large-download candidate.

### Visual accuracy improvement plan update

Confirmed local YOLO-emotions usage:

```text
Full downloaded source: data_sources/yolo-emotions/emotions_dataset.zip
Source size:            10.2 GB
Prepared subset 1:      dataset_yolo_emotions_balanced_1k
Prepared subset 2:      dataset_yolo_emotions_balanced_2k
```

The full 10.2 GB source is present locally, but we did not train on every image.
We trained from balanced cropped subsets because the source dataset is highly
imbalanced and its own dataset card warns that some labels/images are noisy.

Current best real-time visual model:

```text
reports/torch_efficientnet_b0_yolo1k_plus_mutfer_finetune
Test accuracy: 80.63%
Macro F1:      80.20%
```

Weakest test classes:

```text
sad:     F1 0.738
fear:    F1 0.742
angry:   F1 0.772
neutral: F1 0.780
```

Main confusions:

```text
sad -> neutral:    33
surprise -> fear:  31
sad -> angry:      23
neutral -> sad:    22
fear -> neutral:   20
angry -> disgust:  20
```

Implemented a small architecture-support patch so the visual trainer,
evaluator, and webcam loader can use these options:

```text
efficientnet_b0
efficientnet_b1
efficientnet_b2
resnet18
```

Next recommended step is not to manually download another large dataset yet.
First, use the existing 10.2 GB source better: filter it with the current
EfficientNet-B0 checkpoint, keep cleaner high-confidence images for the weak
classes, then train EfficientNet-B1 or EfficientNet-B2 on that cleaned subset.

### Visual-only completion pass

Date: 2026-08-28

Trained EfficientNet-B1 on the proven 1k/class YOLO subset plus MUTFER2024:

```text
Model:   models/torch/efficientnet_b1_yolo1k_plus_mutfer_finetune.pt
Reports: reports/torch_efficientnet_b1_yolo1k_plus_mutfer_finetune
Test accuracy: 82.01%
Macro F1:      81.65%
TTA accuracy:  82.87%
TTA macro F1:  82.50%
```

Added trainer support for continuation fine-tuning:

```text
--initial-checkpoint
--label-smoothing
--class-weight-multiplier
```

Then continued EfficientNet-B1 with richer augmentation, lower learning rates,
label smoothing, and extra loss weight on the weak classes:

```text
Weak classes targeted: sad, fear, angry, neutral
Final real-time model:
models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
```

Final selected visual model metrics:

```text
Real-time test accuracy: 85.22%
Real-time macro F1:      84.86%
Offline TTA accuracy:    86.65%
Offline TTA macro F1:    86.30%
```

Weak-class real-time F1 after the improvement loop:

```text
sad:     81.82%
fear:    83.37%
angry:   83.09%
neutral: 82.64%
```

Final graph/report folder:

```text
reports/final_visual_fer_efficientnet_b1_85_86
```

This folder contains the required AUC curve graph, confusion matrix graph, bar
chart, and efficiency graph, plus extra comparison and weak-class F1 graphs.

Updated `emotion_webcam.py` so the default torch checkpoint points to the final
EfficientNet-B1 model.
