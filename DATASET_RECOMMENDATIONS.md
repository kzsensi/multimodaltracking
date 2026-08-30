# Dataset Recommendations for This Facial Emotion Recognition Project

Research date: 2026-08-23

This file focuses on visual facial-expression datasets. The completed project
audit, audio/text/multimodal dataset research, pretrained model choices, and
quick implementation architecture are in [`docs/`](docs/README.md).

## What the Current Code Expects

The current project is a static-image FER pipeline, not a video/emotion-sequence
pipeline.

- `train_emotion_model.py` reads from `dataset/`.
- Images are resized to `48x48`.
- Images are loaded as grayscale.
- The model outputs 7 classes.
- `emotion_webcam.py` expects this exact order:
  `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

Best drop-in datasets are therefore image datasets that can be organized like:

```text
dataset/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
```

If a dataset uses names like `anger`, `happiness`, or `sadness`, rename them to
the current labels before training. If it includes `contempt`, `unknown`, or
`not face`, either drop those samples or change the model output layer and
webcam labels.

## Top Choices

### 1. RAF-DB

Strong benchmark if official access succeeds, but not the best first download
for a time-limited build because the official server is currently unreliable.

- Type: static facial expression images, in the wild.
- Size: about 30K images; the commonly used basic-expression split is moderate.
- Labels: seven basic expressions plus a separate compound-expression subset.
- Why use it: much more realistic than FER2013, widely used in modern FER papers,
  and still small enough for this project.
- Fit with current code: high. Use the seven basic-expression subset and convert
  or rename labels to the current folder names.
- Caveat: access/licensing terms should be checked; many mirrors exist, but the
  official database is research-oriented.

Official source (no trailing comma):
http://whdeng.cn/RAF/model1.html

If that page does not open, try the parent project page:

- https://www.whdeng.cn/Emotion/projects.html

There are Kaggle copies of RAF-DB, but treat them as unofficial mirrors and
verify licensing/provenance before using them in a public or submitted project.

### 2. MUTFER2024

Best recent small/medium dataset to add diversity.

- Type: static facial emotion images.
- Published: 2025-04-02 on Mendeley Data.
- Size: 13,032 images.
- Labels: seven emotion categories.
- License: CC BY 4.0.
- Why use it: newer, not huge, and valuable because it focuses on South African
  participants under real-world conditions, which can help reduce demographic
  brittleness.
- Fit with current code: likely high, after folder normalization and grayscale
  resizing.
- Caveat: newer and less benchmark-standard than RAF-DB/AffectNet, so treat it
  as a useful training supplement and generalization test, not the only dataset.

Source:
https://data.mendeley.com/datasets/vxtwysdsjw/2

### 3. FERPlus

Best minimal-change improvement if you already have FER2013.

- Type: relabeled FER2013.
- Size: same base images as FER2013, with problematic samples marked separately.
- Labels: vote distributions from multiple annotators; includes extra labels such
  as contempt, unknown, and not-face.
- Why use it: it fixes a major FER2013 weakness: noisy single labels.
- Fit with current code: very high if you collapse labels to the seven current
  classes and remove `contempt`, `unknown`, and `NF`.
- Caveat: not actually a new image dataset; it is a better annotation layer over
  the old FER2013 images.

Source:
https://github.com/microsoft/FERPlus

### 4. BTFER

Best small modern balanced test set.

- Type: static in-the-wild FER images.
- Published: 2024 paper.
- Size: 2,100 images.
- Labels: seven basic emotions.
- Why use it: balanced and designed for cross-domain validation. Excellent for
  checking whether your model generalizes beyond its training dataset.
- Fit with current code: high, but size is too small for primary training.
- Caveat: use it mainly as a final test/benchmark set.

Sources:
https://www.6gflagship.com/publications/benchmarking-deep-facial-expression-recognition-an-extensive-protocol-with-balanced-dataset-in-the-wild/
https://doi.org/10.1016/j.engappai.2024.108983

### 5. ExpW

Best older but strong large static in-the-wild image dataset.

- Type: static web face images.
- Size: 91,793 manually labeled faces.
- Labels: seven basic expressions.
- Why use it: significantly larger than FER2013 and closer to real-world images.
- Fit with current code: high after converting folders/labels.
- Caveat: older than RAF-DB/AffectNet, and quality can vary because images came
  from web search.

Sources:
https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html
https://arxiv.org/abs/1609.06426

## Strong But Heavier Options

### AffectNet+

Best research-grade modern dataset if you can get access.

- Type: large-scale facial expression dataset based on AffectNet, with soft
  labels and metadata.
- Released/announced: 2024 paper; official page now supports requests.
- Size: reprocesses roughly 1M AffectNet images.
- Labels: soft emotion vectors and metadata such as demographics, landmarks, head
  pose, and embeddings.
- Why use it: most modern choice for robust/fairness-aware FER research.
- Fit with current code: medium. You should sample a balanced seven-class subset,
  or update the model to support soft labels and possibly an eighth class.
- Caveat: academic/research request process; too large for quick student-style
  retraining unless you subset it.

Sources:
https://mohammadmahoor.com/pages/databases/affectnetplus/
https://arxiv.org/abs/2410.22506

### AffectNet

Best large classic research dataset.

- Type: static facial images in the wild.
- Size: more than 1M collected images, with roughly 440K manually annotated.
- Labels: categorical expressions plus valence/arousal.
- Why use it: one of the strongest standard datasets for robust FER.
- Fit with current code: medium to high if you use only the seven matching
  categorical classes and convert to folder format.
- Caveat: large and access-controlled for research; may include an extra contempt
  class depending on the subset/annotation file.

Source:
https://mohammadmahoor.com/pages/databases/affectnet/

### CAER-S

Best if you want images sampled from videos and context-aware emotion data.

- Type: static frame images sampled from the CAER video benchmark.
- Size: about 70K frame images.
- Labels: seven emotion categories.
- Why use it: useful when lighting, scene context, and TV/video conditions matter.
- Fit with current code: medium. Your current code trains on face crops, while
  CAER-S is context-aware; for your webcam model, crop detected faces before
  training or explicitly decide to train with scene context.
- Caveat: larger download, about 13.5 GB for CAER-S.

Source:
https://caer-dataset.github.io/

### GFFD-2025

Best very recent small controlled dataset.

- Type: static images with genuine/fake expression annotations.
- Published: 2025-10-13 on Mendeley Data.
- Size: about 1,900 raw facial images plus about 1,500 cropped/augmented images.
- Labels: seven primary emotions, each split into genuine vs fake/acted.
- Why use it: useful for experiments around acted vs authentic expressions.
- Fit with current code: medium to high if you flatten genuine/fake folders into
  the seven emotion folders.
- Caveat: small and controlled indoor data, so it should supplement, not replace,
  in-the-wild training data.

Source:
https://data.mendeley.com/datasets/wmfd4p3z32/1

## Video Datasets for a Future Upgrade

These are good datasets, but your current model uses one 48x48 grayscale image
at a time. Use these only if you extract frames and face crops, or if you later
upgrade to an LSTM/Transformer/3D-CNN style video model.

### DFEW

- Type: dynamic facial expression videos in the wild.
- Size: 16,372 video clips from movies.
- Labels: seven classic expressions plus expression distribution vectors.
- Source: https://dfew-dataset.github.io/

### FERV39k

- Type: dynamic FER video clips across multiple scenes.
- Size: 38,935 video clips.
- Labels: seven classic expressions.
- Source: https://github.com/wangyanckxx/FERV39k

### MAFW

- Type: multimodal video-audio clips with compound affect annotations.
- Size: 10,045 clips.
- Labels: 11 single emotion classes and 32 multi-label compound classes.
- Source: https://mafw-database.github.io/MAFW/

## Recommended Path for This Repo

1. Use MUTFER2024 as the first new dataset because it is directly accessible,
   clearly licensed, manageable, and already has seven categories.
2. Apply FERPlus labels to the existing FER2013 images for a cleaner baseline.
3. Use BTFER only as a held-out benchmark.
4. Use RAF-DB if its official research access becomes available.
5. If you can get academic access, try a balanced subset of AffectNet+.
6. Avoid random Kaggle repacks unless they clearly cite their original source and
   preserve label quality.

For the most practical drop-in route:

```text
MUTFER2024 -> normalize folder names -> dataset/ -> python train_emotion_model.py
```

For the best research-grade route:

```text
FERPlus + MUTFER2024 + optional RAF-DB/AffectNet subset -> normalized labels
-> train with class weights and evaluate on a fixed independent set such as BTFER
```

## Small Code Notes

- The README has command typos: it says `python train_emotion_model.h5` and
  `python webcam_recognition.py`, but the actual files are
  `train_emotion_model.py` and `emotion_webcam.py`.
- `ImageDataGenerator(validation_split=0.2)` is convenient, but for datasets that
  already provide official train/test splits, it is better to update the training
  script to respect those splits.
- Add `class_weight` during training, because FER datasets usually have weak
  `disgust` and `fear` class counts.
- Consider saving `train_data.class_indices` with the model. Right now webcam
  inference assumes the hard-coded label order matches training.
