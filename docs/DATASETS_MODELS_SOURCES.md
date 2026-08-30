# Datasets, Models, and Source-Checked Recommendations

Research date: 2026-08-23

## Short Answer

For the quickest working system:

- **Do not wait to train all three branches.** Use pretrained models first.
- Keep seven common classes for fusion.
- Use MUTFER2024 or FERPlus to improve the visual branch later.
- Use RAVDESS and CREMA-D for understandable speech experiments, but use a
  pretrained `emotion2vec+` model in the application.
- Use GoEmotions and DailyDialog for text experiments, but use the seven-class
  DistilRoBERTa model in the application.
- Use MELD for aligned audio-video-text evaluation and future learned fusion.

## Visual Datasets

## About More Than Seven Facial Expressions

The code can now train more than seven facial classes if the local dataset
folders contain those classes. The dataset choice decides the valid output
labels:

- MUTFER2024: seven categories.
- FERPlus: eight emotion vote columns if `contempt` is included, plus
  `unknown` and `NF` columns that should be filtered or treated as abstentions.
- RAF-DB: common seven-class basic-expression subset plus separate compound
  expression resources.
- AffectNet: often used as seven-class or eight-class with `contempt`.

For a quick reliable facial-only system, train seven first. Add an eighth class
only when there are enough `contempt` examples and the report clearly says the
model is facial-only 8-class, not multimodal 8-class.

In the downloaded FERPlus annotation file, majority-vote `contempt` appears in
only 216 rows overall and 165 training rows, while `neutral` appears in 12,906
rows and `happiness` in 9,355 rows. This is another reason not to rush the
eighth class in the first working build.

### Current FER2013 Kaggle copy

Source: https://www.kaggle.com/datasets/msambare/fer2013

- 35,887 grayscale 48 x 48 face images.
- Seven classes: anger, disgust, fear, happiness, neutral, sadness, surprise.
- Common split: 28,709 train and 7,178 test images.
- Strong imbalance: common training counts include 7,215 `happy` but only 436
  `disgust` samples.
- Advantages: already preprocessed, small, and exactly compatible with the
  current network.
- Problems: old, low resolution, noisy labels, poor crops/non-faces, and class
  imbalance.

Decision: retain it as a baseline, not as the only evidence that the final
system generalizes.

Original challenge report:
https://arxiv.org/abs/1307.0414

### MUTFER2024 - recommended easy new visual dataset

Official source: https://data.mendeley.com/datasets/vxtwysdsjw/2

- Version 2 published 2025-04-02.
- 13,032 images from 300 participants across happy, sad, angry, surprised,
  neutral, disgusted, and fearful.
- Real-world variation in lighting, background, and accessories.
- South African participants add useful demographic variety.
- CC BY 4.0 and direct Mendeley access.

Decision: the best first new dataset for this project because its source works,
its license is clear, its size is manageable, and its seven classes fit.

### FERPlus - recommended relabeling upgrade

Official repository: https://github.com/microsoft/FERPlus

- Uses the same FER2013 images with labels from ten crowd annotators.
- Provides vote counts for neutral, happiness, surprise, sadness, anger,
  disgust, fear, contempt, unknown, and not-face.
- The repository was archived in 2024 but remains accessible.
- The Microsoft code is MIT licensed; the underlying FER2013 images must still
  be obtained from the original challenge/Kaggle source.

Decision: use the improved votes for the existing images. For the seven-class
system, remove `contempt`, `unknown`, and `NF`, or use soft labels only for the
seven shared classes.

### RAF-DB - strong benchmark, unreliable access page

The correct URL has **no trailing comma**:

```text
http://whdeng.cn/RAF/model1.html
```

The URL in the request ends with `model1.html,`; that comma becomes part of the
address and will fail. Even with the comma removed, the official RAF server is
currently unreliable and returned gateway/fetch errors during this research.

Paper PDF that currently resolves:
https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf

- About 29,672 real-world images overall.
- The basic-expression subset uses seven categories.
- Widely used for in-the-wild FER.
- Research access and usage terms must be respected.

Decision: excellent if official access succeeds, but do not block the quick
build on it. Prefer MUTFER2024 or FERPlus instead of an unverified mirror.

### AffectNet / AffectNet+

Official source: https://www.mohammadmahoor.com/

- AffectNet contains roughly one million collected images, with hundreds of
  thousands manually annotated for categorical emotion and valence/arousal.
- AffectNet can be used with seven classes or eight by including `contempt`.
- AffectNet+ adds soft labels and metadata for about one million faces and was
  published in IEEE Transactions on Affective Computing in 2025.
- Access is research-oriented and the full scale is unnecessary for version 1.

Decision: use a pretrained AffectNet-7 model now; request the dataset only for a
later research-quality training phase.

### BTFER

Sources:

- https://www.6gflagship.com/publications/benchmarking-deep-facial-expression-recognition-an-extensive-protocol-with-balanced-dataset-in-the-wild/
- https://doi.org/10.1016/j.engappai.2024.108983

- 2,100 balanced in-the-wild images.
- Seven basic expression classes.
- Designed as a cross-dataset benchmark.

Decision: reserve it as a small independent test set; do not train on it first.

## Audio and Audio-Visual Datasets

### RAVDESS - recommended small speech starting point

Official source: https://zenodo.org/records/1188976

- 7,356 files across audio-only, audio-video, and video-only formats.
- 24 professional actors.
- Speech labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised.
- The speech-only audio archive is 1,440 files and about 208-215 MB, so the
  entire 24.8 GB collection is unnecessary for an audio experiment.
- CC BY-NC-SA 4.0; commercial licensing is separate.

Decision: download only `Audio_Speech_Actors_01-24.zip`. Drop `calm` or map it
to neutral only as an explicitly documented experiment. Split by actor.

### CREMA-D - recommended diverse audio-visual supplement

Official source: https://github.com/CheyneyComputerScience/CREMA-D

- 7,442 clips from 91 actors aged 20-74.
- Six classes: anger, disgust, fear, happy, neutral, sad.
- Audio-only, video-only, and audiovisual perceptual ratings are available.
- More actor diversity than RAVDESS, but no surprise class.
- Full Git LFS clone is about 7.55 GB.

Decision: useful for robustness and six-class training; it cannot alone train
the project's `surprise` output.

### IEMOCAP - best classic aligned research corpus

Official source: https://sail.usc.edu/iemocap/

- About 12 hours of acted, dyadic audiovisual data.
- Ten actors, speech, video, face/head/hand motion capture, transcripts, and
  categorical plus valence/activation/dominance annotations.
- Requires an electronic release form and license agreement.
- Most public pretrained speech models use a four-class subset: angry, happy,
  neutral, sad.

Decision: scientifically strong, but not the fastest route to seven classes.

### MSP-Podcast / MSP-Conversation - best naturalistic speech direction

Official sources:

- https://lab-msp.com/MSP/MSP-Podcast.html
- https://lab-msp.com/MSP/MSP-Conversation.html

MSP-Podcast is a large naturalistic emotional-speech corpus assembled from
podcasts. MSP-Conversation release 2.0 contains more than 77 hours of long-form
conversations with time-continuous annotations.

Decision: better for later real-world SER evaluation than small acted sets, but
access, scale, and label protocol make it a phase-2 resource.

### UTeMo - recent small audiovisual exact-label option

Official source: https://data.mendeley.com/datasets/x5rmd28h73/1

- Published 2024-02-20.
- 1,801 high-resolution Spanish audiovisual clips, about 105 minutes.
- Balanced across sadness, surprise, joy, anger, fear, disgust, and neutral.
- CC BY-NC 3.0.

Decision: a useful recent cross-language audio-visual test, but it has no text
transcript modality and should not be the only English microphone training set.

## Text Datasets

### BRIGHTER / SemEval-2025 Task 11 - recommended recent multilingual source

Official sources:

- https://brighter-dataset.github.io/
- https://github.com/emotion-analysis-project/BRIGHTER
- https://github.com/emotion-analysis-project/semeval2025-task11

- Published for ACL/SemEval 2025 with refreshed label releases in 2026.
- About 139,583 examples across 28 language dataset configurations.
- Includes English, Hindi, Marathi, and many other lower-resource languages.
- Human-annotated multi-label presence and intensity for anger, disgust, fear,
  joy, sadness, and surprise; no selected emotion represents neutral/none.

Decision: this is the best recent text dataset found, especially if the final
system should support Indian languages. Its multi-label format must be adapted
for a single-label seven-class demo, so it is a phase-2 training/evaluation
resource rather than a reason to delay the pretrained English text branch.

### GoEmotions - recommended fine-grained text source

Official sources:

- https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/
- https://github.com/google-research/google-research/tree/master/goemotions

- 58K English Reddit comments.
- 27 emotion categories plus neutral.
- Multi-label and much richer than a seven-class taxonomy.

Decision: best for text-emotion experimentation. Use an explicit, documented
mapping into seven classes, or use the already trained seven-class DistilRoBERTa
model rather than retraining immediately.

### DailyDialog - recommended small exact-taxonomy text dataset

Paper: https://aclanthology.org/I17-1099/

- 13,118 human-written multi-turn daily-life dialogues.
- Seven labels: neutral, happiness, surprise, sadness, anger, disgust, fear.

Decision: its labels match the project exactly after `happiness -> joy`. It is
better for dialogue-like typed text than Reddit comments, though neutral is
dominant.

### MELD text annotations

Official source: https://github.com/declare-lab/MELD

MELD's utterance text uses the same seven labels as this project. It is useful
both as text training data and, more importantly, as aligned multimodal data.

Decision: do not train the text branch first; the recommended pretrained text
model was already trained on a balanced mixture that includes MELD,
GoEmotions, and other sources.

## Aligned Multimodal Datasets

### MELD - recommended aligned dataset for this project

Official source: https://github.com/declare-lab/MELD

- More than 1,400 conversations and roughly 13,700 utterances from the TV show
  *Friends*.
- Every utterance has audio, visual, and text modalities.
- Exactly seven labels: anger, disgust, sadness, joy, neutral, surprise, fear.
- Official train/dev/test partitions are provided.
- Strong class imbalance: neutral is much more common than fear or disgust.

Decision: this is the best label-aligned dataset for evaluating all three
branches and training a later lightweight fusion layer. It is not ideal as the
only source because it is scripted TV dialogue and contains scene/context
effects.

### CMU-MOSEI

Official sources:

- https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI

- More than 65 hours from over 1,000 speakers and 250 topics.
- Video, audio, and language from online monologues.
- Sentiment plus six emotion intensity labels: happy, sad, anger, surprise,
  disgust, fear.
- Multiple emotions can be present at once; neutral is not a normal seventh
  categorical output.

Decision: strong for advanced multi-label/intensity research, but it does not
drop directly into the project's single-label seven-class output.

### MER Challenge datasets and MERTools

Official toolkit: https://github.com/zeroQiaoba/MERTools

The official toolkit includes MER2023, MER2024, MER2025, and MER2026 resources.
The newer challenges cover semi-supervised, noise-robust, open-vocabulary, LLM,
and generative emotion understanding tasks. Downloads require accepting an EULA
on Hugging Face and are restricted to academic research.

Decision: these are the latest research resources, but they are intentionally
more complex than the requested quick build. Use MERTools as a reference and
future benchmark, not as the version-1 application base.

## Recommended Pretrained Models

### Visual: EmotiEffLib AffectNet-7

Official sources:

- https://github.com/sb-ai-lab/EmotiEffLib
- https://sb-ai-lab.github.io/EmotiEffLib/

Recommended model: `enet_b0_7`.

- Apache 2.0 library.
- Lightweight PyTorch/ONNX real-time facial-expression inference.
- Seven exact outputs: anger, disgust, fear, happiness, neutral, sadness,
  surprise.
- Repository reports about 65.74% AffectNet-7 validation accuracy for
  `enet_b0_7` and a roughly 16 MB model size.

Decision: compare this against the current CNN. It is the strongest low-effort
visual upgrade found for this project.

### Audio: emotion2vec+ base

Official sources:

- https://github.com/ddlBoJack/emotion2vec
- https://huggingface.co/emotion2vec/emotion2vec_plus_base

- ACL 2024 emotion representation work.
- The base classifier is roughly 90M parameters and was iteratively fine-tuned
  on thousands of hours of pseudo-labeled speech.
- Nine outputs: angry, disgusted, fearful, happy, neutral, other, sad,
  surprised, unknown.
- Seven outputs map directly to this project; `other` and `unknown` permit an
  honest abstention.

Decision: best first acoustic model. Use the base model for laptop latency; try
the large model only after measuring hardware performance. Review the model's
specific license before any non-academic deployment.

Fallback official benchmark model:
https://huggingface.co/superb/wav2vec2-base-superb-er

The SUPERB model is easy to call through Transformers but only supports the
common four-class IEMOCAP protocol, so it does not satisfy the seven-class goal.

### Text: English emotion DistilRoBERTa

Official model card:
https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

- Seven exact outputs: anger, disgust, fear, joy, neutral, sadness, surprise.
- Fine-tuned on a balanced subset of six sources, including GoEmotions and
  MELD.
- Model card reports 66% evaluation accuracy on its balanced seven-class set.
- Directly usable with a Hugging Face text-classification pipeline.

Decision: use it without initial retraining.

### Speech-to-text: Faster-Whisper

Official repository: https://github.com/SYSTRAN/faster-whisper

Faster-Whisper is a CTranslate2 implementation of Whisper and supports CPU int8
inference. It converts spoken words to text; it is not the vocal-emotion model.

Decision: use `tiny.en` first for latency, then compare `base.en` for transcript
quality.

## Reusable Research Code

- EmotiEffLib: production-shaped visual expression component.
- emotion2vec: modern speech emotion component and training references.
- Faster-Whisper: local speech-to-text component.
- Hugging Face Transformers: text pipeline and model loading.
- MERTools: latest official multimodal emotion challenge toolkit.
- CMU MultimodalSDK: aligned multimodal dataset access and feature sequences.
- MMSA: https://github.com/thuiar/MMSA - useful fusion-model reference for
  MOSI/MOSEI/CH-SIMS, but primarily sentiment analysis rather than this exact
  real-time seven-class task.
- Gradio streaming inputs: https://gradio.app/guides/streaming-inputs

No single GitHub repository found should be copied wholesale into this project.
The reliable route is to integrate maintained specialist components behind a
small common probability schema.

## Final Dataset Selection

For the immediate build:

```text
Visual baseline: current FER2013 model
Visual comparison/fine-tuning: MUTFER2024 + FERPlus
Audio model validation: RAVDESS speech-only + selected CREMA-D audio
Text model validation: DailyDialog + selected GoEmotions examples
Multimodal validation: MELD official dev/test splits
```

For the report, compare:

```text
visual only vs audio only vs text only vs fused
```

This comparison is more important than claiming that the system recognizes the
largest possible number of emotion names.
