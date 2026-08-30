# Fast Multimodal Implementation Blueprint

Design date: 2026-08-23

## Goal

Deliver a local, working, near-real-time system that accepts:

1. visual expression from a webcam;
2. vocal emotion from microphone acoustics;
3. textual emotion from typed text or a microphone transcript;
4. a fused emotion with per-modality probabilities and confidence.

The first version should use pretrained unimodal models and late fusion. It
should not attempt to train one end-to-end network from unrelated visual,
speech, and text datasets.

## Recommended Version-1 Stack

| Part | Recommended component | Reason |
|---|---|---|
| Interface | Gradio Blocks | Webcam, microphone, text, and streaming are supported in Python |
| Visual baseline | Existing `emotion_model.h5` | Immediate reuse and tiny inference cost |
| Visual upgrade | EmotiEffLib `enet_b0_7` | AffectNet-7, lightweight, real-time, stronger baseline |
| Vocal emotion | `emotion2vec_plus_base` | Modern nine-output SER model; seven labels map directly |
| Speech-to-text | Faster-Whisper `tiny.en` or `base.en`, CPU int8 | Local transcription with manageable latency |
| Text emotion | `j-hartmann/emotion-english-distilroberta-base` | Exactly the desired seven semantic emotion labels |
| Fusion | Weighted late fusion | Works without an aligned training set and handles missing inputs |
| Runtime | Python 3.10, mostly local inference | Fits the current project and avoids an external API dependency |

Use `emotion2vec+ base`, not `large`, for the first laptop build. The base model
is roughly 90M parameters; the large model is roughly 300M and is a better GPU
experiment than a quick CPU prototype.

## Architecture

```text
Webcam frames ----> face detector ----> visual model ----> 7-class adapter --+
                                                                            |
Microphone -------> 3 s rolling window -> acoustic SER ---> 7-class adapter -+--> fusion
       |                                                                    |      |
       +----------> Faster-Whisper ------> text model ----> 7-class adapter -+      +--> EMA
                                                                            |             |
Typed text -----------------------------> text model ----> 7-class adapter --+             +--> UI
```

Each branch must return the same structure:

```python
{
    "modality": "visual",
    "timestamp": 0.0,
    "available": True,
    "probabilities": {
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "neutral": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
    },
}
```

## Canonical Label Adapters

### Current visual model

```text
angry -> anger
disgust -> disgust
fear -> fear
happy -> joy
neutral -> neutral
sad -> sadness
surprise -> surprise
```

### EmotiEffLib AffectNet-7

```text
Anger -> anger
Disgust -> disgust
Fear -> fear
Happiness -> joy
Neutral -> neutral
Sadness -> sadness
Surprise -> surprise
```

### emotion2vec+

```text
angry -> anger
disgusted -> disgust
fearful -> fear
happy -> joy
neutral -> neutral
sad -> sadness
surprised -> surprise
other -> abstain
unknown -> abstain
```

When `other` or `unknown` wins, mark audio unavailable for fusion or reduce its
weight sharply. Do not renormalize a confident `unknown` prediction into a
false known emotion.

### Text DistilRoBERTa

Its labels already match except that the project should display `joy` rather
than `happy`. No semantic collapsing is needed.

## Timing and Real-Time Behavior

The modalities operate at different natural rates:

| Branch | Suggested update | Window |
|---|---:|---:|
| Visual | every 200-500 ms | average recent 5-10 valid face predictions |
| Audio emotion | every 1.5 s | most recent 3 s of voiced audio |
| Speech-to-text | every 2-4 s | finalized voiced segment |
| Typed text | on submit/change | latest submitted text |
| Fusion | every 500 ms | only non-stale branch results |

Expire visual results after about 2 seconds without a face. Expire audio and
transcribed-text results after about 6 seconds without new speech. A typed text
result can remain until the user clears or replaces it.

Skip acoustic inference for silence using a simple RMS threshold first. A
voice-activity detector can be added later if background noise causes problems.

## Fusion

For available modality probability vectors `p_m` and reliability weights
`r_m`, compute:

```text
p_fused = sum(available_m * r_m * p_m) / sum(available_m * r_m)
```

Start with equal weights so the behavior is explainable:

```text
visual = 1.0
audio  = 1.0
text   = 1.0
```

Then tune the weights on MELD validation data or a small locally recorded test
set. Do not use raw maximum softmax confidence as the only reliability weight;
neural classifiers can be confidently wrong.

Smooth the fused vector with an exponential moving average:

```text
smoothed_t = 0.65 * smoothed_(t-1) + 0.35 * fused_t
```

Display `uncertain` when the fused maximum is below an initial threshold such
as 0.40, or when two top classes are very close. This threshold must later be
calibrated, not presented as a scientific constant.

## Why Late Fusion Is the Correct Quick Choice

- Existing datasets are mostly unimodal and use different people, situations,
  labels, and sampling methods.
- Concatenating features from unrelated examples is invalid.
- A trainable early-fusion network needs aligned audio-video-text samples.
- Late fusion allows each pretrained specialist to work immediately.
- Missing face, silence, and empty text can be handled naturally.
- Every branch remains independently testable and replaceable.

After version 1 works, MELD can support a small learned fusion layer over the
three seven-class probability vectors. That is a controlled upgrade, not a
prerequisite for the demonstration.

## Proposed Project Structure

```text
facial_recognition/
  app.py
  config.py
  requirements.txt
  emotion_model.h5
  models/
    labels.json
  src/
    schemas.py
    visual.py
    audio.py
    transcription.py
    text.py
    fusion.py
    state.py
  training/
    train_visual.py
    evaluate_visual.py
    prepare_visual_dataset.py
  tests/
    test_label_mapping.py
    test_fusion.py
    test_model_smoke.py
  docs/
```

`schemas.py` should own the canonical labels and validate that every probability
vector has seven finite non-negative values summing approximately to one.

## Build Order

### Phase 1: stabilize the existing visual branch

- Extract the current model into `src/visual.py`.
- Return all probabilities, not only argmax.
- Save/load canonical label metadata.
- Add face-loss handling, smoothing, and confidence display.
- Correct the README commands.

Acceptance check: webcam prediction runs for five minutes without crashing and
labels do not flicker on every frame.

### Phase 2: add typed text

- Load the DistilRoBERTa text pipeline once at startup.
- Add a text box and submit/clear controls.
- Map its seven outputs into the shared schema.

Acceptance check: each branch can be used alone, and empty text is treated as
unavailable.

### Phase 3: add microphone acoustic emotion

- Capture 16 kHz mono audio into a thread-safe rolling buffer.
- Reject silence.
- Run `emotion2vec_plus_base` on a three-second window.
- Map seven labels and abstain for `other`/`unknown`.

Acceptance check: the UI updates audio emotion without blocking webcam updates.

### Phase 4: add transcription

- Use Faster-Whisper locally on finalized voice segments.
- Feed transcript text to the same text classifier.
- Let explicitly typed text take precedence while it is non-empty.

Acceptance check: the screen distinguishes `voice tone` from `spoken words`.

### Phase 5: fuse and evaluate

- Add availability-aware probability fusion.
- Add timestamps, expiry, and EMA smoothing.
- Log predictions and latency without saving raw camera/microphone data by
  default.
- Evaluate each branch and the fused output separately.

Acceptance check: disabling any one or two modalities still produces a valid
result from the remaining input.

### Phase 6: improve the visual model

- Compare the current CNN with EmotiEffLib `enet_b0_7`.
- Prepare MUTFER2024 and FERPlus using official splits where available.
- Use class weights and macro F1.
- Retain BTFER or another independent set for final testing only.

## Minimum Evaluation Report

Report all of the following:

- dataset and exact split;
- class counts;
- accuracy, macro F1, and weighted F1;
- per-class precision and recall;
- confusion matrix;
- average latency for every branch;
- fused performance versus each unimodal branch;
- behavior when one or more modalities are missing;
- a small real-webcam/microphone test with participants not in training data.

For speaker and actor datasets, split by person, not randomly by clip. Otherwise
the same actor can appear in training and test data and inflate performance.

## Scope Guardrails

The application estimates **displayed or expressed affect**, not a person's
private internal emotional state, mental health diagnosis, honesty, intent, or
risk. Results should be framed as uncertain model predictions. Camera and
microphone processing should stay local by default, and raw recordings should
not be stored without explicit consent.

