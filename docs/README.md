# Multimodal Emotion Recognition Documentation

Research and audit date: 2026-08-23

This directory turns the existing facial-expression demo into a concrete plan
for the proposed project:

> Design and develop a real-time multimodal emotion recognition system
> integrating visual, audio, and textual inputs using deep learning.

Read the documents in this order:

1. [What we did](WHAT_WE_DID.md) - facial-only changes completed in this pass,
   dataset download status, and exact commands to run.
2. [Current state audit](CURRENT_STATE_AUDIT.md) - what is implemented, what
   works, what is missing, and a realistic completion estimate.
3. [Implementation blueprint](IMPLEMENTATION_BLUEPRINT.md) - the fastest
   architecture that produces a working system without training a large
   end-to-end model.
4. [Datasets, models, and sources](DATASETS_MODELS_SOURCES.md) - source-checked
   visual, speech, text, and aligned multimodal resources.
5. [Existing visual dataset research](../DATASET_RECOMMENDATIONS.md) - the
   earlier, longer comparison of facial-expression datasets.

## Decision Summary

- Use seven shared fusion classes: `anger`, `disgust`, `fear`, `joy`,
  `neutral`, `sadness`, and `surprise`.
- For the facial-only milestone, the code can now train 7 or 8 classes from the
  folders present in `dataset/`.
- Do not add `contempt` to only the visual branch. It would create an output
  that the audio and text branches cannot consistently support.
- Keep the current CNN as the first visual baseline, then replace it with or
  compare it against EmotiEffLib's AffectNet-7 model.
- Use `emotion2vec+ base` for vocal emotion and DistilRoBERTa for text emotion.
- Use Faster-Whisper only to turn microphone speech into text; vocal emotion
  must still come from the acoustic model.
- Use availability-aware late fusion and temporal smoothing for version 1.
- Use MELD later for aligned multimodal evaluation or learned fusion.
