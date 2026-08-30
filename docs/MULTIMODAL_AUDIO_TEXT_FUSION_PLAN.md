# Multimodal Audio, Text, and Fusion Plan

Date: 2026-08-28

Project goal:

```text
Design and develop a real-time multimodal emotion recognition system integrating
visual, audio, and textual inputs.
```

## Current Status

Visual is now strong enough to promote as the facial module:

```text
Model:    models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
Accuracy: 85.22%
Macro F1: 84.86%
TTA accuracy for report: 86.65%
TTA macro F1 for report: 86.30%
```

Final visual deliverables are in:

```text
FINAL/VISUAL
```

## Shared Emotion Labels

Keep the project at seven shared labels:

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

Text label mapping:

```text
anger -> angry
joy -> happy
sadness -> sad
```

Audio label mapping:

```text
fearful -> fear
surprised -> surprise
calm -> neutral
```

## Fast Build Plan

Do not train audio/text from scratch first. Use pretrained models and fuse their
probability vectors.

Recommended first version:

```text
Visual: EfficientNet-B1 checkpoint from this project
Audio:  Hugging Face Dpngtm/wav2vec2-emotion-recognition
Text:   Hugging Face j-hartmann/emotion-english-distilroberta-base
Fusion: weighted late fusion
```

Fusion rule:

```text
final_probs = 0.50 visual + 0.30 audio + 0.20 text
```

If one modality is missing, renormalize the available weights.

Examples:

```text
visual + text only: 0.70 visual + 0.30 text
visual + audio only: 0.60 visual + 0.40 audio
text only: 1.00 text
audio only: 1.00 audio
```

## What Needs Downloading

No manual dataset download is required for the first working multimodal demo.

Expected automatic model downloads after approval:

```text
Text model:  j-hartmann/emotion-english-distilroberta-base
Audio model: Dpngtm/wav2vec2-emotion-recognition
```

These are pretrained Hugging Face models. They are much smaller than the 10 GB
visual dataset and can be cached locally.

Manual dataset downloads only become useful after the demo works:

```text
RAVDESS audio-only speech: about 208 MB
CREMA-D full repository: about 7.55 GB with git-lfs
MELD raw multimodal dataset: about 10.9 GB on Hugging Face
GoEmotions text dataset: can be loaded through Hugging Face datasets
```

## Dataset Choices

Best for complete multimodal report:

```text
MELD
```

Reason: it has text, audio, and visual modalities with seven labels: anger,
disgust, sadness, joy, neutral, surprise, and fear.

Best small audio dataset:

```text
RAVDESS audio speech
```

Reason: around 208 MB for audio-only speech, includes neutral, calm, happy, sad,
angry, fearful, disgust, and surprised.

Good but larger audio/audiovisual dataset:

```text
CREMA-D
```

Reason: 7,442 clips from 91 actors. It has angry, disgust, fear, happy, neutral,
and sad, but no surprise class.

Best text dataset:

```text
GoEmotions
```

Reason: 58k curated Reddit comments with 27 emotion categories plus neutral.

## Implementation Steps

1. Add a shared emotion schema and label-mapping utility.
2. Add text emotion inference using the DistilRoBERTa model.
3. Add audio emotion inference using the Wav2Vec2 model.
4. Add a multimodal fusion module.
5. Add a command-line demo that accepts image/webcam, audio file, and text.
6. Add a real-time demo mode with webcam + microphone chunks + optional typed text.
7. Create `FINAL/AUDIO`, `FINAL/TEXT`, and `FINAL/FUSION` report folders after
   evaluation.

## Implementation Progress

Date: 2026-08-28

Added the first multimodal code layer without downloading large models or
datasets:

```text
emotion_schema.py
text_emotion.py
audio_emotion.py
fusion_emotion.py
multimodal_emotion_demo.py
prepare_ravdess_audio.py
evaluate_audio_emotion_model.py
requirements-multimodal.txt
```

Current runnable backends:

```text
visual: final EfficientNet-B1 PyTorch checkpoint
text:   lexicon fallback backend
audio:  acoustic heuristic fallback backend for WAV files
fusion: weighted late fusion
```

Important research note:

```text
The current text/audio fallback backends are for system integration and smoke
testing only. They prove the multimodal pipeline works. For final research
results, replace them with pretrained Hugging Face text/audio models or evaluate
on downloaded datasets.
```

Smoke tests created normal report outputs in:

```text
reports/multimodal_smoke_latest
```

No files were added to `FINAL/AUDIO`, `FINAL/TEXT`, or `FINAL/FUSION` yet. Those
folders should be created only after approval and final evaluation.

Current smoke-test commands:

```powershell
venv_gpu\Scripts\python.exe multimodal_emotion_demo.py --image dataset_mutfer2024\test\happy\Happy_1724392681279.jpg --text "I am really happy and excited today!" --visual-device cuda --face-detector yunet --output reports\multimodal_smoke_latest\visual_text_result.json
```

```powershell
venv_gpu\Scripts\python.exe multimodal_emotion_demo.py --audio-file reports\multimodal_smoke_latest\sample_excited.wav --text "I am shocked and surprised!" --audio-backend acoustic --text-backend lexicon --output reports\multimodal_smoke_latest\audio_text_result.json
```

Next recommended step after your dataset downloads complete:

```text
1. Prepare RAVDESS if downloaded.
2. Install requirements-multimodal.txt only after approval.
3. Run pretrained text/audio model checks.
4. Evaluate audio and text separately.
5. Evaluate visual + audio + text fusion.
```

RAVDESS preparation command after the zip finishes downloading:

```powershell
venv_gpu\Scripts\python.exe prepare_ravdess_audio.py --zip data_sources\RAVDESS\Audio_Speech_Actors_01-24.zip --output dataset_ravdess_audio --overwrite
```

Audio fallback evaluation command:

```powershell
venv_gpu\Scripts\python.exe evaluate_audio_emotion_model.py --dataset dataset_ravdess_audio\test --backend acoustic --reports reports\audio_ravdess_acoustic_latest
```

Pretrained audio evaluation command after optional dependencies/model are ready:

```powershell
venv_gpu\Scripts\python.exe evaluate_audio_emotion_model.py --dataset dataset_ravdess_audio\test --backend hf --reports reports\audio_ravdess_hf_latest
```

## Sources

```text
MELD official: https://affective-meld.github.io/
MELD GitHub: https://github.com/declare-lab/MELD
RAVDESS Zenodo: https://zenodo.org/records/1188976
CREMA-D GitHub: https://github.com/CheyneyComputerScience/CREMA-D
GoEmotions HF: https://huggingface.co/datasets/google-research-datasets/go_emotions
Text model: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
Audio model: https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition
```
