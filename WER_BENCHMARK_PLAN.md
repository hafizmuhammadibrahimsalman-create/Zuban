# Automated WER Benchmarking for Urdu Datasets

## Updated Implementation Plan

Based on comprehensive research of the zuban codebase and Urdu ASR benchmarking best practices.

---

## 1. Overview

This plan outlines creating an automated Python script to benchmark Whisper model's transcription accuracy (Word Error Rate - WER) on Urdu datasets. This provides a baseline metric for Zuban's transcription quality.

---

## 2. Research Findings

### 2.1 Best Urdu Speech Datasets

| Dataset | Description | Recommended For |
|---------|-------------|-----------------|
| `azeem-ahmed/Common_Voice_Corpus_22_0_Urdu` | Common Voice 22.0 Urdu subset | Primary benchmark |
| `khawajaaliarshad/common-voice-urdu-processed` | Processed Common Voice Urdu | Training/evaluation |
| `muhammadsaadgondal/urdu-tts` | Urdu TTS with transcriptions | Additional testing |

### 2.2 Benchmark Baselines (COLING 2025)

| Model | WER (%) | Dataset |
|-------|---------|---------|
| Whisper Large v3 (fine-tuned) | 2.29 | Common Voice 17.0 |
| Whisper Small (no fine-tuning) | 33.68 | Curated |
| Whisper Base (no fine-tuning) | 53.67 | Curated |
| Whisper Tiny (no fine-tuning) | 67.08 | Curated |

### 2.3 Recommended Whisper Models

- **Production**: `whisper-large-v3` or fine-tuned `kingabzpro/whisper-large-v3-urdu`
- **Balanced**: `whisper-medium` or `whisper-small`
- **Quick testing**: `whisper-base`

---

## 3. Implementation

### 3.1 Dependencies

```
openai-whisper
datasets
jiwer
urduhack
soundfile
librosa
torch
```

### 3.2 Key Considerations for Urdu WER

1. **Text Preprocessing**: Use `urduhack` for normalization (Arabic to Urdu character conversion, diacritics removal)
2. **Unicode Range**: Handle `\u0600-\u06FF` (Arabic script)
3. **Whitespace**: Normalize all whitespace to single spaces
4. **Punctuation**: Handle Urdu-specific punctuation (۔،؟)

### 3.3 Script Features

- Auto-download Urdu dataset from Hugging Face
- Load Whisper model (configurable: tiny/base/small/medium/large)
- Transcribe audio samples
- Compare with ground truth transcriptions
- Calculate overall WER with detailed metrics
- Support `--limit` for small test runs
- Support `--model` to specify Whisper model size

---

## 4. Files to Create

```
scripts/
├── evaluate_wer.py          # Main evaluation script
└── requirements_bench.txt  # Dependencies
```

---

## 5. Verification Plan

### 5.1 Automated Tests
```bash
# Help
python scripts/evaluate_wer.py --help

# Quick test (5 samples)
python scripts/evaluate_wer.py --limit 5

# Full benchmark
python scripts/evaluate_wer.py
```

### 5.2 Manual Verification
- Review transcripts for reasonableness
- Confirm WER percentage is within expected range (30-50% for base models)
- Verify selected dataset provides high-quality ground truth

---

## 6. References

- WER We Stand Paper: https://arxiv.org/abs/2409.11252
- jiwer library: https://pypi.org/project/jiwer/
- urduhack: https://urduhack.readthedocs.io/
- Fine-tuned Urdu Whisper: https://huggingface.co/kingabzpro/whisper-large-v3-urdu
