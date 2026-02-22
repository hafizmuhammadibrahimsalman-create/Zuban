# Automated WER Benchmarking for Urdu Datasets

## Updated Implementation Plan v2

Based on comprehensive research of the zuban codebase and Urdu ASR benchmarking best practices.

---

## 1. Overview

This plan outlines creating an automated Python script to benchmark Whisper model's transcription accuracy (Word Error Rate - WER) on Urdu datasets. This provides a baseline metric for Zuban's transcription quality.

---

## 2. Research Findings

### 2.1 Best Urdu Speech Datasets

| Dataset | Size | URL | Notes |
|---------|------|-----|-------|
| **UrduMegaSpeech-1M** | 1M+ samples | humair025/UrduMegaSpeech | Large-scale with quality metrics |
| **Urdu-ONYX-WAV** | 100K-1M | humair025/Urdu-ONYX-WAV | High-quality TTS/ASR |
| **Common Voice 17_0** | Standard | mozilla-foundation/common_voice_17_0 | Standard benchmark |
| **UrduSpeech** | 10K-100K | humairawan/UrduSpeech | CC-BY-4.0 license |

### 2.2 Fine-tuned Whisper Models for Urdu

| Model | WER | URL |
|-------|-----|-----|
| **kingabzpro/whisper-large-v3-urdu** | 21.47% | huggingface.co/kingabzpro/whisper-large-v3-urdu |
| **kingabzpro/whisper-large-v3-turbo-urdu** | 25.78% | huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu |
| **HowMannyMore/whisper-small-urdu** | 33.31% | huggingface.co/HowMannyMore/whisper-small-urdu |

### 2.3 Benchmark Baselines (COLING 2025)

| Model | WER (%) | Notes |
|-------|---------|-------|
| Whisper Large v3 (fine-tuned) | 21-26% | Best results |
| Whisper Small (no fine-tuning) | ~33% | Good baseline |
| Whisper Base (no fine-tuning) | ~53% | Lightweight |
| Whisper Tiny (no fine-tuning) | ~67% | Quick testing |

### 2.4 Implementation Recommendations

- **Use faster-whisper**: 4x faster, ~50% less memory, same accuracy
- **Preprocessing**: Use urduhack for Urdu text normalization
- **WER + CER**: Report both Word and Character Error Rate

---

## 3. Implementation

### 3.1 Dependencies

```
openai-whisper
faster-whisper
datasets
jiwer
soundfile
librosa
torch
tqdm
urduhack
```

### 3.2 Key Features

1. Auto-download Urdu dataset from Hugging Face
2. Load Whisper model (configurable)
3. Transcribe audio samples
4. Compare with ground truth
5. Calculate WER with Urdu preprocessing
6. Support --limit for small test runs
7. Support --model to specify Whisper model size
8. JSON/CSV export for result tracking

---

## 4. Files

```
scripts/
├── evaluate_wer.py          # Main evaluation script
└── requirements_bench.txt  # Dependencies
```

---

## 5. Usage

```bash
# Help
python scripts/evaluate_wer.py --help

# Quick test (5 samples)
python scripts/evaluate_wer.py --limit 5

# Full benchmark
python scripts/evaluate_wer.py

# Specific model
python scripts/evaluate_wer.py --model small
```

---

## 6. Verification

### 5.1 Automated Tests
- [x] Script runs without syntax errors: `python scripts/evaluate_wer.py --help`
- [ ] Run on small subset: `python scripts/evaluate_wer.py --limit 5`
- [ ] Verify WER is within expected range

### 5.2 Manual Verification
- Review transcripts for reasonableness
- Confirm WER percentage is within expected range

---

## 7. References

- WER We Stand Paper: https://arxiv.org/html/2409.11252v1
- jiwer library: https://pypi.org/project/jiwer/
- urduhack: https://urduhack.readthedocs.io/
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Fine-tuned Urdu Whisper: https://huggingface.co/kingabzpro/whisper-large-v3-urdu
