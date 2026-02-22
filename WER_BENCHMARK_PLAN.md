# Automated WER Benchmarking for Urdu Datasets

## Updated Implementation Plan v3 - CPU Optimized

Based on comprehensive research for CPU-only environments (no GPU).

---

## 1. Overview

This plan outlines creating an automated Python script to benchmark Whisper model's transcription accuracy (Word Error Rate - WER) on Urdu datasets for CPU-only environments.

---

## 2. Research Findings (CPU-Optimized)

### 2.1 Best CPU-Optimized Whisper Implementation

**faster-whisper (CTranslate2)** is recommended for CPU:
- **4x faster** than OpenAI Whisper
- **~50% less memory** usage
- Supports **int8 quantization** for 2-3x additional speedup
- URL: https://github.com/SYSTRAN/faster-whisper

### 2.2 Best Urdu Whisper Models for CPU

| Model | WER | CPU Speed | Recommended |
|-------|-----|-----------|-------------|
| **whisper-large-v3-turbo** (fine-tuned) | 25.78% | ~120s/min | Best accuracy |
| **whisper-large-v3-turbo** (base) | ~26% | ~120s/min | Good balance |
| **whisper-small** | ~33% | ~36s/min | **Fastest** |
| **whisper-base** | ~53% | ~18s/min | Quick testing |

### 2.3 Fine-tuned Urdu Models

| Model | WER | URL |
|-------|-----|-----|
| **kingabzpro/whisper-large-v3-turbo-urdu** | 25.78% | huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu |
| **kingabzpro/whisper-large-v3-urdu** | 21.47% | huggingface.co/kingabzpro/whisper-large-v3-urdu |

### 2.4 Best Datasets for CPU

| Dataset | Size | Samples | URL |
|---------|------|---------|-----|
| **UrduSpeech-IndicVoices-ST-kProcessed** | 31.7 MB | 20K | humair025/UrduSpeech-IndicVoices-ST-kProcessed |
| **Common Voice Urdu 11** | ~2-3 GB | 10K | Talha185/Common-voice-urdu-11 |
| **common-voice-urdu-processed** | ~1.85 GB | 13.5K | UmarRamzan/common-voice-urdu-processed |

### 2.5 CPU Performance Benchmarks

| Model | Time per 1min audio | Memory (int8) |
|-------|---------------------|---------------|
| tiny | ~6s | ~500MB |
| base | ~18s | ~800MB |
| **small** | **~36s** | **~1GB** |
| medium | ~90s | ~2GB |
| large-v3-turbo | ~120s | ~3GB |

---

## 3. Implementation

### 3.1 Dependencies

```
faster-whisper
datasets
jiwer
soundfile
librosa
tqdm
```

### 3.2 Key Features (CPU-Optimized)

1. Use **faster-whisper** instead of openai-whisper
2. **int8 quantization** by default for CPU speed
3. Load smaller Urdu datasets for quick testing
4. Configurable model sizes (tiny/base/small)
5. WER calculation with Urdu preprocessing
6. --limit for testing with small samples
7. --model and --compute-type options

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

# Quick test (5 samples) - CPU optimized
python scripts/evaluate_wer.py --limit 5

# Using small model (fastest)
python scripts/evaluate_wer.py --model small

# Using int8 for faster CPU
python scripts/evaluate_wer.py --compute-type int8

# Full benchmark with turbo model
python scripts/evaluate_wer.py --model large-v3-turbo
```

---

## 6. Verification

- [x] Script runs: `python scripts/evaluate_wer.py --help`
- [ ] Run small test: `python scripts/evaluate_wer.py --limit 5`
- [ ] Verify WER within expected range (25-35%)

---

## 7. References

- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Fine-tuned Urdu: https://huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu
- Dataset: https://huggingface.co/datasets/humair025/UrduSpeech-IndicVoices-ST-kProcessed
- jiwer: https://pypi.org/project/jiwer/
