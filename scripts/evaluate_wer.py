#!/usr/bin/env python3
"""
Automated Word Error Rate (WER) Benchmarking for Urdu Speech Recognition

This script benchmarks Whisper model transcription accuracy on Urdu datasets
to provide baseline metrics for Zuban's transcription quality.

Usage:
    python scripts/evaluate_wer.py --help
    python scripts/evaluate_wer.py --limit 10
    python scripts/evaluate_wer.py --model small
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset, load_metric
from jiwer import process_words, wer
from tqdm import tqdm

try:
    import whisper
except ImportError:
    print("Error: openai-whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not installed. Audio will be loaded via librosa.")
    import librosa


def preprocess_urdu_text(text: str) -> str:
    """Preprocess Urdu text for WER calculation."""
    if not text:
        return ""
    
    text = str(text).strip()
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^\w\s\u0600-\u06FF۔،؟ؤ ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيٱٲٳٴٵٶٷٸٹٺٻټٽپٿٹڈډڑژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ەۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬ۮۯ۰۱۲۳۴۵۶۷۸۹۝۔۔۔]',
                  '', text)
    
    return text.strip()


def load_audio(path: str) -> tuple:
    """Load audio file and return waveform and sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
        return audio, sr
    except ImportError:
        import librosa
        audio, sr = librosa.load(path, sr=16000)
        return audio, sr


def transcribe_audio(model, audio_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> str:
    """Transcribe audio file using Whisper."""
    try:
        result = model.transcribe(audio_path, language="ur", fp16=(device == "cuda"))
        return result["text"].strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


def load_whisper_model(model_size: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load Whisper model."""
    print(f"Loading Whisper {model_size} model on {device}...")
    model = whisper.load_model(model_size, device=device)
    return model


def load_urdu_dataset(limit: int = None):
    """Load Urdu dataset from Hugging Face."""
    print("Loading Common Voice Urdu dataset...")
    
    try:
        dataset = load_dataset(
            "azeem-ahmed/Common_Voice_Corpus_22_0_Urdu",
            split="test",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load primary dataset: {e}")
        print("Trying alternative dataset...")
        try:
            dataset = load_dataset(
                "khawajaaliarshad/common-voice-urdu-processed",
                split="test",
                trust_remote_code=True
            )
        except Exception as e2:
            print(f"Failed to load alternative dataset: {e2}")
            sys.exit(1)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    return dataset


def calculate_wer(references: list, hypotheses: list) -> dict:
    """Calculate WER and detailed metrics."""
    ref_processed = [preprocess_urdu_text(r) for r in references]
    hyp_processed = [preprocess_urdu_text(h) for h in hypotheses]
    
    ref_filtered = [r for r in ref_processed if r]
    hyp_filtered = []
    idx_map = []
    for i, r in enumerate(ref_processed):
        if r:
            hyp_filtered.append(hyp_processed[i])
            idx_map.append(i)
    
    if not ref_filtered:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0}
    
    try:
        output = process_words(ref_filtered, hyp_filtered)
        
        return {
            "wer": output.wer * 100,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "total_words": output.references
        }
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0}


def run_benchmark(model_size: str = "base", limit: int = None, device: str = None):
    """Run WER benchmark on Urdu dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"WER Benchmark for Urdu Speech Recognition")
    print(f"{'='*60}")
    print(f"Model: whisper-{model_size}")
    print(f"Device: {device}")
    print(f"Samples: {limit if limit else 'all'}")
    print(f"{'='*60}\n")
    
    model = load_whisper_model(model_size, device)
    
    dataset = load_urdu_dataset(limit)
    
    references = []
    hypotheses = []
    
    print("\nTranscribing audio samples...")
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            if "audio" in item:
                audio_data = item["audio"]
                if isinstance(audio_data, dict) and "array" in audio_data:
                    import tempfile
                    import numpy as np
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_data["array"], audio_data["sample_rate"])
                        audio_path = tmp.name
                else:
                    audio_path = audio_data
            elif "file" in item:
                audio_path = item["file"]
            else:
                print(f"Sample {idx}: No audio file found, skipping")
                continue
            
            reference = item.get("sentence", item.get("text", ""))
            if not reference:
                continue
            
            hypothesis = transcribe_audio(model, audio_path, device)
            
            references.append(reference)
            hypotheses.append(hypothesis)
            
            if isinstance(audio_path, str) and audio_path.endswith(".wav") and "tempfile" in locals():
                try:
                    os.unlink(audio_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"\nProcessed {len(references)} samples")
    
    if not references:
        print("No samples processed successfully!")
        return
    
    metrics = calculate_wer(references, hypotheses)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total Samples: {len(references)}")
    print(f"Total Reference Words: {metrics['total_words']}")
    print(f"Substitutions: {metrics['substitutions']}")
    print(f"Deletions: {metrics['deletions']}")
    print(f"Insertions: {metrics['insertions']}")
    print(f"\n>>> Word Error Rate (WER): {metrics['wer']:.2f}% <<<")
    print(f"{'='*60}\n")
    
    print("Sample Transcriptions:")
    print("-" * 60)
    for i in range(min(3, len(references))):
        print(f"\nSample {i+1}:")
        print(f"Reference: {references[i][:100]}...")
        print(f"Hypothesis: {hypotheses[i][:100]}...")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Automated WER Benchmarking for Urdu Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_wer.py --help
  python scripts/evaluate_wer.py --limit 5
  python scripts/evaluate_wer.py --model small --limit 10
  python scripts/evaluate_wer.py --model medium --limit 50
  python scripts/evaluate_wer.py --model base --device cpu

Available models: tiny, base, small, medium, large, large-v2, large-v3
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        model_size=args.model,
        limit=args.limit,
        device=args.device
    )


if __name__ == "__main__":
    main()
