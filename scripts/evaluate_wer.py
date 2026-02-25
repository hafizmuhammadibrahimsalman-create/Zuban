#!/usr/bin/env python3
"""
Automated Word Error Rate (WER) Benchmarking for Urdu Speech Recognition
CPU-Optimized version using faster-whisper

Usage:
    python scripts/evaluate_wer.py --help
    python scripts/evaluate_wer.py --limit 10
    python scripts/evaluate_wer.py --model small
"""

import argparse
import os
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from jiwer import process_words
from tqdm import tqdm


def preprocess_urdu_text(text: str) -> str:
    """Preprocess Urdu text for WER calculation."""
    if not text:
        return ""
    
    text = str(text).strip()
    # Remove punctuations both English and Urdu
    text = re.sub(r'[۔،؟؛:\.,\?!\'\"\-\(\)\[\]\{\}]', '', text)
    # Remove any extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def load_model(model_size: str, compute_type: str = "int8"):
    """Load faster-whisper model (CPU-optimized)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)
    
    print(f"Loading faster-whisper {model_size} (compute_type={compute_type}) on CPU...")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    return model


def transcribe_audio(model, audio_path: str) -> str:
    """Transcribe audio file using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        segments, info = model.transcribe(audio_path, language="ur", beam_size=1)
        text = " ".join([seg.text.strip() for seg in segments])
        return text
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


def calculate_wer(references: list, hypotheses: list) -> dict:
    """Calculate WER and detailed metrics."""
    ref_processed = [preprocess_urdu_text(r) for r in references]
    hyp_processed = [preprocess_urdu_text(h) for h in hypotheses]
    
    ref_filtered = [r for r in ref_processed if r]
    hyp_filtered = [hyp_processed[i] for i, r in enumerate(ref_processed) if r]
    
    if not ref_filtered:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0}
    
    try:
        output = process_words(ref_filtered, hyp_filtered)
        # jiwer output.references gives tokenized list of lists.
        # We can calculate total ground-truth words manually or via sum of operation counts
        total_ref_words = sum(len(r.split()) for r in ref_filtered)
        
        return {
            "wer": output.wer * 100,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "total_words": total_ref_words
        }
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0}


def run_benchmark(model_size: str = "small", limit: int = None, compute_type: str = "int8"):
    """Run WER benchmark on Urdu dataset (CPU-optimized)."""
    print(f"\n{'='*60}")
    print(f"WER Benchmark for Urdu Speech Recognition (CPU)")
    print(f"{'='*60}")
    print(f"Model: whisper-{model_size}")
    print(f"Compute type: {compute_type}")
    print(f"Samples: {limit if limit else 'all'}")
    print(f"{'='*60}\n")
    
    model = load_model(model_size, compute_type)
    
    import subprocess
    import tempfile
    import os
    import shutil

    sample_urdu_texts = [
        "یہ ایک چھوٹا سا امتحان ہے۔",
        "اردو ہماری قومی زبان ہے۔",
        "میں روزانہ صبح سویرے اٹھتا ہوں۔",
        "کمپیوٹر سائنس کا مستقبل بہت روشن ہے۔",
        "آرٹیفیشل انٹیلیجنس نے دنیا کو بدل دیا ہے۔",
        "مجھے کتابیں پڑھنا بہت پسند ہے۔",
        "پاکستان کے شمال میں خوبصورت پہاڑ ہیں۔",
        "محنت کامیابی کی کنجی ہے۔",
        "آج موسم کیسا ہے؟",
        "ہمیں دوسروں کی مدد کرنی چاہیے۔",
        "وقت کی قدر کرنا سیکھیں۔",
        "سافٹ ویئر انجینئرنگ ایک اچھا پیشہ ہے۔",
    ]
    
    num_samples = min(limit if limit else 3, len(sample_urdu_texts))
    print(f"Generating and testing {num_samples} real spoken audios using edge-tts...")

    references = []
    hypotheses = []

    temp_dir = tempfile.mkdtemp()
    
    for i in range(num_samples):
        text = sample_urdu_texts[i]
        references.append(text)
        
        audio_path = os.path.join(temp_dir, f"sample_{i}.mp3")
        
        try:
            # Generate high-quality Urdu speech dynamically
            subprocess.run(["edge-tts", "--voice", "ur-PK-AsadNeural", "--text", text, "--write-media", audio_path], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Sample {i+1}: Failed to generate audio -> {e}")
            hypotheses.append("")
            continue

        try:
            hypothesis = transcribe_audio(model, audio_path)
            hypotheses.append(hypothesis)
            print(f"Sample {i+1}: Ground Truth : {text[:50]}")
            print(f"Sample {i+1}: Transcription: {hypothesis[:50]}")
        except Exception as e:
            print(f"Error on sample {i+1}: {e}")
            hypotheses.append("")
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\nProcessed {len(references)} samples")
    
    if not references:
        print("No samples processed!")
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
    
    print("For real benchmarking, consider using a larger dataset or different models.")
    print("-" * 60)
    for i in range(min(3, len(references))):
        print(f"\nSample {i+1}:")
        print(f"Reference: {references[i]}")
        print(f"Hypothesis: {hypotheses[i]}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="WER Benchmark for Urdu Speech Recognition (CPU-Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_wer.py --help
  python scripts/evaluate_wer.py --limit 5
  python scripts/evaluate_wer.py --model small --limit 10
  python scripts/evaluate_wer.py --model base --compute-type int8
  python scripts/evaluate_wer.py --model tiny --limit 5
  python scripts/evaluate_wer.py --model large-v3-turbo --limit 5

CPU-Optimized models (recommended for CPU):
  - tiny: Fastest, ~6s per minute audio
  - base: Quick, ~18s per minute audio
  - small: Best balance, ~36s per minute audio (recommended)
  - medium: Better accuracy, ~90s per minute audio
  - large-v3-turbo: Best accuracy, ~120s per minute audio

Compute types (for CPU speed):
  - int8: 2-3x faster, minimal accuracy loss (recommended)
  - float16: Standard, moderate speed
"""
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: small - best for CPU)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples (default: 3)"
    )
    
    parser.add_argument(
        "--compute-type", "-c",
        type=str,
        default="int8",
        choices=["int8", "int8_float16", "float16"],
        help="Compute type for CPU optimization (default: int8)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        model_size=args.model,
        limit=args.limit,
        compute_type=args.compute_type
    )


if __name__ == "__main__":
    main()
