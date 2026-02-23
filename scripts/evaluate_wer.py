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

from datasets import load_dataset
from jiwer import process_words
from tqdm import tqdm


def preprocess_urdu_text(text: str) -> str:
    """Preprocess Urdu text for WER calculation."""
    if not text:
        return ""
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF۔،؟ؤ ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيٱٲٳٴٵٶٷٸٹٺٻټٽپٿٹڈډڑژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ەۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬ۮۯ۰۱۲۳۴۵۶۷۸۹۝۔۔۔]',
                  '', text)
    
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
        segments, info = model.transcribe(audio_path, language="ur", beam_size=1)
        text = " ".join([seg.text.strip() for seg in segments])
        return text
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""


def load_urdu_dataset(limit: int = None):
    """Load Urdu dataset from Hugging Face (smaller datasets for CPU)."""
    print("Loading Urdu dataset...")
    
    datasets_to_try = [
        ("humair025/UrduSpeech-IndicVoices-ST-kProcessed", "train"),
        ("Talha185/Common-voice-urdu-11", "train"),
        ("mozilla-foundation/common_voice_17_0", "ur", "train"),
    ]
    
    for ds_config in datasets_to_try:
        try:
            ds_name = ds_config[0]
            print(f"Trying dataset: {ds_name}...")
            
            if len(ds_config) == 2:
                dataset = load_dataset(ds_name, split=ds_config[1])
            else:
                dataset = load_dataset(ds_name, lang=ds_config[1], split=ds_config[2])
            
            print(f"Successfully loaded {ds_name}: {len(dataset)} samples")
            break
        except Exception as e:
            print(f"Failed to load {ds_config[0]}: {e}")
            continue
    else:
        print("All datasets failed. Using synthetic test data.")
        return None
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples")
    return dataset


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


def run_benchmark(model_size: str = "small", limit: int = None, compute_type: str = "int8", use_synthetic: bool = False):
    """Run WER benchmark on Urdu dataset (CPU-optimized)."""
    print(f"\n{'='*60}")
    print(f"WER Benchmark for Urdu Speech Recognition (CPU)")
    print(f"{'='*60}")
    print(f"Model: whisper-{model_size}")
    print(f"Compute type: {compute_type}")
    print(f"Samples: {limit if limit else 'all'}")
    print(f"{'='*60}\n")
    
    if use_synthetic:
        print("Using synthetic test data for demonstration...")
        synthetic_data = [
            {"text": "یہ ایک جانچ ہے", "hypothesis": "یہ ایک جانچ ہے"},
            {"text": "اردو بہت حسین زبان ہے", "hypothesis": "اردو بہت اچھی زبان ہے"},
            {"text": "میں پاکستان سے ہوں", "hypothesis": "میں پاکستان کا ہوں"},
            {"text": "آج بہت اچھا دن ہے", "hypothesis": "آج اچھا دن ہے"},
            {"text": "کمپیوٹر پر کام کر رہا ہوں", "hypothesis": "کمپیوٹر پر کام کر رہا ہوں"},
        ]
        references = [item["text"] for item in synthetic_data[:limit if limit else len(synthetic_data)]]
        hypotheses = [item["hypothesis"] for item in synthetic_data[:limit if limit else len(synthetic_data)]]
        
        print(f"Processed {len(references)} synthetic samples")
        metrics = calculate_wer(references, hypotheses)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS (Synthetic Data)")
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
            print(f"Reference: {references[i]}")
            print(f"Hypothesis: {hypotheses[i]}")
        print("-" * 60)
        return
    
    model = load_model(model_size, compute_type)
    dataset = load_urdu_dataset(limit)
    
    if dataset is None:
        print("Note: Dataset loading failed. Please check internet connection.")
        print("For CPU testing, use smaller models like --model tiny or --model base")
        return
    
    references = []
    hypotheses = []
    
    print("\nTranscribing audio samples...")
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            audio_path = None
            temp_file = None
            
            if "audio" in item:
                audio_data = item["audio"]
                if isinstance(audio_data, dict) and "array" in audio_data:
                    import tempfile
                    import numpy as np
                    import soundfile as sf
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_data["array"].astype(np.float32), audio_data.get("sampling_rate", 16000))
                        audio_path = tmp.name
                    temp_file = tmp.name
                else:
                    audio_path = audio_data
            elif "audio_filepath" in item:
                audio_path = item["audio_filepath"]
            elif "file" in item:
                audio_path = item["file"]
            
            if not audio_path:
                print(f"Sample {idx}: No audio field, skipping")
                continue
            
            reference = item.get("text", item.get("sentence", item.get("transcript", "")))
            if not reference:
                continue
            
            hypothesis = transcribe_audio(model, audio_path)
            
            references.append(reference)
            hypotheses.append(hypothesis)
            
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
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
    
    print("Sample Transcriptions:")
    print("-" * 60)
    for i in range(min(3, len(references))):
        print(f"\nSample {i+1}:")
        print(f"Reference: {references[i][:100]}...")
        print(f"Hypothesis: {hypotheses[i][:100]}...")
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

CPU-Optimized models (recommended for CPU):
  - tiny: Fastest, ~6s per minute audio
  - base: Quick, ~18s per minute audio
  - small: Best balance, ~36s per minute audio (recommended)
  - medium: Better accuracy, ~90s per minute audio

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
        help="Limit number of samples (default: all)"
    )
    
    parser.add_argument(
        "--compute-type", "-c",
        type=str,
        default="int8",
        choices=["int8", "int8_float16", "float16"],
        help="Compute type for CPU optimization (default: int8)"
    )
    
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Use synthetic test data (no internet/dataset needed)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        model_size=args.model,
        limit=args.limit,
        compute_type=args.compute_type,
        use_synthetic=args.synthetic
    )


if __name__ == "__main__":
    main()
