"""Register MyDigits Manual Recordings for Domain Adaption

This script processes manual recordings from Android/PC and integrates them
into the training pipeline for domain adaptation.

Input:
    data/raw/manual_recordings/mydigits_android/0-9/*.wav (950 files)
    data/raw/manual_recordings/mydigits_pc/0-9/*.wav (100 files)

Output: Processed 16kHz Mono WAV files

Usage:
    python register_mydigits.py                   # Process both sources
    python register_mydigits.py --source android  # Only Android recordings
    python register_mydigits.py --source pc       # Only PC recordings
    # or
    python -m digit_pipeline.scripts.register_mydigits
"""

import sys
from pathlib import Path

# Add digit_pipeline to Python path (for direct script invocation)
SCRIPT_DIR = Path(__file__).resolve().parent
DIGIT_PIPELINE_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DIGIT_PIPELINE_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

import librosa
import soundfile as sf
import numpy as np

# Import from digit_pipeline package
from digit_pipeline.config.config import (
    MYDIGITS_ANDROID_DIR, MYDIGITS_PC_DIR, MANUAL_RECORDINGS_DIR
)


def process_mydigits_source(
    source_path: Path,
    source_name: str,
    target_sr: int = 16000,
    target_duration: float = 1.0,
    in_place: bool = True
) -> dict:
    """
    Process MyDigits recordings from a single source directory.

    Processing:
    1. Check/convert to 16kHz Mono
    2. Pad/truncate to 1.0 second
    3. Save processed files (in-place or to new location)

    Args:
        source_path: Source directory with manual recordings
        source_name: Name of the source for logging
        target_sr: Target sample rate (16000 Hz)
        target_duration: Target duration in seconds (1.0s)
        in_place: If True, overwrite source files; else copy to processed dir

    Returns:
        Statistics dictionary
    """
    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        print(f"        Please create {source_path}/0-9/ and add WAV files.")
        return {'total': 0, 'error': True}

    stats = {
        'total': 0,
        'resampled': 0,
        'converted_to_mono': 0,
        'padded': 0,
        'truncated': 0,
        'errors': 0,
        'per_digit': {str(d): 0 for d in range(10)}
    }

    target_samples = int(target_sr * target_duration)

    print(f"\n{'='*60}")
    print(f"Processing: {source_name}")
    print(f"Source: {source_path}")
    print(f"{'='*60}")

    # Process each digit class
    for digit in range(10):
        digit_source = source_path / str(digit)

        if not digit_source.exists():
            print(f"[WARN] Digit {digit} directory not found: {digit_source}")
            continue

        wav_files = list(digit_source.glob("*.wav"))
        print(f"\nDigit {digit}: {len(wav_files)} files")

        for wav_file in wav_files:
            try:
                # Load audio
                audio, sr = librosa.load(wav_file, sr=None, mono=False)
                needs_update = False

                # Convert to mono if needed
                if audio.ndim > 1:
                    audio = librosa.to_mono(audio)
                    stats['converted_to_mono'] += 1
                    needs_update = True

                # Resample if needed
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    stats['resampled'] += 1
                    print(f"  Resampled: {wav_file.name} ({sr}Hz -> {target_sr}Hz)")
                    needs_update = True

                # Pad or truncate to target length
                if len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
                    stats['padded'] += 1
                    needs_update = True
                elif len(audio) > target_samples:
                    audio = audio[:target_samples]
                    stats['truncated'] += 1
                    needs_update = True

                # Save file (in-place update)
                if needs_update or in_place:
                    sf.write(wav_file, audio, target_sr, subtype='PCM_16')

                stats['total'] += 1
                stats['per_digit'][str(digit)] += 1

            except Exception as e:
                print(f"  [ERROR] Processing {wav_file.name}: {e}")
                stats['errors'] += 1

    return stats


def register_mydigits(
    source: str = "all",
    target_sr: int = 16000,
    target_duration: float = 1.0
):
    """
    Process and register MyDigits recordings from Android and/or PC.

    Args:
        source: Which source to process - "android", "pc", or "all"
        target_sr: Target sample rate (16000 Hz)
        target_duration: Target duration in seconds (1.0s)
    """
    sources = []

    if source in ["all", "android"]:
        sources.append((MYDIGITS_ANDROID_DIR, "MyDigits Android"))
    if source in ["all", "pc"]:
        sources.append((MYDIGITS_PC_DIR, "MyDigits PC"))

    if not sources:
        print(f"[ERROR] Unknown source: {source}")
        print("        Valid options: 'android', 'pc', 'all'")
        return

    all_stats = []

    for src_path, src_name in sources:
        stats = process_mydigits_source(
            source_path=src_path,
            source_name=src_name,
            target_sr=target_sr,
            target_duration=target_duration
        )
        all_stats.append((src_name, stats))

    # Print overall summary
    print("\n" + "="*60)
    print("REGISTRATION SUMMARY")
    print("="*60)

    grand_total = 0
    for src_name, stats in all_stats:
        if stats.get('error'):
            print(f"\n{src_name}: [NOT FOUND]")
            continue

        print(f"\n{src_name}:")
        print(f"  Total files: {stats['total']}")
        print(f"  Resampled: {stats['resampled']}")
        print(f"  Converted to mono: {stats['converted_to_mono']}")
        print(f"  Padded: {stats['padded']}")
        print(f"  Truncated: {stats['truncated']}")
        print(f"  Errors: {stats['errors']}")
        per_digit_str = ', '.join(f"{d}:{stats['per_digit'][str(d)]}" for d in range(10))
        print(f"  Per digit: {per_digit_str}")
        grand_total += stats['total']

    print(f"\n{'='*60}")
    print(f"[OK] Grand total: {grand_total} files processed")
    print(f"[OK] Ready for data_loader.py integration")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register MyDigits recordings")
    parser.add_argument(
        '--source',
        type=str,
        default="all",
        choices=["all", "android", "pc"],
        help='Which source to process: "android", "pc", or "all" (default: all)'
    )

    args = parser.parse_args()

    register_mydigits(source=args.source)
