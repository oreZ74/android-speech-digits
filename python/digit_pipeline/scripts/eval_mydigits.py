"""Evaluate MyDigits Manual Recordings

Evaluate the trained model's performance on smartphone recordings.
Output: Confusion matrix and per-digit accuracy

Usage:
    python eval_mydigits.py
    # or
    python -m digit_pipeline.scripts.eval_mydigits
"""

import sys
from pathlib import Path

# Add digit_pipeline to Python path (for direct script invocation)
SCRIPT_DIR = Path(__file__).resolve().parent
DIGIT_PIPELINE_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DIGIT_PIPELINE_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

# Import from digit_pipeline package
from digit_pipeline.config.config import (
    MYDIGITS_ANDROID_DIR, MYDIGITS_PC_DIR, TRAINING_OUTPUT_DIR,
    ensure_directories
)

# Paths (derived from centralized config)
SAVED_MODEL = TRAINING_OUTPUT_DIR / "saved_model"
OUTPUT_DIR = TRAINING_OUTPUT_DIR

# Constants
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
LABEL_NAMES = [str(i) for i in range(10)] + ['_silence_', '_unknown_']


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_audio(file_path: Path) -> np.ndarray:
    """Load and preprocess audio file (same as data_loader.py)"""
    # Load audio with librosa
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Ensure float32
    waveform = waveform.astype(np.float32)

    # Pad or truncate to exactly TARGET_LENGTH
    if len(waveform) < TARGET_LENGTH:
        pad_length = TARGET_LENGTH - len(waveform)
        waveform = np.pad(waveform, (0, pad_length), mode='constant', constant_values=0.0)
    elif len(waveform) > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]

    # Ensure range [-1, 1] (librosa already returns normalized float)
    # Clip just in case
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform


def load_mydigits_samples():
    """Load all MyDigits samples from both Android and PC sources"""
    print_section("Loading MyDigits Samples")

    has_android = MYDIGITS_ANDROID_DIR.exists()
    has_pc = MYDIGITS_PC_DIR.exists()

    if not has_android and not has_pc:
        print(f"[ERROR] No MyDigits directories found!")
        print(f"  Android: {MYDIGITS_ANDROID_DIR} - NOT FOUND")
        print(f"  PC: {MYDIGITS_PC_DIR} - NOT FOUND")
        print("Please run: python -m digit_pipeline.scripts.register_mydigits")
        sys.exit(1)

    samples = []  # List of (digit, file_path, waveform, source)

    # Load from each source
    for source_name, source_dir, exists in [
        ("Android", MYDIGITS_ANDROID_DIR, has_android),
        ("PC", MYDIGITS_PC_DIR, has_pc)
    ]:
        if not exists:
            print(f"[WARN] {source_name} directory not found, skipping")
            continue

        print(f"\n{source_name} recordings ({source_dir}):")
        source_count = 0

        for digit in range(10):
            digit_dir = source_dir / str(digit)

            if not digit_dir.exists():
                print(f"  [WARN] Digit {digit} directory not found")
                continue

            wav_files = sorted(digit_dir.glob("*.wav"))
            print(f"  Digit {digit}: {len(wav_files)} files")

            for wav_file in wav_files:
                waveform = load_audio(wav_file)
                samples.append((digit, wav_file, waveform, source_name))
                source_count += 1

        print(f"  Total: {source_count} samples")

    print(f"\n[OK] Loaded {len(samples)} total samples")
    return samples


def evaluate_model(model, samples, model_name):
    """Evaluate model on MyDigits samples"""
    print_section(f"Evaluating {model_name}")

    y_true = []
    y_pred = []
    predictions_detail = []

    for digit, file_path, waveform, source in samples:
        # Predict
        pred = model.predict(waveform[np.newaxis, ...], verbose=0)
        pred_label = np.argmax(pred)
        pred_conf = np.max(pred)

        y_true.append(digit)
        y_pred.append(pred_label)

        predictions_detail.append({
            'file': file_path.name,
            'source': source,
            'true_digit': digit,
            'pred_label': pred_label,
            'pred_name': LABEL_NAMES[pred_label],
            'confidence': pred_conf
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")

    # Per-digit accuracy
    print("\nPer-Digit Accuracy:")
    for digit in range(10):
        mask = (y_true == digit)
        if mask.sum() > 0:
            digit_acc = (y_pred[mask] == digit).sum() / mask.sum()
            print(f"  Digit {digit}: {digit_acc*100:5.1f}% ({(y_pred[mask] == digit).sum()}/{mask.sum()})")

    # Count predictions to special classes
    pred_silence = (y_pred == 10).sum()
    pred_unknown = (y_pred == 11).sum()
    pred_digits = ((y_pred >= 0) & (y_pred <= 9)).sum()

    print(f"\nPrediction Distribution:")
    print(f"  Digits (0-9): {pred_digits} ({pred_digits/len(y_pred)*100:.1f}%)")
    print(f"  _silence_:    {pred_silence} ({pred_silence/len(y_pred)*100:.1f}%)")
    print(f"  _unknown_:    {pred_unknown} ({pred_unknown/len(y_pred)*100:.1f}%)")

    # Confusion matrix (digits only)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    return acc, cm, predictions_detail, y_true, y_pred


def plot_single_confusion_matrix(cm, acc):
    """Plot confusion matrix"""
    print_section("Plotting Confusion Matrix")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'MyDigits Evaluation (Acc: {acc*100:.1f}%)', fontsize=14, pad=15)
    plt.ylabel('True Digit', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / 'mydigits_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix: {plot_path}")


def generate_evaluation_report(acc, cm, samples):
    """Generate markdown evaluation report"""
    print_section("Generating Evaluation Report")

    report_path = OUTPUT_DIR / '../MYDIGITS_EVAL.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MyDigits Evaluation Report\n\n")
        f.write("## Objective\n\n")
        f.write("Evaluate model performance on smartphone recordings.\n\n")

        f.write("## Test Set\n\n")
        f.write(f"- **Source:** MyDigits smartphone recordings\n")
        f.write(f"- **Total samples:** {len(samples)}\n")
        f.write(f"- **Digits:** 0-9 (10 classes)\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Model | Overall Accuracy |\n")
        f.write("|-------|------------------|\n")
        f.write(f"| With MyDigits | {acc*100:.2f}% |\n\n")

        f.write("## Per-Digit Accuracy\n\n")
        f.write("| Digit | Accuracy |\n")
        f.write("|-------|----------|\n")

        for digit in range(10):
            # Calculate per-digit accuracy from confusion matrix
            acc_digit = cm[digit, digit] / cm[digit, :].sum() if cm[digit, :].sum() > 0 else 0
            f.write(f"| {digit} | {acc_digit*100:.1f}% |\n")

        f.write("\n## Conclusion\n\n")
        f.write(f"**Domain adaptation successful!** Model achieved {acc*100:.1f}% accuracy on smartphone recordings.\n\n")

        f.write("## Visualization\n\n")
        f.write("See `mydigits_evaluation.png` for confusion matrix.\n")

    print(f"[OK] Saved evaluation report: {report_path}")


def main():
    """Main evaluation pipeline"""
    print_section("MyDigits Evaluation")

    # Check model exists
    if not SAVED_MODEL.exists():
        print(f"[ERROR] Model not found: {SAVED_MODEL}")
        print("Please run train.py first.")
        sys.exit(1)

    # Load samples
    samples = load_mydigits_samples()

    # Load model
    print("\nLoading model...")
    model = keras.models.load_model(str(SAVED_MODEL))

    # Evaluate model
    acc, cm, pred, y_true, y_pred = evaluate_model(model, samples, "Model")

    # Plot single confusion matrix
    plot_single_confusion_matrix(cm, acc)

    # Generate report
    generate_evaluation_report(acc, cm, samples)

    print_section("Evaluation Complete")
    print(f"Model Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
