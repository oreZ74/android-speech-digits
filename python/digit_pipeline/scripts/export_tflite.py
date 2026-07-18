"""Export TFLite Model with Validation

Export trained model to TFLite format and validate on test samples.

Date: 2025-11-25

Usage:
    python export_tflite.py
    # or
    python -m digit_pipeline.scripts.export_tflite
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
from pathlib import Path
import sys

# Import from digit_pipeline package
from digit_pipeline.src.data_loader import GermanDigitRawWaveformLoader
from digit_pipeline.config.config import (
    AUDIO_DEUTSCH_DIR, MYDIGITS_DIR, PIPELINE_LABELS_PATH,
    NOISE_DIR, SPEECH_COMMANDS_DIR, TRAINING_OUTPUT_DIR,
    TFLITE_DIR, TFLITE_MODEL_PATH, TrainingConfig
)

# Training specific paths
SAVED_MODEL = TRAINING_OUTPUT_DIR / "saved_model"
TFLITE_PATH = TFLITE_DIR / "digits_rawwave_12cls.tflite"
TFLITE_PATH_LOCAL = TRAINING_OUTPUT_DIR / "digits_rawwave_12cls.tflite"

VALIDATION_SAMPLES = 100
SEED = TrainingConfig.SEED


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def convert_to_tflite(model):
    """Convert Keras model to TFLite with SELECT_TF_OPS"""
    print_section("TFLite Conversion")

    print("Converting model with SELECT_TF_OPS (Flex Delegate)...")
    print("[WARNING] SELECT_TF_OPS adds ~4 MB to APK")

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        tflite_model = converter.convert()

        # Save model to both locations
        TFLITE_DIR.mkdir(parents=True, exist_ok=True)
        TFLITE_PATH.write_bytes(tflite_model)
        TFLITE_PATH_LOCAL.write_bytes(tflite_model)

        model_size_mb = len(tflite_model) / (1024 * 1024)

        print(f"[SUCCESS] TFLite conversion succeeded!")
        print(f"[INFO] Model saved to:")
        print(f"       - {TFLITE_PATH}")
        print(f"       - {TFLITE_PATH_LOCAL}")
        print(f"[INFO] Model size: {model_size_mb:.2f} MB")

        return tflite_model

    except Exception as e:
        print(f"[FAILED] TFLite conversion failed!")
        print(f"[ERROR] {e}")
        sys.exit(1)


def validate_tflite(tflite_model, keras_model, loader, test_infos):
    """Validate TFLite model against Keras model"""
    print_section("TFLite Validation")

    # Initialize TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite Model Info:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    # Sample validation samples
    np.random.seed(SEED)
    validation_indices = np.random.choice(len(test_infos), min(VALIDATION_SAMPLES, len(test_infos)), replace=False)
    validation_infos = [test_infos[i] for i in validation_indices]

    print(f"\nValidating on {len(validation_infos)} test samples...")

    class_match_count = 0
    max_prob_diff = 0.0
    total_prob_diff = 0.0

    for i, info in enumerate(validation_infos):
        # Load audio
        waveform = loader.load_waveform(info.file_path)
        waveform = waveform[np.newaxis, ...]  # Add batch dimension

        # Keras prediction
        keras_pred = keras_model.predict(waveform, verbose=0)[0]
        keras_class = np.argmax(keras_pred)

        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], waveform.astype(np.float32))
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0]
        tflite_class = np.argmax(tflite_pred)

        # Compare
        if keras_class == tflite_class:
            class_match_count += 1

        # Probability difference
        prob_diff = np.abs(keras_pred - tflite_pred).max()
        max_prob_diff = max(max_prob_diff, prob_diff)
        total_prob_diff += prob_diff

        # Print progress every 20 samples
        if (i + 1) % 20 == 0:
            print(f"  Validated {i+1}/{len(validation_infos)} samples...")

    # Results
    class_match_rate = class_match_count / len(validation_infos)
    avg_prob_diff = total_prob_diff / len(validation_infos)

    print("\n" + "-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)
    print(f"Class Match Rate: {class_match_count}/{len(validation_infos)} ({class_match_rate*100:.2f}%)")
    print(f"Max Probability Diff: {max_prob_diff:.6f}")
    print(f"Avg Probability Diff: {avg_prob_diff:.6f}")

    if class_match_rate == 1.0 and max_prob_diff < 1e-5:
        print("\n[OK] VALIDATION PASSED: TFLite model matches Keras model!")
    elif class_match_rate == 1.0:
        print(f"\n[WARN] Classes match but probability diff is {max_prob_diff:.6f}")
    else:
        print(f"\n[ERROR] VALIDATION FAILED: Only {class_match_rate*100:.2f}% class match!")

    return class_match_rate, max_prob_diff, avg_prob_diff


def main():
    """Main export pipeline"""
    print_section("TFLite Export & Validation")

    # Check if model exists
    if not SAVED_MODEL.exists():
        print(f"[ERROR] Model not found: {SAVED_MODEL}")
        print("Please run train.py first.")
        sys.exit(1)

    # Load Keras model
    print(f"Loading model from: {SAVED_MODEL}")
    keras_model = keras.models.load_model(str(SAVED_MODEL))
    print(f"[OK] Model loaded: {keras_model.count_params():,} parameters")

    # Convert to TFLite
    tflite_model = convert_to_tflite(keras_model)

    # Load test data for validation
    print("\nLoading test data for validation...")
    loader = GermanDigitRawWaveformLoader(
        audio_dir=str(AUDIO_DEUTSCH_DIR),
        labels_file=str(PIPELINE_LABELS_PATH),
        noise_dir=str(NOISE_DIR),
        speech_commands_dir=str(SPEECH_COMMANDS_DIR),
        unknown_samples_per_class=50,
        mydigits_dir=str(MYDIGITS_DIR)
    )

    train_infos, val_infos, test_infos = loader.split_by_speaker(seed=SEED)
    print(f"[OK] Loaded {len(test_infos)} test samples")

    # Validate TFLite model
    class_match_rate, max_prob_diff, avg_prob_diff = validate_tflite(
        tflite_model, keras_model, loader, test_infos
    )

    # Save validation report
    report_path = TRAINING_OUTPUT_DIR / "tflite_validation.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TFLite Validation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Keras Model: {SAVED_MODEL}\n")
        f.write(f"TFLite Model: {TFLITE_PATH}\n")
        f.write(f"TFLite Size: {TFLITE_PATH.stat().st_size / (1024*1024):.2f} MB\n\n")
        f.write(f"Validation Samples: {VALIDATION_SAMPLES}\n")
        f.write(f"Class Match Rate: {class_match_rate*100:.2f}%\n")
        f.write(f"Max Probability Diff: {max_prob_diff:.6f}\n")
        f.write(f"Avg Probability Diff: {avg_prob_diff:.6f}\n\n")

        if class_match_rate == 1.0 and max_prob_diff < 1e-5:
            f.write("Status: PASSED\n")
        else:
            f.write("Status: WARNING - Review differences\n")

    print(f"\n[OK] Validation report saved: {report_path}")

    print_section("Export Complete")
    print(f"TFLite Model: {TFLITE_PATH}")
    print(f"Model Size: {TFLITE_PATH.stat().st_size / (1024*1024):.2f} MB")
    print(f"Validation: {class_match_rate*100:.2f}% class match")


if __name__ == "__main__":
    main()
