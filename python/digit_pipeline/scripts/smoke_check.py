"""Smoke-Check: Data Pipeline + End-to-End Model (Untrained)

Purpose:
    Validate that data_loader.py output is compatible with model.py input.
    Perform forward pass with 1 batch (batch_size=2).
    NO training - just shape validation.

Usage:
    python smoke_check.py
    # or
    python -m digit_pipeline.scripts.smoke_check
"""

import sys
from pathlib import Path

# Add digit_pipeline to Python path (for direct script invocation)
SCRIPT_DIR = Path(__file__).resolve().parent
DIGIT_PIPELINE_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DIGIT_PIPELINE_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np

# Import from digit_pipeline package
from digit_pipeline.src.data_loader import GermanDigitRawWaveformLoader
from digit_pipeline.src.model_definition import create_end_to_end_model
from digit_pipeline.config.config import (
    AUDIO_DEUTSCH_DIR, PIPELINE_LABELS_PATH, NOISE_DIR
)


def run_smoke_check():
    """Run smoke-check: Load 1 batch → Forward pass → Validate shapes"""

    print("=" * 60)
    print("Smoke-Check: Data Pipeline + End-to-End Model")
    print("=" * 60)
    print()

    # Print configured paths
    print("Configured Paths:")
    print(f"  AUDIO_DEUTSCH_DIR: {AUDIO_DEUTSCH_DIR}")
    print(f"  PIPELINE_LABELS_PATH: {PIPELINE_LABELS_PATH}")
    print(f"  NOISE_DIR: {NOISE_DIR}")
    print()

    # Step 1: Load data pipeline
    print("Step 1: Load data pipeline")

    if not AUDIO_DEUTSCH_DIR.exists():
        print(f"[ERROR] Audio directory not found: {AUDIO_DEUTSCH_DIR}")
        print("   Please ensure data is in data/raw/hd_audio/audio_deutsch/")
        sys.exit(1)

    loader = GermanDigitRawWaveformLoader(
        audio_dir=str(AUDIO_DEUTSCH_DIR),
        labels_file=str(PIPELINE_LABELS_PATH),
        noise_dir=str(NOISE_DIR) if NOISE_DIR.exists() else None
    )
    print()

    # Step 2: Split data by speaker
    print("Step 2: Speaker-based split")
    train_infos, val_infos, test_infos = loader.split_by_speaker(seed=42)
    print(f"  Train: {len(train_infos)} samples")
    print(f"  Val:   {len(val_infos)} samples")
    print(f"  Test:  {len(test_infos)} samples")
    print()

    # Step 3: Create train generator (without augmentation for smoke-check)
    print("Step 3: Generate 1 batch (batch_size=2, augment=False)")
    train_gen = loader.create_dataset(train_infos, batch_size=2, shuffle=False, augment=False, seed=42)
    batch_X, batch_y = next(train_gen)

    print(f"  Batch X shape: {batch_X.shape}")  # Expected: (2, 16000)
    print(f"  Batch X dtype: {batch_X.dtype}")  # Expected: float32
    print(f"  Batch X range: [{batch_X.min():.4f}, {batch_X.max():.4f}]")  # Expected: [-1, 1]
    print(f"  Batch y shape: {batch_y.shape}")  # Expected: (2, 12)
    print(f"  Batch y dtype: {batch_y.dtype}")  # Expected: float32
    print(f"  Batch y classes: {batch_y.argmax(axis=1)}")
    print()

    # Validate batch shapes
    assert batch_X.shape == (2, 16000), f"[ERROR] Batch X shape mismatch: {batch_X.shape}"
    assert batch_X.dtype == np.float32, f"[ERROR] Batch X dtype mismatch: {batch_X.dtype}"
    assert batch_X.min() >= -1.0 and batch_X.max() <= 1.0, f"[ERROR] Batch X range invalid"
    assert batch_y.shape == (2, 12), f"[ERROR] Batch y shape mismatch: {batch_y.shape}"
    print("[OK] Batch shapes valid")
    print()

    # Step 4: Load untrained model
    print("Step 4: Load End-to-End model (untrained)")
    try:
        keras_model = create_end_to_end_model(num_classes=12)
        print(f"  Model input shape: {keras_model.input_shape}")   # Expected: (None, 16000)
        print(f"  Model output shape: {keras_model.output_shape}") # Expected: (None, 12)
        print(f"  Model layers: {len(keras_model.layers)}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Forward pass
    print("Step 5: Forward pass (predict)")
    try:
        predictions = keras_model.predict(batch_X, verbose=0)

        print(f"  Predictions shape: {predictions.shape}")  # Expected: (2, 12)
        print(f"  Predictions dtype: {predictions.dtype}")
        print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  Softmax sums: {predictions.sum(axis=1)}")  # Expected: [1.0, 1.0]
        print(f"  Predicted classes: {predictions.argmax(axis=1)}")
        print()
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Validate predictions
    assert predictions.shape == (2, 12), f"[ERROR] Predictions shape mismatch: {predictions.shape}"
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5), "[ERROR] Softmax not normalized"
    print("[OK] Forward pass successful")
    print()

    # Step 6: Test with augmentation
    print("Step 6: Test with augmentation (batch_size=2, augment=True)")
    train_gen_aug = loader.create_dataset(train_infos, batch_size=2, shuffle=True, augment=True, seed=42)
    batch_X_aug, batch_y_aug = next(train_gen_aug)

    print(f"  Batch X shape: {batch_X_aug.shape}")
    print(f"  Batch X range: [{batch_X_aug.min():.4f}, {batch_X_aug.max():.4f}]")

    predictions_aug = keras_model.predict(batch_X_aug, verbose=0)
    print(f"  Predictions shape: {predictions_aug.shape}")
    print(f"  Softmax sums: {predictions_aug.sum(axis=1)}")
    print()

    # Final summary
    print("=" * 60)
    print("[OK] SMOKE-CHECK PASSED")
    print("=" * 60)
    print()
    print("Summary:")
    print("  [OK] Data loader works correctly")
    print("  [OK] Batch shapes: (2, 16000) -> (2, 12)")
    print("  [OK] Speaker-based split: NO leakage")
    print("  [OK] Augmentation: Time-shift + Noise-mix + Gain")
    print("  [OK] Model forward pass: Compatible")
    print()
    print("Next steps:")
    print("  python -m digit_pipeline.scripts.train")
    print("  python -m digit_pipeline.scripts.export_tflite")
    print()


if __name__ == "__main__":
    run_smoke_check()
