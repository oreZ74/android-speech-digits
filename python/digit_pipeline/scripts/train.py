"""
End-to-End Training with MyDigits Smartphone Recordings

Training Strategy:
1. Build fresh End-to-End model (Raw-Waveform -> MFCC -> CNN)
2. Include MyDigits manual smartphone recordings (0-9) in training data
3. Train from scratch with all data sources combined
4. Goal: Strong generalization to real smartphone audio

Dataset:
- Zenke HD-Audio FLAC digits (0-9)
- Noise samples for _silence_ class
- Speech Commands for _unknown_ class
- MyDigits Android recordings (0-9)

Date: 2025-11-25

Usage:
    python train.py
    # or
    python -m digit_pipeline.scripts.train
"""

import sys
import os
from pathlib import Path

# Add digit_pipeline to Python path (for direct script invocation)
SCRIPT_DIR = Path(__file__).resolve().parent
DIGIT_PIPELINE_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DIGIT_PIPELINE_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import from digit_pipeline package
from digit_pipeline.src.data_loader import GermanDigitRawWaveformLoader
from digit_pipeline.src.model_definition import create_end_to_end_model
from digit_pipeline.config.config import (
    AUDIO_DEUTSCH_DIR, MYDIGITS_ANDROID_DIR, MYDIGITS_PC_DIR, PIPELINE_LABELS_PATH,
    NOISE_DIR, SPEECH_COMMANDS_DIR, TRAINING_OUTPUT_DIR,
    TrainingConfig
)

# Phase 4C Baseline values for comparison (reference only)
PHASE4C_TEST_ACC = 0.9634
PHASE4C_UNKNOWN_RECALL = 0.9980

# Ensure output directory exists
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters from config
BATCH_SIZE = TrainingConfig.BATCH_SIZE
EPOCHS = TrainingConfig.EPOCHS
LEARNING_RATE = TrainingConfig.LEARNING_RATE
SEED = TrainingConfig.SEED
UNKNOWN_SAMPLES_PER_CLASS = TrainingConfig.UNKNOWN_SAMPLES_PER_CLASS
EARLY_STOPPING_PATIENCE = getattr(TrainingConfig, 'EARLY_STOPPING_PATIENCE', 7)

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_data():
    """Load train/val/test data with MyDigits recordings added"""
    print_section("Load Data with MyDigits Recordings")

    # Check if at least one MyDigits directory exists
    has_android = MYDIGITS_ANDROID_DIR.exists()
    has_pc = MYDIGITS_PC_DIR.exists()

    if not has_android and not has_pc:
        print(f"[ERROR] No MyDigits recordings found!")
        print(f"  Android: {MYDIGITS_ANDROID_DIR} - NOT FOUND")
        print(f"  PC: {MYDIGITS_PC_DIR} - NOT FOUND")
        print("Please run: python -m digit_pipeline.scripts.register_mydigits")
        sys.exit(1)

    print(f"MyDigits Android: {MYDIGITS_ANDROID_DIR} - {'FOUND' if has_android else 'NOT FOUND'}")
    print(f"MyDigits PC: {MYDIGITS_PC_DIR} - {'FOUND' if has_pc else 'NOT FOUND'}")

    # Initialize loader with Speech Commands + MyDigits (Android + PC)
    print("\nInitializing data loader with Speech Commands + MyDigits recordings...")
    loader = GermanDigitRawWaveformLoader(
        audio_dir=str(AUDIO_DEUTSCH_DIR),
        labels_file=str(PIPELINE_LABELS_PATH),
        noise_dir=str(NOISE_DIR),
        speech_commands_dir=str(SPEECH_COMMANDS_DIR),
        unknown_samples_per_class=UNKNOWN_SAMPLES_PER_CLASS,
        mydigits_android_dir=str(MYDIGITS_ANDROID_DIR) if has_android else None,
        mydigits_pc_dir=str(MYDIGITS_PC_DIR) if has_pc else None
    )
    print()

    # Hybrid split (speaker-based for digits, sample-based for special classes)
    print("Performing hybrid split (digits + MyDigits as separate speaker)...")
    train_infos, val_infos, test_infos = loader.split_by_speaker(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED
    )
    print()

    # Class distribution per split
    print("Class distribution per split:")
    for split_name, split_infos in [("Train", train_infos), ("Val", val_infos), ("Test", test_infos)]:
        class_counts = {}
        for info in split_infos:
            class_counts[info.label] = class_counts.get(info.label, 0) + 1

        print(f"\n  {split_name}:")
        for label_idx in range(12):
            label_name = loader.label_names[label_idx]
            count = class_counts.get(label_idx, 0)
            print(f"    {label_name:12} (idx {label_idx:2}): {count:4} samples")
        print(f"    {'Total':12}        : {len(split_infos):4} samples")

    # Count MyDigits samples per split
    print("\n  MyDigits samples distribution:")
    for split_name, split_infos in [("Train", train_infos), ("Val", val_infos), ("Test", test_infos)]:
        android_count = sum(1 for info in split_infos if info.speaker_id == 'mydigits_android')
        pc_count = sum(1 for info in split_infos if info.speaker_id == 'mydigits_pc')
        print(f"    {split_name}: {android_count} Android + {pc_count} PC = {android_count + pc_count} MyDigits")
    print()

    # Create generators
    print("Creating data generators...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train: augment=True, shuffle=True")
    print(f"  Val:   augment=False, shuffle=False")
    print(f"  Test:  augment=False, shuffle=False")
    print(f"\n  Augmentation settings:")
    print(f"    Time shift: ±{loader.TIME_SHIFT_RANGE} samples (±{loader.TIME_SHIFT_RANGE*1000/16000:.0f}ms)")
    print(f"    Noise mix prob: {loader.NOISE_MIX_PROB*100:.0f}%")
    print(f"    Noise volume: {loader.NOISE_VOLUME_RANGE}")
    print(f"    Gain range: {loader.GAIN_RANGE}")

    train_gen = loader.create_dataset(
        train_infos, batch_size=BATCH_SIZE, shuffle=True, augment=True, seed=SEED
    )
    val_gen = loader.create_dataset(
        val_infos, batch_size=BATCH_SIZE, shuffle=False, augment=False, seed=SEED
    )
    test_gen = loader.create_dataset(
        test_infos, batch_size=BATCH_SIZE, shuffle=False, augment=False, seed=SEED
    )

    steps_per_epoch = len(train_infos) // BATCH_SIZE
    val_steps = len(val_infos) // BATCH_SIZE
    test_steps = len(test_infos) // BATCH_SIZE

    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Val steps: {val_steps}")
    print(f"  Test steps: {test_steps}")

    return loader, train_gen, val_gen, test_gen, steps_per_epoch, val_steps, test_steps, test_infos


def build_model(loader: GermanDigitRawWaveformLoader) -> keras.Model:
    """
    Create End-to-End model from scratch.

    Args:
        loader: Data loader instance (used for num_classes and logging)

    Returns:
        Compiled Keras model ready for training
    """
    print_section("Build Model (from scratch)")

    num_classes = len(loader.label_names)
    desired_samples = loader.TARGET_LENGTH  # 16000 @ 16kHz

    print("Creating new end-to-end model (Raw-Waveform -> MFCC -> CNN)")
    print(f"  Input samples: {desired_samples}")
    print(f"  Output classes: {num_classes}")

    # Create model - create_end_to_end_model only takes num_classes
    model = create_end_to_end_model(num_classes=num_classes)

    print(f"\nModel input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    print("\nCompiling model for training...")
    print(f"  Optimizer: Adam(lr={LEARNING_RATE})")
    print(f"  Loss: categorical_crossentropy")
    print(f"  Metrics: accuracy")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, train_gen, val_gen, steps_per_epoch, val_steps):
    """Train model from scratch with callbacks"""
    print_section("Training")

    # Callbacks
    saved_model_dir = TRAINING_OUTPUT_DIR / 'best_model'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(saved_model_dir),  # SavedModel format
            monitor='val_accuracy',
            save_best_only=True,
            save_format='tf',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Allow more patience before reducing LR
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(TRAINING_OUTPUT_DIR / 'training_log.csv')
        )
    ]

    print(f"Training for up to {EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {val_steps}")
    print(f"  Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, test_gen, test_steps, loader, test_infos):
    """Evaluate model on test set"""
    print_section("Evaluate Model")

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, steps=test_steps, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Predictions for confusion matrix
    print("\nGenerating predictions for confusion matrix...")
    y_true = []
    y_pred = []

    for info in test_infos:
        # Load audio
        waveform = loader.load_waveform(info.file_path)

        # Predict
        pred = model.predict(waveform[np.newaxis, ...], verbose=0)
        pred_label = np.argmax(pred)

        y_true.append(info.label)
        y_pred.append(pred_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # All possible labels (0-11)
    all_labels = list(range(len(loader.label_names)))

    # Classification report (compute once, reuse)
    report_str = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=loader.label_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_str)

    # Save report
    report_path = TRAINING_OUTPUT_DIR / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report_str)
    print(f"[OK] Saved classification report: {report_path}")

    # Confusion matrix (include all labels)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=loader.label_names,
        yticklabels=loader.label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix (12 Classes)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    cm_path = TRAINING_OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix: {cm_path}")

    # Special classes confusion matrix (10-11 only)
    special_mask = (y_true >= 10)
    if special_mask.sum() > 0:
        y_true_special = y_true[special_mask]
        y_pred_special = y_pred[special_mask]

        cm_special = confusion_matrix(
            y_true_special, y_pred_special,
            labels=[10, 11]
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_special, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['_silence_', '_unknown_'],
            yticklabels=['_silence_', '_unknown_'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Special Classes Confusion Matrix', fontsize=14, pad=15)
        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()

        cm_special_path = TRAINING_OUTPUT_DIR / 'special_confusion_matrix.png'
        plt.savefig(cm_special_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved special confusion matrix: {cm_special_path}")

    return test_acc, cm


def plot_training_curves(history):
    """Plot training curves"""
    print_section("Plot Training Curves")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Train', marker='o')
    ax2.plot(history.history['val_loss'], label='Val', marker='s')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = TRAINING_OUTPUT_DIR / 'training_curves.png'
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training curves: {curves_path}")


def generate_report(history, test_acc, cm, loader):
    """Generate markdown report"""
    print_section("Generate Report")

    report_path = TRAINING_OUTPUT_DIR / '../TRAINING_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Training Report - End-to-End with MyDigits\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Objective\n\n")
        f.write("Train End-to-End model from scratch with MyDigits smartphone/PC recordings included.\n\n")

        f.write("## Dataset\n\n")
        f.write("- **Zenke HD-Audio:** FLAC digits (0-9)\n")
        f.write("- **Noise samples:** _silence_ class\n")
        f.write("- **Speech Commands:** _unknown_ class\n")
        f.write("- **MyDigits Android:** Smartphone recordings (0-9)\n")
        f.write("- **MyDigits PC:** PC recordings (0-9)\n")
        f.write("- **Split:** Hybrid (speaker-based for digits, sample-based for special classes)\n\n")

        f.write("## Training Configuration\n\n")
        f.write("```\n")
        f.write(f"Model: End-to-End (Raw-Waveform -> MFCC -> CNN)\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Seed: {SEED}\n")
        f.write("```\n\n")

        f.write("## Results\n\n")
        f.write(f"### Test Accuracy: {test_acc*100:.2f}%\n\n")

        f.write("### Training Progress\n\n")
        f.write("| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |\n")
        f.write("|-------|-----------|---------|------------|----------|\n")
        for i in range(len(history.history['accuracy'])):
            f.write(f"| {i+1:2d} | {history.history['accuracy'][i]:.4f} | "
                   f"{history.history['val_accuracy'][i]:.4f} | "
                   f"{history.history['loss'][i]:.4f} | "
                   f"{history.history['val_loss'][i]:.4f} |\n")

        f.write("\n### Best Epoch\n\n")
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        f.write(f"- **Epoch:** {best_epoch}\n")
        f.write(f"- **Val Accuracy:** {best_val_acc*100:.2f}%\n\n")

        f.write("### Model Performance\n\n")
        f.write(f"Final Test Accuracy: {test_acc*100:.2f}%\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `training/best_model/` - Best model (SavedModel format)\n")
        f.write("- `training/saved_model/` - Exported SavedModel (for TFLite conversion)\n")
        f.write("- `training/confusion_matrix.png` - Full 12-class confusion matrix\n")
        f.write("- `training/special_confusion_matrix.png` - _silence_/_unknown_ confusion\n")
        f.write("- `training/training_curves.png` - Accuracy/loss curves\n")
        f.write("- `training/classification_report.txt` - Per-class metrics\n")
        f.write("- `training/training_log.csv` - Full training history\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Run `eval_mydigits.py` to evaluate on MyDigits recordings\n")
        f.write("2. Export to TFLite: `digits_rawwave_12cls.tflite`\n")
        f.write("3. Test on Android device with real smartphone audio\n")

    print(f"[OK] Saved report: {report_path}")


def main():
    """Main training pipeline - trains End-to-End model from scratch"""
    print_section("End-to-End Training with MyDigits")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader, train_gen, val_gen, test_gen, steps_per_epoch, val_steps, test_steps, test_infos = load_data()

    # Build fresh End-to-End model (no Phase 4C dependency)
    model = build_model(loader)

    # Train model
    history = train_model(model, train_gen, val_gen, steps_per_epoch, val_steps)

    # Evaluate model
    test_acc, cm = evaluate_model(model, test_gen, test_steps, loader, test_infos)

    # Plot training curves
    plot_training_curves(history)

    # Generate report
    generate_report(history, test_acc, cm, loader)

    # Save as SavedModel format for compatibility
    saved_model_dir = TRAINING_OUTPUT_DIR / "saved_model"
    print(f"\nSaving model in SavedModel format: {saved_model_dir}")
    model.save(saved_model_dir, save_format='tf')
    print(f"[OK] SavedModel saved to: {saved_model_dir}")

    print_section("Training Complete")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Model saved: {TRAINING_OUTPUT_DIR / 'best_model'}")
    print(f"Report saved: {TRAINING_OUTPUT_DIR.parent / 'TRAINING_REPORT.md'}")


if __name__ == "__main__":
    main()
