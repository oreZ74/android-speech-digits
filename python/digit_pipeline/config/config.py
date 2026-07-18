"""
Centralized Path Configuration for Digit Pipeline

All paths are relative to the repository root (REPO_ROOT).
This ensures portability across different machines and environments.

Usage:
    from digit_pipeline.config.config import DATA_DIR, MODELS_DIR, TrainingConfig, AugmentationConfig
"""

from pathlib import Path
import os

# =============================================================================
# REPOSITORY ROOT DETECTION
# =============================================================================
# Find the repository root by looking for characteristic files
def _find_repo_root() -> Path:
    """
    Locate the repository root by searching upward for marker files.
    
    Markers checked (in order):
    1. .git directory + data directory (main repo marker)
    2. android_app directory (unique to main repo)
    
    Falls back to 4 levels up from this file if no markers found.
    """
    current = Path(__file__).resolve()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for main repository markers (data + .git or android_app)
        if (parent / ".git").exists() and (parent / "data").exists():
            return parent
        if (parent / "android_app").exists() and (parent / "data").exists():
            return parent
    
    # Fallback: 4 levels up from this file
    # python/digit_pipeline/config/config.py -> REPO_ROOT
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _find_repo_root()

# =============================================================================
# DATA DIRECTORIES
# =============================================================================
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data subdirectories
HD_AUDIO_DIR = RAW_DATA_DIR / "hd_audio"
AUDIO_DEUTSCH_DIR = HD_AUDIO_DIR / "audio_deutsch"
MANUAL_RECORDINGS_DIR = RAW_DATA_DIR / "manual_recordings"
NOISE_DIR = RAW_DATA_DIR / "noise"
SPEECH_COMMANDS_DIR = RAW_DATA_DIR / "speech_commands"

# MyDigits manual recordings (for domain adaptation)
MYDIGITS_ANDROID_DIR = MANUAL_RECORDINGS_DIR / "mydigits_android"  # Smartphone recordings
MYDIGITS_PC_DIR = MANUAL_RECORDINGS_DIR / "mydigits_pc"            # PC recordings
MYDIGITS_DIR = MYDIGITS_ANDROID_DIR  # Default for training (compatibility)

# =============================================================================
# MODEL DIRECTORIES
# =============================================================================
MODELS_DIR = REPO_ROOT / "models"
TFLITE_DIR = MODELS_DIR / "tflite"
KERAS_DIR = MODELS_DIR / "keras"
LABELS_DIR = MODELS_DIR / "labels"

# Model file paths
TFLITE_MODEL_PATH = TFLITE_DIR / "digits_rawwave_12cls.tflite"
LABELS_PATH = LABELS_DIR / "labels.txt"

# =============================================================================
# PIPELINE DIRECTORIES
# =============================================================================
PIPELINE_ROOT = REPO_ROOT / "python" / "digit_pipeline"
SRC_DIR = PIPELINE_ROOT / "src"
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
CONFIG_DIR = PIPELINE_ROOT / "config"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"

# Training specific outputs
BEST_MODEL_DIR = TRAINING_OUTPUT_DIR / "best_model"
SAVED_MODEL_DIR = TRAINING_OUTPUT_DIR / "saved_model"
BEST_MODEL_KERAS = TRAINING_OUTPUT_DIR / "best_model.keras"
TRAINING_LOG = TRAINING_OUTPUT_DIR / "training_log.csv"

# =============================================================================
# ANDROID APP DIRECTORIES
# =============================================================================
ANDROID_APP_DIR = REPO_ROOT / "android_app"
ANDROID_ASSETS_DIR = ANDROID_APP_DIR / "app" / "src" / "main" / "assets"

# Android model location (copy target)
ANDROID_TFLITE_PATH = ANDROID_ASSETS_DIR / "digits_rawwave_12cls.tflite"
ANDROID_LABELS_PATH = ANDROID_ASSETS_DIR / "raw_digits_labels.txt"

# =============================================================================
# DOCUMENTATION
# =============================================================================
DOCS_DIR = REPO_ROOT / "docs"

# =============================================================================
# PIPELINE LABELS FILE (local copy)
# =============================================================================
PIPELINE_LABELS_PATH = CONFIG_DIR / "labels.txt"

# =============================================================================
# TRAINING CONFIGURATION (non-path settings)
# =============================================================================
class TrainingConfig:
    """Training hyperparameters and settings."""
    SAMPLE_RATE = 16000
    AUDIO_LENGTH_SECONDS = 1.0
    NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH_SECONDS)  # 16000
    
    NUM_CLASSES = 12  # 0-9, _silence_, _unknown_
    BATCH_SIZE = 32
    
    # Training from scratch (not fine-tuning!)
    EPOCHS = 25
    LEARNING_RATE = 1e-3  # Higher LR for training from scratch
    EARLY_STOPPING_PATIENCE = 7  # Patience for early stopping
    
    SEED = 42
    
    UNKNOWN_SAMPLES_PER_CLASS = 50


# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
class AugmentationConfig:
    """Data augmentation settings - moderate for from-scratch training."""
    
    # Time Shift: ±100ms (±1600 samples at 16kHz)
    TIME_SHIFT_RANGE = 1600
    
    # Gain Augmentation: ±20% (more moderate)
    GAIN_RANGE = (0.8, 1.2)
    
    # Noise Mixing: SNR between 10-30 dB (less noise)
    NOISE_VOLUME_RANGE = (0.0, 0.1)  # Max 10% noise amplitude
    
    # Noise-mix probability
    NOISE_MIX_PROBABILITY = 0.3


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, TFLITE_DIR, KERAS_DIR, LABELS_DIR,
        OUTPUTS_DIR, TRAINING_OUTPUT_DIR,
        DOCS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def print_paths():
    """Print all configured paths for debugging."""
    print("=" * 70)
    print("DIGIT PIPELINE - PATH CONFIGURATION")
    print("=" * 70)
    print(f"\nREPO_ROOT:           {REPO_ROOT}")
    print(f"\nDATA:")
    print(f"  DATA_DIR:          {DATA_DIR}")
    print(f"  RAW_DATA_DIR:      {RAW_DATA_DIR}")
    print(f"  HD_AUDIO_DIR:      {HD_AUDIO_DIR}")
    print(f"  AUDIO_DEUTSCH_DIR: {AUDIO_DEUTSCH_DIR}")
    print(f"  MYDIGITS_DIR:      {MYDIGITS_DIR}")
    print(f"  NOISE_DIR:         {NOISE_DIR}")
    print(f"  SPEECH_COMMANDS:   {SPEECH_COMMANDS_DIR}")
    print(f"\nMODELS:")
    print(f"  MODELS_DIR:        {MODELS_DIR}")
    print(f"  TFLITE_DIR:        {TFLITE_DIR}")
    print(f"  TFLITE_MODEL:      {TFLITE_MODEL_PATH}")
    print(f"\nOUTPUTS:")
    print(f"  OUTPUTS_DIR:       {OUTPUTS_DIR}")
    print(f"  TRAINING_OUTPUT:   {TRAINING_OUTPUT_DIR}")
    print(f"\nANDROID:")
    print(f"  ANDROID_APP_DIR:   {ANDROID_APP_DIR}")
    print(f"  ANDROID_ASSETS:    {ANDROID_ASSETS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_paths()
