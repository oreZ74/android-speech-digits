"""
Digit Pipeline Package

A complete training pipeline for German digit speech recognition.

Modules:
- src: Core source modules (data_loader, preprocessing, model_definition)
- scripts: Executable training and evaluation scripts
- config: Path configuration and settings
- outputs: Training outputs and saved models

Usage:
    from digit_pipeline.src import GermanDigitRawWaveformLoader, create_end_to_end_model
    from digit_pipeline.config.config import DATA_DIR, MODELS_DIR, TrainingConfig
"""

__version__ = "4.0.0"

