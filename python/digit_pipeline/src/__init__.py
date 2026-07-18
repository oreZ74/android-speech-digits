"""
Digit Pipeline Source Module

Core components for the German digit recognition training pipeline.

Components:
- data_loader: Dataset loading and augmentation
- preprocessing: Audio preprocessing (MFCC in-graph)
- model_definition: End-to-end CNN model architecture
"""

from .data_loader import GermanDigitRawWaveformLoader
from .preprocessing import create_preprocessing_block
from .model_definition import create_end_to_end_model

__all__ = [
    'GermanDigitRawWaveformLoader',
    'create_preprocessing_block', 
    'create_end_to_end_model'
]
