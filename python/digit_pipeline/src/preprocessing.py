"""
Option A - Preprocessing Block: Raw Waveform → MFCC in TensorFlow Graph

This module implements MFCC feature extraction entirely within TensorFlow/Keras,
allowing the preprocessing to be embedded in the model graph for deployment.

Fixed Parameters:
- sample_rate = 16000 Hz
- clip_duration = 1.0s => 16000 samples
- n_fft = 2048
- hop_length = 512
- n_mels = 128
- n_mfcc = 13
- fmax = 8000 Hz
- window = Hann
- power_spectrogram = magnitude^2

Reference:
- referenzen/input_data.py (TF Speech Commands implementation)
- Legacy librosa parameters for validation
"""

import tensorflow as tf
import numpy as np

from digit_pipeline.config.config import TrainingConfig


# Fixed parameters (NEVER change these!)
SAMPLE_RATE = TrainingConfig.SAMPLE_RATE  # 16000
CLIP_DURATION_SEC = TrainingConfig.AUDIO_LENGTH_SECONDS  # 1.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13
FMAX = 8000.0

# Derived parameters
DESIRED_SAMPLES = TrainingConfig.NUM_SAMPLES  # 16000
WINDOW_SIZE_SAMPLES = N_FFT  # 2048
WINDOW_STRIDE_SAMPLES = HOP_LENGTH  # 512

# Calculate expected output shape
# Number of frames = (n_samples - n_fft) // hop_length + 1
# For 16000 samples: (16000 - 2048) // 512 + 1 = 28 frames
EXPECTED_TIME_FRAMES = (DESIRED_SAMPLES - N_FFT) // HOP_LENGTH + 1  # 28


def create_preprocessing_block(input_shape=(16000,), name="mfcc_preprocessing"):
    """
    Creates a Keras preprocessing block that converts raw waveform to MFCC features.

    This function builds a TensorFlow graph that performs:
    1. STFT (Short-Time Fourier Transform)
    2. Power Spectrogram (magnitude squared)
    3. Mel-scale conversion
    4. Log compression
    5. MFCC extraction

    Args:
        input_shape: Shape of input waveform, default (16000,) for 1-second @ 16kHz
        name: Name for the preprocessing layer

    Returns:
        tf.keras.layers.Layer: Preprocessing layer that outputs MFCC features

    Output Shape:
        (batch_size, 28, 13) where:
        - 28 = time frames
        - 13 = MFCC coefficients

    Notes:
        - Uses TensorFlow Signal API (tf.signal.*)
        - Compatible with TFLite conversion (requires SELECT_TF_OPS)
        - Parameters match legacy librosa implementation
    """

    class MFCCPreprocessing(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(MFCCPreprocessing, self).__init__(name=name, **kwargs)

            # Store parameters as layer attributes
            self.sample_rate = SAMPLE_RATE
            self.n_fft = N_FFT
            self.hop_length = HOP_LENGTH
            self.n_mels = N_MELS
            self.n_mfcc = N_MFCC
            self.fmax = FMAX

            # Pre-compute Mel weight matrix (can be done once)
            # This matrix converts linear spectrogram bins to Mel-scale bins
            self.mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.n_mels,
                num_spectrogram_bins=self.n_fft // 2 + 1,  # 1025 bins
                sample_rate=self.sample_rate,
                lower_edge_hertz=0.0,
                upper_edge_hertz=self.fmax
            )

        def call(self, waveform):
            """
            Forward pass: Waveform → MFCC

            Args:
                waveform: Tensor of shape (batch_size, 16000) or (16000,)

            Returns:
                mfcc: Tensor of shape (batch_size, 28, 13)
            """
            # Ensure input is at least 2D
            if len(waveform.shape) == 1:
                waveform = tf.expand_dims(waveform, 0)

            # Step 1: STFT (Short-Time Fourier Transform)
            # Computes the complex-valued STFT of the signal
            stft = tf.signal.stft(
                waveform,
                frame_length=self.n_fft,
                frame_step=self.hop_length,
                fft_length=self.n_fft,
                window_fn=tf.signal.hann_window,
                pad_end=False  # Don't pad, match librosa behavior
            )
            # Output shape: (batch, time_frames, n_fft//2 + 1) = (batch, 28, 1025)

            # Step 2: Power Spectrogram (magnitude squared)
            # Convert complex STFT to real-valued power spectrogram
            magnitude = tf.abs(stft)
            power_spectrogram = tf.square(magnitude)
            # Output shape: (batch, 28, 1025)

            # Step 3: Mel-scale conversion
            # Apply Mel weight matrix to convert linear frequency bins to Mel-scale
            mel_spectrogram = tf.tensordot(
                power_spectrogram,
                self.mel_weight_matrix,
                axes=1  # Multiply along frequency axis
            )
            # Output shape: (batch, 28, 128)

            # Step 4: Log compression
            # Apply log to compress dynamic range (add small epsilon to avoid log(0))
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
            # Output shape: (batch, 28, 128)

            # Step 5: MFCC extraction
            # Compute DCT (Discrete Cosine Transform) to get MFCC coefficients
            mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            # Output shape: (batch, 28, 128) - full DCT coefficients

            # Step 6: Keep only first N_MFCC coefficients (13)
            mfcc = mfcc[..., :self.n_mfcc]
            # Final output shape: (batch, 28, 13)

            return mfcc

        def get_config(self):
            """Required for model serialization"""
            config = super(MFCCPreprocessing, self).get_config()
            config.update({
                'sample_rate': self.sample_rate,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'n_mfcc': self.n_mfcc,
                'fmax': self.fmax
            })
            return config

    return MFCCPreprocessing()


def create_preprocessing_model(input_shape=(16000,)):
    """
    Creates a standalone Keras model for preprocessing (for testing purposes).

    Args:
        input_shape: Shape of input waveform

    Returns:
        tf.keras.Model: Model that outputs MFCC features
    """
    inputs = tf.keras.Input(shape=input_shape, name='waveform_input')
    mfcc_layer = create_preprocessing_block(input_shape=input_shape)
    outputs = mfcc_layer(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mfcc_preprocessing_model')
    return model


# Utility functions for testing and validation

def extract_mfcc_tf(waveform, return_numpy=True):
    """
    Convenience function to extract MFCC from a single waveform using TensorFlow.

    Args:
        waveform: 1D numpy array or tensor of shape (16000,)
        return_numpy: If True, return numpy array; if False, return tensor

    Returns:
        mfcc: MFCC features of shape (28, 13)
    """
    # Convert to tensor if needed
    if isinstance(waveform, np.ndarray):
        waveform = tf.constant(waveform, dtype=tf.float32)

    # Ensure correct shape
    if len(waveform.shape) == 1:
        waveform = tf.expand_dims(waveform, 0)  # Add batch dimension

    # Create preprocessing layer and apply
    preprocessing = create_preprocessing_block()
    mfcc = preprocessing(waveform)

    # Remove batch dimension
    mfcc = tf.squeeze(mfcc, axis=0)

    if return_numpy:
        return mfcc.numpy()
    return mfcc


def get_output_shape():
    """
    Returns the expected output shape of the preprocessing block.

    Returns:
        tuple: (time_frames, n_mfcc) = (28, 13)
    """
    return (EXPECTED_TIME_FRAMES, N_MFCC)


def print_preprocessing_info():
    """Prints configuration information about the preprocessing block."""
    print("=" * 60)
    print("OPTION A - MFCC Preprocessing Block Configuration")
    print("=" * 60)
    print(f"Sample Rate:        {SAMPLE_RATE} Hz")
    print(f"Clip Duration:      {CLIP_DURATION_SEC} s => {DESIRED_SAMPLES} samples")
    print(f"FFT Size (n_fft):   {N_FFT}")
    print(f"Hop Length:         {HOP_LENGTH}")
    print(f"Mel Bins:           {N_MELS}")
    print(f"MFCC Coefficients:  {N_MFCC}")
    print(f"Max Frequency:      {FMAX} Hz")
    print(f"Window Function:    Hann")
    print(f"Power Spectrogram:  magnitude^2")
    print("-" * 60)
    print(f"Input Shape:        (batch_size, {DESIRED_SAMPLES})")
    print(f"Output Shape:       (batch_size, {EXPECTED_TIME_FRAMES}, {N_MFCC})")
    print("=" * 60)


if __name__ == "__main__":
    # Quick test of the preprocessing block
    print_preprocessing_info()

    print("\n[TEST] Creating preprocessing model...")
    model = create_preprocessing_model()
    model.summary()

    print("\n[TEST] Testing with random waveform...")
    test_waveform = np.random.randn(16000).astype(np.float32)
    mfcc_features = extract_mfcc_tf(test_waveform)

    print(f"Input shape:  {test_waveform.shape}")
    print(f"Output shape: {mfcc_features.shape}")
    print(f"MFCC range:   [{mfcc_features.min():.2f}, {mfcc_features.max():.2f}]")
    print(f"MFCC mean:    {mfcc_features.mean():.2f}")
    print(f"MFCC std:     {mfcc_features.std():.2f}")

    print("\n[OK] Preprocessing block created successfully!")
