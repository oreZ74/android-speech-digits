"""
Option A - End-to-End Model: Raw Waveform → MFCC → CNN → Classification

Architecture:
1. Input: [batch, 16000] raw waveform
2. MFCC Preprocessing: → [batch, 28, 13]
3. Reshape for CNN: → [batch, 28, 13, 1]
4. CNN Classifier: → [batch, 12]

NO TRAINING - Only model graph construction + shape validation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

# Import preprocessing block from same package
from digit_pipeline.src.preprocessing import create_preprocessing_block, EXPECTED_TIME_FRAMES, N_MFCC


def create_end_to_end_model(num_classes: int = 12) -> tf.keras.Model:
    """
    Creates end-to-end model: Raw Waveform → MFCC → CNN → Classification
    
    Architecture COPIED from legacy CnnModel.py with adaptations:
    - Input changed: (13, 32, 1) → (28, 13, 1)
    - Output changed: 10 classes → 12 classes (0-9, _silence_, _unknown_)
    - CNN structure: UNCHANGED (same filters, layers, dropout rates)
    
    Args:
        num_classes: Number of output classes (default: 12)
        
    Returns:
        Compiled Keras Model
    """
    # Step 1: Input layer - raw waveform
    waveform_input = tf.keras.Input(shape=(16000,), name='waveform_input')
    
    # Step 2: MFCC Preprocessing (from Phase 1)
    preprocessing_layer = create_preprocessing_block()
    mfcc_features = preprocessing_layer(waveform_input)  # Output: (batch, 28, 13)
    
    # Step 3: Reshape for CNN - add channel dimension
    # Legacy CNN expected: (13, 32, 1)
    # We have: (28, 13) → reshape to (28, 13, 1)
    cnn_input = tf.keras.layers.Reshape((EXPECTED_TIME_FRAMES, N_MFCC, 1), 
                                        name='reshape_for_cnn')(mfcc_features)
    
    # Step 4: CNN Architecture 
    # Input shape: (28, 13, 1) instead of legacy (13, 32, 1)
    # All other parameters UNCHANGED
    
    # Conv Block 1 - Initial feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      name='conv1_1')(cnn_input)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Conv Block 2 - Mid-level features
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                      name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                      name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Conv Block 3 - High-level features
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                      name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # Dense layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Dropout(0.5, name='dropout_dense1')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.4, name='dropout_dense2')(x)
    
    # Output layer - CHANGED from 10 to 12 classes
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=waveform_input, outputs=outputs, 
                          name='digits_rawwave')
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_info(model: tf.keras.Model) -> None:
    """
    Prints detailed model information including layer shapes.
    
    Args:
        model: Keras Model to analyze
    """
    print("\n" + "=" * 70)
    print("END-TO-END MODEL INFORMATION")
    print("=" * 70)
    
    # Model summary
    print("\n[MODEL SUMMARY]")
    model.summary()
    
    # Detailed tensor flow
    print("\n" + "-" * 70)
    print("TENSOR FLOW (Layer-by-Layer)")
    print("-" * 70)
    
    # Get intermediate layer outputs
    layer_names_of_interest = [
        'waveform_input',
        'mfcc_preprocessing',
        'reshape_for_cnn',
        'conv1_1', 'pool1',
        'conv2_1', 'pool2',
        'conv3', 'pool3',
        'flatten',
        'dense1',
        'dense2',
        'output'
    ]
    
    for layer in model.layers:
        if layer.name in layer_names_of_interest or \
           layer.name.startswith('conv') or \
           layer.name.startswith('pool') or \
           layer.name.startswith('dense') or \
           layer.name in ['waveform_input', 'mfcc_preprocessing', 
                         'reshape_for_cnn', 'flatten', 'output']:
            try:
                output_shape = layer.output_shape
                print(f"{layer.name:30s} → {str(output_shape):30s}")
            except:
                pass
    
    # Parameter count
    print("\n" + "-" * 70)
    print("PARAMETER COUNT")
    print("-" * 70)
    
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters:         {total_params:,}")
    
    # Input/Output info
    print("\n" + "-" * 70)
    print("INPUT/OUTPUT SPECIFICATION")
    print("-" * 70)
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Input dtype:  {model.input.dtype}")
    print(f"Output dtype: {model.output.dtype}")


def test_forward_pass(model: tf.keras.Model, batch_size: int = 2) -> None:
    """
    Tests forward pass with dummy input.
    
    Validates:
    - No shape errors
    - Output shape is (batch, 12)
    - Softmax sums to 1.0
    
    Args:
        model: Keras Model to test
        batch_size: Batch size for test (default: 2)
    """
    print("\n" + "=" * 70)
    print("FORWARD PASS SANITY CHECK")
    print("=" * 70)
    
    # Generate dummy input: random waveforms in [-1, 1]
    print(f"\n[STEP 1] Generating dummy input...")
    dummy_input = np.random.uniform(-1.0, 1.0, size=(batch_size, 16000)).astype(np.float32)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    # Forward pass
    print(f"\n[STEP 2] Running forward pass...")
    try:
        predictions = model.predict(dummy_input, verbose=0)
        print(f"  [OK] Forward pass successful")
    except Exception as e:
        print(f"  [X] Forward pass FAILED: {e}")
        raise
    
    # Validate output shape
    print(f"\n[STEP 3] Validating output shape...")
    expected_shape = (batch_size, 12)
    if predictions.shape == expected_shape:
        print(f"  [OK] Output shape correct: {predictions.shape}")
    else:
        print(f"  [X] Output shape MISMATCH!")
        print(f"    Expected: {expected_shape}")
        print(f"    Got:      {predictions.shape}")
        raise ValueError("Output shape mismatch")
    
    # Validate softmax summation
    print(f"\n[STEP 4] Validating softmax probabilities...")
    for i in range(batch_size):
        prob_sum = np.sum(predictions[i])
        max_prob = np.max(predictions[i])
        pred_class = np.argmax(predictions[i])
        
        print(f"  Sample {i+1}:")
        print(f"    Probability sum: {prob_sum:.6f} (should be ~1.0)")
        print(f"    Predicted class: {pred_class}")
        print(f"    Max probability: {max_prob:.4f}")
        
        # Check if sum is approximately 1.0 (tolerance: 1e-5)
        if not np.isclose(prob_sum, 1.0, atol=1e-5):
            print(f"    [X] WARNING: Softmax sum not close to 1.0!")
        else:
            print(f"    [OK] Softmax validation passed")
    
    print("\n" + "=" * 70)
    print("[OK] FORWARD PASS SANITY CHECK PASSED")
    print("=" * 70)


def run_model_construction_test():
    """
    Complete test workflow:
    1. Create model
    2. Print model info
    3. Test forward pass
    """
    print("\n" + "=" * 70)
    print("OPTION A - PHASE 2: END-TO-END MODEL CONSTRUCTION")
    print("=" * 70)
    print("Task: Build model graph, validate shapes, NO TRAINING")
    print("=" * 70)
    
    # Step 1: Create model
    print("\n[STEP 1] Creating end-to-end model...")
    model = create_end_to_end_model(num_classes=12)
    print("  [OK] Model created successfully")
    
    # Step 2: Print model info
    print("\n[STEP 2] Analyzing model architecture...")
    print_model_info(model)
    
    # Step 3: Test forward pass
    print("\n[STEP 3] Testing forward pass...")
    test_forward_pass(model, batch_size=2)
    
    print("\n" + "=" * 70)
    print("[OK] MODEL CONSTRUCTION TEST COMPLETE")
    print("=" * 70)
    print("\nNext: TFLite conversion test")
    
    return model


if __name__ == "__main__":
    # Run construction test
    model = run_model_construction_test()

