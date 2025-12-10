from maestro_preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
import keras
import json
import numpy as np

# Model hyperparameters
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"  # This is key - no one-hot encoding needed!
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32  # Reduced batch size
SAVE_MODEL_PATH = "model_combined.h5"

def build_model(output_units, num_units, loss, learning_rate):
    """Build LSTM model - adjusted input shape for sparse data"""
    # Input is now (sequence_length,) of integers, not one-hot encoded
    input_layer = keras.layers.Input(shape=(None,))
    
    # Embedding layer to convert integers to dense vectors
    x = keras.layers.Embedding(input_dim=output_units, output_dim=128)(input_layer)
    
    # LSTM layers
    x = keras.layers.LSTM(num_units[0])(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Output
    output = keras.layers.Dense(output_units, activation='softmax')(x)
    
    model = keras.Model(input_layer, output)
    
    # Compile
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

def create_data_generator(inputs, targets, batch_size):
    """
    Generator to yield batches of data
    Avoids loading entire dataset into memory at once
    """
    num_samples = len(inputs)
    indices = np.arange(num_samples)
    
    while True:
        # Shuffle at the start of each epoch
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            
            yield batch_inputs, batch_targets

def train_single_model():
    """Train single combined model with memory-efficient approach"""
    print("Generating training sequences...")
    melody_inputs, melody_targets, harmony_inputs, harmony_targets, vocab_size = \
        generate_training_sequences(SEQUENCE_LENGTH)
    
    # Combine melody and harmony data
    combined_inputs = np.concatenate([melody_inputs, harmony_inputs], axis=0)
    combined_targets = np.concatenate([melody_targets, harmony_targets], axis=0)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Combined sequences: {len(combined_inputs)}")
    print(f"Input shape: {combined_inputs.shape}")
    print(f"Target shape: {combined_targets.shape}")
    
    # Split into train/validation
    val_split = 0.2
    num_val = int(len(combined_inputs) * val_split)
    
    train_inputs = combined_inputs[:-num_val]
    train_targets = combined_targets[:-num_val]
    val_inputs = combined_inputs[-num_val:]
    val_targets = combined_targets[-num_val:]
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Build model
    print("\n" + "="*50)
    print("TRAINING COMBINED MODEL")
    print("="*50)
    
    model = build_model(vocab_size, NUM_UNITS, LOSS, LEARNING_RATE)
    
    # Create data generators
    train_generator = create_data_generator(train_inputs, train_targets, BATCH_SIZE)
    val_generator = create_data_generator(val_inputs, val_targets, BATCH_SIZE)
    
    steps_per_epoch = len(train_inputs) // BATCH_SIZE
    validation_steps = len(val_inputs) // BATCH_SIZE
    
    # Train with generator (memory efficient!)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1
    )
    
    # Save
    model.save(SAVE_MODEL_PATH)
    print(f"\nCombined model saved to {SAVE_MODEL_PATH}")
    
    return model, history

if __name__ == '__main__':
    model, history = train_single_model()