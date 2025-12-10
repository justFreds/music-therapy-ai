import sys, os
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import keras
import numpy as np

# Model hyperparameters
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
SAVE_MODEL_PATH = "model_piano.h5"

def build_model(output_units, num_units, loss, learning_rate):
    """Build LSTM model for full piano generation"""
    # Input: (sequence_length,) of integers
    input_layer = keras.layers.Input(shape=(None,))
    
    # Embedding layer
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
    """Generator to yield batches of data"""
    num_samples = len(inputs)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            
            yield batch_inputs, batch_targets

def train():
    """Train piano model"""
    print("Generating training sequences...")
    inputs, targets, vocab_size = generate_training_sequences(SEQUENCE_LENGTH)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Total sequences: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Split into train/validation
    val_split = 0.2
    num_val = int(len(inputs) * val_split)
    
    train_inputs = inputs[:-num_val]
    train_targets = targets[:-num_val]
    val_inputs = inputs[-num_val:]
    val_targets = targets[-num_val:]
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Build model
    print("\n" + "="*50)
    print("TRAINING PIANO MODEL")
    print("="*50)
    
    model = build_model(vocab_size, NUM_UNITS, LOSS, LEARNING_RATE)
    
    # Create generators
    train_generator = create_data_generator(train_inputs, train_targets, BATCH_SIZE)
    val_generator = create_data_generator(val_inputs, val_targets, BATCH_SIZE)
    
    steps_per_epoch = len(train_inputs) // BATCH_SIZE
    validation_steps = len(val_inputs) // BATCH_SIZE
    
    # Train
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
    print(f"\nModel saved to {SAVE_MODEL_PATH}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history

if __name__ == '__main__':
    model, history = train()