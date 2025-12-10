import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
from vae_model import VAE
import keras
import numpy as np
import json

# VAE Hyperparameters - REDUCED FOR STABILITY
LATENT_DIM = 64  # Reduced from 256
INTERMEDIATE_DIM = 256  # Reduced from 512
LEARNING_RATE = 0.0001  # Much lower learning rate
EPOCHS = 100
BATCH_SIZE = 32  # Smaller batch size
SAVE_VAE_DIR = "vae_piano_model"
SAVE_CONFIG_PATH = "vae_config.json"

# KL annealing parameters
KL_ANNEAL_EPOCHS = 30
KL_START_WEIGHT = 0.0
KL_END_WEIGHT = 0.05  # Very small to prevent issues

def prepare_vae_data(inputs, sequence_length):
    """Prepare data for VAE training"""
    num_samples = inputs.shape[0]
    current_length = inputs.shape[1]
    
    if current_length > sequence_length:
        inputs = inputs[:, :sequence_length]
    elif current_length < sequence_length:
        padding = np.zeros((num_samples, sequence_length - current_length), dtype=int)
        inputs = np.concatenate([inputs, padding], axis=1)
    
    return inputs

class KLAnnealingCallback(keras.callbacks.Callback):
    """Callback to gradually increase KL weight during training"""
    
    def __init__(self, vae, anneal_epochs, start_weight, end_weight):
        super().__init__()
        self.vae = vae
        self.anneal_epochs = anneal_epochs
        self.start_weight = start_weight
        self.end_weight = end_weight
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.anneal_epochs:
            weight = self.start_weight + (self.end_weight - self.start_weight) * (epoch / self.anneal_epochs)
        else:
            weight = self.end_weight
        
        self.vae.set_kl_weight(weight)
        print(f"\nKL weight: {weight:.6f}")

def train_vae():
    """Train VAE for piano generation"""
    print("\n" + "="*50)
    print("TRAINING PIANO VAE WITH KL ANNEALING")
    print("="*50)
    
    # Generate training sequences
    print("Generating training sequences...")
    inputs, targets, vocab_size = generate_training_sequences(SEQUENCE_LENGTH)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Total sequences: {len(inputs)}")
    
    # Prepare data
    vae_inputs = prepare_vae_data(inputs, SEQUENCE_LENGTH)
    print(f"VAE input shape: {vae_inputs.shape}")
    
    # Split into train/validation
    val_split = 0.2
    num_val = int(len(vae_inputs) * val_split)
    
    train_data = vae_inputs[:-num_val]
    val_data = vae_inputs[-num_val:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Build VAE
    print("\nBuilding VAE...")
    vae = VAE(
        input_dim=vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        latent_dim=LATENT_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        kl_weight=KL_START_WEIGHT
    )
    
    # Compile with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0  # Gradient clipping to prevent explosion
    )
    vae.compile(optimizer=optimizer)
    
    print("\nVAE Architecture:")
    print("Encoder:")
    vae.encoder.summary()
    print("\nDecoder:")
    vae.decoder.summary()
    
    # Callbacks
    kl_callback = KLAnnealingCallback(vae, KL_ANNEAL_EPOCHS, KL_START_WEIGHT, KL_END_WEIGHT)
    
    # Fixed early stopping - use correct metric name
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_total_loss',  # Changed from 'val_loss'
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce LR on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_total_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # NaN termination
    nan_terminate = keras.callbacks.TerminateOnNaN()
    
    # Train
    print("\nStarting training...")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"KL annealing: {KL_START_WEIGHT} -> {KL_END_WEIGHT} over {KL_ANNEAL_EPOCHS} epochs")
    
    history = vae.fit(
        train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_data),
        callbacks=[kl_callback, early_stop, reduce_lr, nan_terminate],
        verbose=1
    )
    
    # Save model
    print(f"\nSaving VAE model...")
    vae.save(SAVE_VAE_DIR, save_format='tf')
    print(f"VAE saved to {SAVE_VAE_DIR}")
    
    # Save configuration
    config = {
        'vocab_size': vocab_size,
        'sequence_length': SEQUENCE_LENGTH,
        'latent_dim': LATENT_DIM,
        'intermediate_dim': INTERMEDIATE_DIM,
        'kl_weight': KL_END_WEIGHT
    }
    with open(SAVE_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"VAE config saved to {SAVE_CONFIG_PATH}")
    
    print("\n" + "="*50)
    print("VAE TRAINING COMPLETE!")
    print("="*50)
    if len(history.history['loss']) > 0:
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
        print(f"Final reconstruction_loss: {history.history['reconstruction_loss'][-1]:.4f}")
        print(f"Final kl_loss: {history.history['kl_loss'][-1]:.4f}")
    
    return vae, history

if __name__ == '__main__':
    vae, history = train_vae()