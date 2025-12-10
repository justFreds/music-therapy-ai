from maestro_preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
from maestro_vae_model import VAE
import keras
import numpy as np
import json

# VAE Hyperparameters
LATENT_DIM = 128
INTERMEDIATE_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 64
SAVE_MELODY_VAE_PATH = "vae_melody.keras"
SAVE_HARMONY_VAE_PATH = "vae_harmony.keras"

def prepare_vae_data(inputs, vocab_size):
    """
    Prepare data for VAE training
    VAE expects sequences of fixed length with one-hot encoding
    """
    # One-hot encode
    inputs_encoded = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    
    # Pad/truncate to SEQUENCE_LENGTH
    num_samples = inputs_encoded.shape[0]
    seq_length = inputs_encoded.shape[1]
    
    if seq_length > SEQUENCE_LENGTH:
        inputs_encoded = inputs_encoded[:, :SEQUENCE_LENGTH, :]
    elif seq_length < SEQUENCE_LENGTH:
        padding = np.zeros((num_samples, SEQUENCE_LENGTH - seq_length, vocab_size))
        inputs_encoded = np.concatenate([inputs_encoded, padding], axis=1)
    
    return inputs_encoded

def train_melody_vae(melody_inputs, vocab_size, 
                     latent_dim=LATENT_DIM, 
                     intermediate_dim=INTERMEDIATE_DIM,
                     learning_rate=LEARNING_RATE):
    """Train VAE for melody generation"""
    print("\n" + "="*50)
    print("TRAINING MELODY VAE")
    print("="*50)
    
    # Prepare data
    melody_data = prepare_vae_data(melody_inputs, vocab_size)
    print(f"Melody data shape: {melody_data.shape}")
    
    # Build VAE
    vae = VAE(
        input_dim=vocab_size,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim
    )
    
    # Compile
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Train
    history = vae.fit(
        melody_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    # Save
    vae.save(SAVE_MELODY_VAE_PATH)
    print(f"\nMelody VAE saved to {SAVE_MELODY_VAE_PATH}")
    
    return vae, history

def train_harmony_vae(harmony_inputs, vocab_size,
                      latent_dim=LATENT_DIM,
                      intermediate_dim=INTERMEDIATE_DIM,
                      learning_rate=LEARNING_RATE):
    """Train VAE for harmony generation"""
    print("\n" + "="*50)
    print("TRAINING HARMONY VAE")
    print("="*50)
    
    # Prepare data
    harmony_data = prepare_vae_data(harmony_inputs, vocab_size)
    print(f"Harmony data shape: {harmony_data.shape}")
    
    # Build VAE
    vae = VAE(
        input_dim=vocab_size,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim
    )
    
    # Compile
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Train
    history = vae.fit(
        harmony_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    # Save
    vae.save(SAVE_HARMONY_VAE_PATH)
    print(f"\nHarmony VAE saved to {SAVE_HARMONY_VAE_PATH}")
    
    return vae, history

def train_both_vaes():
    """Train both melody and harmony VAEs"""
    # Generate training sequences
    print("Generating training sequences...")
    melody_inputs, melody_targets, harmony_inputs, harmony_targets, vocab_size = \
        generate_training_sequences(SEQUENCE_LENGTH)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Melody sequences: {len(melody_inputs)}")
    print(f"Harmony sequences: {len(harmony_inputs)}")
    
    # Train melody VAE
    melody_vae, melody_history = train_melody_vae(
        melody_inputs,
        vocab_size,
        latent_dim=LATENT_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        learning_rate=LEARNING_RATE
    )
    
    # Train harmony VAE
    harmony_vae, harmony_history = train_harmony_vae(
        harmony_inputs,
        vocab_size,
        latent_dim=LATENT_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        learning_rate=LEARNING_RATE
    )
    
    print("\n" + "="*50)
    print("VAE TRAINING COMPLETE!")
    print("="*50)
    print(f"Melody VAE final loss: {melody_history.history['loss'][-1]:.4f}")
    print(f"Harmony VAE final loss: {harmony_history.history['loss'][-1]:.4f}")
    
    return melody_vae, harmony_vae, melody_history, harmony_history

if __name__ == '__main__':
    melody_vae, harmony_vae, melody_history, harmony_history = train_both_vaes()