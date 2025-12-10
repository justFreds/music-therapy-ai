import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from vae_model import VAE, Sampling
import keras
import numpy as np
import json
import music21 as m21

class VAEPianoGenerator:
    def __init__(self, vae_path='vae_piano_model', config_path='vae_config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loading VAE with config: {config}")
        
        # Reconstruct the VAE model manually
        self.vae = VAE(
            input_dim=config['vocab_size'],
            sequence_length=config['sequence_length'],
            latent_dim=config['latent_dim'],
            intermediate_dim=config['intermediate_dim']
        )
        
        # Build the model by calling it once
        dummy_input = np.zeros((1, config['sequence_length']), dtype=np.int32)
        _ = self.vae(dummy_input)
        
        # Load weights from saved model
        self.vae.load_weights(os.path.join(vae_path, 'variables', 'variables'))
        
        # Load mappings
        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        # Create reverse mapping (int -> symbol)
        self._reverse_mappings = {v: k for k, v in self._mappings.items()}
        
        print(f"Loaded VAE model from {vae_path}")
        print(f"Vocabulary size: {len(self._mappings)}")
        print(f"Latent dimension: {config['latent_dim']}")
    
    def generate_from_latent(self, latent_vector=None, temperature=1.0):
        """
        Generate piano music from latent space
        
        :param latent_vector: Optional latent vector. If None, samples from N(0,1)
        :param temperature: Temperature for sampling (higher = more random)
        :return: Generated sequence as list of symbols
        """
        # Sample from latent space if no vector provided
        if latent_vector is None:
            latent_vector = np.random.normal(0, 1, size=(1, self.vae.latent_dim))
        
        # Generate sequence
        reconstruction = self.vae.decoder.predict(latent_vector, verbose=0)[0]
        
        # Convert probabilities to symbols
        generated_sequence = []
        for timestep_probs in reconstruction:
            # Apply temperature
            timestep_probs = np.log(timestep_probs + 1e-10) / temperature
            timestep_probs = np.exp(timestep_probs) / np.sum(np.exp(timestep_probs))
            
            # Sample from distribution
            symbol_int = np.random.choice(len(timestep_probs), p=timestep_probs)
            symbol = self._reverse_mappings.get(symbol_int, '/')
            
            # Stop at delimiter
            if symbol == '/':
                break
            
            generated_sequence.append(symbol)
        
        return generated_sequence
    
    def generate_music(self, latent_vector=None, temperature=1.0):
        """Generate piano music"""
        return self.generate_from_latent(latent_vector, temperature)
    
    def interpolate(self, latent_start=None, latent_end=None, num_steps=5):
        """
        Interpolate between two latent vectors
        Creates smooth musical transitions
        """
        # Generate random points if not provided
        if latent_start is None:
            latent_start = np.random.normal(0, 1, size=(1, self.vae.latent_dim))
        if latent_end is None:
            latent_end = np.random.normal(0, 1, size=(1, self.vae.latent_dim))
        
        # Generate interpolated sequences
        sequences = []
        for alpha in np.linspace(0, 1, num_steps):
            latent_interp = (1 - alpha) * latent_start + alpha * latent_end
            sequence = self.generate_music(latent_interp)
            sequences.append(sequence)
        
        return sequences
    
    def _parse_sequence_to_stream(self, sequence, step_duration=0.25):
        """Parse sequence into music21 stream"""
        stream = m21.stream.Stream()
        
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(sequence):
            if symbol != '_' or i + 1 == len(sequence):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    
                    # Handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    
                    # Handle chord [60,64,67]
                    elif start_symbol.startswith('[') and start_symbol.endswith(']'):
                        try:
                            chord_str = start_symbol[1:-1]
                            pitches = [int(p) for p in chord_str.split(',')]
                            m21_event = m21.chord.Chord(pitches, quarterLength=quarter_length_duration)
                        except:
                            # Skip invalid chords
                            step_counter = 1
                            start_symbol = symbol
                            continue
                    
                    # Handle single note
                    else:
                        try:
                            m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        except ValueError:
                            # Skip invalid symbols
                            step_counter = 1
                            start_symbol = symbol
                            continue
                    
                    stream.append(m21_event)
                    step_counter = 1
                
                start_symbol = symbol
            else:
                step_counter += 1
        
        return stream
    
    def save_music(self, music, step_duration=0.25, format="midi", file_name='vae_piano.mid'):
        """Save generated piano music as MIDI"""
        stream = self._parse_sequence_to_stream(music, step_duration)
        stream.write(format, file_name)
        print(f"Saved music to {file_name}")
        return stream


if __name__ == '__main__':
    vg = VAEPianoGenerator(vae_path='vae_piano_model', config_path='vae_config.json')
    
    print("=" * 50)
    print("GENERATING PIANO MUSIC WITH VAE")
    print("=" * 50)
    
    # Generate random samples
    print("\n1. Generating 3 random samples...")
    for i in range(3):
        music = vg.generate_music(temperature=0.9)
        print(f"  Sample {i+1} length: {len(music)}")
        vg.save_music(music, file_name=f'vae_sample_{i+1}.mid')
    
    # Generate with different temperatures
    print("\n2. Generating with different temperatures...")
    for temp in [0.5, 0.8, 1.0, 1.2]:
        music = vg.generate_music(temperature=temp)
        print(f"  Temperature {temp}: {len(music)} symbols")
        vg.save_music(music, file_name=f'vae_temp_{temp}.mid')
    
    # Interpolation demo
    print("\n3. Generating interpolated sequence...")
    latent_start = np.random.normal(0, 1, size=(1, vg.vae.latent_dim))
    latent_end = np.random.normal(0, 1, size=(1, vg.vae.latent_dim))
    interpolated = vg.interpolate(latent_start, latent_end, num_steps=5)
    
    for i, music in enumerate(interpolated):
        print(f"  Interpolation step {i+1}: {len(music)} symbols")
        vg.save_music(music, file_name=f'vae_interp_{i+1}.mid')
    
    print("\n" + "=" * 50)
    print("VAE GENERATION COMPLETE!")
    print("=" * 50)
    print("Files created:")
    print("  - vae_sample_1-3.mid (random samples)")
    print("  - vae_temp_0.5-1.2.mid (temperature variations)")
    print("  - vae_interp_1-5.mid (interpolation)")