import keras
import numpy as np
import json
import music21 as m21
from maestro_preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class VAEMusicGenerator:
    def __init__(self, melody_vae_path='vae_melody.keras', harmony_vae_path='vae_harmony.keras'):
        self.melody_vae = keras.models.load_model(melody_vae_path, custom_objects={'VAE': VAE, 'Sampling': Sampling})
        self.harmony_vae = keras.models.load_model(harmony_vae_path, custom_objects={'VAE': VAE, 'Sampling': Sampling})
        
        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        # Create reverse mapping (int -> symbol)
        self._reverse_mappings = {v: k for k, v in self._mappings.items()}
    
    def generate_from_latent(self, vae_model, latent_vector=None, temperature=1.0):
        """
        Generate music from latent space
        
        :param vae_model: VAE model to use
        :param latent_vector: Optional latent vector. If None, samples from normal distribution
        :param temperature: Temperature for sampling (higher = more random)
        :return: Generated sequence as list of symbols
        """
        # Sample from latent space if no vector provided
        if latent_vector is None:
            latent_vector = np.random.normal(0, 1, size=(1, vae_model.latent_dim))
        
        # Generate sequence
        reconstruction = vae_model.decoder.predict(latent_vector, verbose=0)[0]
        
        # Convert probabilities to symbols
        generated_sequence = []
        for timestep_probs in reconstruction:
            # Apply temperature
            timestep_probs = np.log(timestep_probs + 1e-10) / temperature
            timestep_probs = np.exp(timestep_probs) / np.sum(np.exp(timestep_probs))
            
            # Sample from distribution
            symbol_int = np.random.choice(len(timestep_probs), p=timestep_probs)
            symbol = self._reverse_mappings[symbol_int]
            
            # Stop at delimiter
            if symbol == '/':
                break
            
            generated_sequence.append(symbol)
        
        return generated_sequence
    
    def generate_melody(self, latent_vector=None, temperature=1.0):
        """Generate melody using melody VAE"""
        return self.generate_from_latent(self.melody_vae, latent_vector, temperature)
    
    def generate_harmony(self, latent_vector=None, temperature=1.0):
        """Generate harmony using harmony VAE"""
        return self.generate_from_latent(self.harmony_vae, latent_vector, temperature)
    
    def interpolate_melodies(self, latent_start, latent_end, num_steps=5):
        """
        Interpolate between two latent vectors to create smooth transitions
        """
        melodies = []
        for alpha in np.linspace(0, 1, num_steps):
            latent_interp = (1 - alpha) * latent_start + alpha * latent_end
            melody = self.generate_melody(latent_interp)
            melodies.append(melody)
        return melodies
    
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
                        chord_str = start_symbol[1:-1]
                        pitches = [int(p) for p in chord_str.split(',')]
                        m21_event = m21.chord.Chord(pitches, quarterLength=quarter_length_duration)
                    
                    # Handle single note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    
                    stream.append(m21_event)
                    step_counter = 1
                
                start_symbol = symbol
            else:
                step_counter += 1
        
        return stream
    
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name='vae_melody.mid'):
        """Save melody as MIDI"""
        stream = self._parse_sequence_to_stream(melody, step_duration)
        stream.write(format, file_name)
        print(f"Saved melody to {file_name}")
    
    def save_harmony(self, harmony, step_duration=0.25, format="midi", file_name='vae_harmony.mid'):
        """Save harmony as MIDI"""
        stream = self._parse_sequence_to_stream(harmony, step_duration)
        stream.write(format, file_name)
        print(f"Saved harmony to {file_name}")
    
    def save_combined(self, melody, harmony, step_duration=0.25, format="midi", file_name='vae_combined.mid'):
        """Save melody and harmony as combined MIDI"""
        score = m21.stream.Score()
        
        # Melody part
        melody_part = m21.stream.Part()
        melody_part.id = 'Melody'
        melody_stream = self._parse_sequence_to_stream(melody, step_duration)
        for element in melody_stream:
            melody_part.append(element)
        
        # Harmony part
        harmony_part = m21.stream.Part()
        harmony_part.id = 'Harmony'
        harmony_stream = self._parse_sequence_to_stream(harmony, step_duration)
        for element in harmony_stream:
            harmony_part.append(element)
        
        score.insert(0, melody_part)
        score.insert(0, harmony_part)
        
        score.write(format, file_name)
        print(f"Saved combined music to {file_name}")
        
        return score
    
    def generate_music(self, melody_latent=None, harmony_latent=None, 
                      temperature_melody=1.0, temperature_harmony=1.0):
        """Generate both melody and harmony from latent space"""
        print("Generating melody from latent space...")
        melody = self.generate_melody(melody_latent, temperature_melody)
        
        print("Generating harmony from latent space...")
        harmony = self.generate_harmony(harmony_latent, temperature_harmony)
        
        return melody, harmony


# Import VAE classes
from maestro_vae_model import VAE, Sampling

if __name__ == '__main__':
    vg = VAEMusicGenerator()
    
    print("=" * 50)
    print("GENERATING MUSIC WITH VAE")
    print("="* 50)
    
    # Generate random samples
    print("\n1. Generating random sample...")
    melody, harmony = vg.generate_music(
        temperature_melody=0.8,
        temperature_harmony=0.7
    )
    
    print(f"Generated melody length: {len(melody)}")
    print(f"Generated harmony length: {len(harmony)}")
    
    # Save
    vg.save_melody(melody, file_name='vae_melody.mid')
    vg.save_harmony(harmony, file_name='vae_harmony.mid')
    vg.save_combined(melody, harmony, file_name='vae_combined.mid')
    
    # Generate multiple variations
    print("\n2. Generating 3 variations...")
    for i in range(3):
        melody, harmony = vg.generate_music(temperature_melody=0.9, temperature_harmony=0.8)
        vg.save_combined(melody, harmony, file_name=f'vae_variation_{i+1}.mid')
    
    # Interpolation demo
    print("\n3. Generating interpolated melodies...")
    latent_start = np.random.normal(0, 1, size=(1, vg.melody_vae.latent_dim))
    latent_end = np.random.normal(0, 1, size=(1, vg.melody_vae.latent_dim))
    interpolated = vg.interpolate_melodies(latent_start, latent_end, num_steps=5)
    
    for i, melody in enumerate(interpolated):
        vg.save_melody(melody, file_name=f'vae_interpolation_{i+1}.mid')
    
    print("\n" + "=" * 50)
    print("VAE GENERATION COMPLETE!")
    print("=" * 50)