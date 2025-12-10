import sys, os
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import keras
import numpy as np
import json
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class PianoGenerator:
    def __init__(self, model_path='model_piano.h5'):
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path)
        
        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ['/'] * SEQUENCE_LENGTH
        
    def generate_music(self, seed, num_steps, max_sequence_length, temperature):
        """Generate full piano music"""
        # Create seed with start symbols
        seed = seed.split()
        music = seed
        seed = self._start_symbols + seed
        
        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            
            # Prepare input (integers only, no one-hot encoding)
            input_seed = np.array(seed)[np.newaxis, :]
            
            # Make prediction
            probabilities = self.model.predict(input_seed, verbose=0)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # Update seed
            seed.append(output_int)
            
            # Map int to symbol
            output_symbols = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # Check for end
            if output_symbols == '/':
                break
            
            # Update music
            music.append(output_symbols)
            
        return music
    
    def _sample_with_temperature(self, probabilities, temperature):
        """Sample from probability distribution with temperature"""
        predictions = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        
        return index
    
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
                        try:
                            m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        except ValueError:
                            step_counter = 1
                            start_symbol = symbol
                            continue
                    
                    stream.append(m21_event)
                    step_counter = 1
                
                start_symbol = symbol
            else:
                step_counter += 1
        
        return stream
    
    def save_music(self, music, step_duration=0.25, format="midi", file_name='generated_piano.mid'):
        """Save generated piano music as MIDI"""
        stream = self._parse_sequence_to_stream(music, step_duration)
        stream.write(format, file_name)
        print(f"Saved music to {file_name}")
        return stream


if __name__ == '__main__':
    pg = PianoGenerator()
    
    # Example seed 2.txt
    seed = "[28,40,52,64] r [27,51,63] [28,40,64] r [29,65] [30,42,66] [31,67] [30,42,54,66] [29,53,65] [30,31,42,54,66] [55,67] [32,44,56,68] r [33,45,57,69] [34,46,58,70] [34,35,46,47,59,70,71]"
    # example seed 43.txt
    seed2 = "[51] _ _ _ _ [52] [52,53] [53] _ [56] _ _ [57] _ _ _ [60] _ _ _ _ [60] _ _ _ _ _ _ _ _ _ [58,60] _ _ [59] [60] [61] _ [61,62] [62] _ _ "
    #example seed 30.txt
    seed3 = "[72] [48,52,55,72] _ [72] _ _ _ _ [69,71] [67,68] [64,69] [65,67] [60] _ r _ _ _ _ [36,48,52,55,60] [60] _ [64] _ [36,48,52,55,60] [52,55,60] r _ _ _ "
    
    
    print("=" * 50)
    print("GENERATING PIANO MUSIC")
    print("=" * 50)
    
    # Generate music
    music = pg.generate_music(
        seed=seed3,
        num_steps=500,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=0.8
    )
    
    print(f"\nGenerated music length: {len(music)}")
    print(f"First 50 symbols: {' '.join(music[:50])}")
    
    # Save
    stream = pg.save_music(music, file_name='generated_piano3.mid')
    
    # Optional: view in MuseScore
    # stream.show()
    
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE!")
    print("=" * 50)