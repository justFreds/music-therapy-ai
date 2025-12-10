import keras
import numpy as np
import json
import music21 as m21
from maestro_preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MusicGenerator:
    def __init__(self, model_path='model_combined.h5'):
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path)
        
        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ['/'] * SEQUENCE_LENGTH
        
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generate melody sequence"""
        # Create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            
            # NO ONE-HOT ENCODING! Just reshape for batch dimension
            # Model expects (batch_size, sequence_length) of integers
            input_seed = np.array(seed)[np.newaxis, :]  # Shape: (1, sequence_length)
            
            # Make a prediction
            probabilities = self.model.predict(input_seed, verbose=0)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # Update seed
            seed.append(output_int)
            
            # Map int to our encoding
            output_symbols = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # Check whether we're at the end of a melody
            if output_symbols == '/':
                break
            
            # Update melody
            melody.append(output_symbols)
            
        return melody
    
    def generate_harmony(self, seed, num_steps, max_sequence_length, temperature):
        """Generate harmony sequence - same logic as melody but for harmony"""
        # Create seed with start symbols
        seed = seed.split()
        harmony = seed
        seed = self._start_symbols + seed
        
        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            
            # NO ONE-HOT ENCODING! Just reshape for batch dimension
            input_seed = np.array(seed)[np.newaxis, :]  # Shape: (1, sequence_length)
            
            # Make a prediction
            probabilities = self.model.predict(input_seed, verbose=0)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # Update seed
            seed.append(output_int)
            
            # Map int to our encoding
            output_symbols = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # Check whether we're at the end
            if output_symbols == '/':
                break
            
            # Update harmony
            harmony.append(output_symbols)
            
        return harmony
    
    def _sample_with_temperature(self, probabilities, temperature):
        """Sample from probability distribution with temperature"""
        predictions = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        
        return index
    
    def _parse_sequence_to_stream(self, sequence, step_duration=0.25):
        """
        Parse a sequence (melody or harmony) into a music21 stream
        Handles both regular notes and chord notation [60,64,67]
        """
        stream = m21.stream.Stream()
        
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(sequence):
            # Handle case in which we have a note/rest/chord
            if symbol != '_' or i + 1 == len(sequence):
                # Ensure we are dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    
                    # Case 1: handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    
                    # Case 2: handle chord [60,64,67]
                    elif start_symbol.startswith('[') and start_symbol.endswith(']'):
                        # Parse chord notation
                        chord_str = start_symbol[1:-1]  # Remove brackets
                        pitches = [int(p) for p in chord_str.split(',')]
                        m21_event = m21.chord.Chord(pitches, quarterLength=quarter_length_duration)
                    
                    # Case 3: handle single note
                    else:
                        try:
                            m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                        except ValueError:
                            # Skip invalid symbols
                            step_counter = 1
                            start_symbol = symbol
                            continue
                    
                    stream.append(m21_event)
                    
                    # Reset step counter
                    step_counter = 1
                
                start_symbol = symbol
            
            # Handle case in which we have a prolongation sign '_'
            else:
                step_counter += 1
        
        return stream
    
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name='generated_melody.mid'):
        """Save melody as MIDI file"""
        stream = self._parse_sequence_to_stream(melody, step_duration)
        stream.write(format, file_name)
        print(f"Saved melody to {file_name}")
    
    def save_harmony(self, harmony, step_duration=0.25, format="midi", file_name='generated_harmony.mid'):
        """Save harmony as MIDI file"""
        stream = self._parse_sequence_to_stream(harmony, step_duration)
        stream.write(format, file_name)
        print(f"Saved harmony to {file_name}")
    
    def save_combined(self, melody, harmony, step_duration=0.25, format="midi", file_name='generated_combined.mid'):
        """
        Save melody and harmony as a combined MIDI file with two parts
        """
        # Create main score
        score = m21.stream.Score()
        
        # Create melody part (right hand / treble)
        melody_part = m21.stream.Part()
        melody_part.id = 'Melody'
        melody_stream = self._parse_sequence_to_stream(melody, step_duration)
        for element in melody_stream:
            melody_part.append(element)
        
        # Create harmony part (left hand / bass)
        harmony_part = m21.stream.Part()
        harmony_part.id = 'Harmony'
        harmony_stream = self._parse_sequence_to_stream(harmony, step_duration)
        for element in harmony_stream:
            harmony_part.append(element)
        
        # Add both parts to score
        score.insert(0, melody_part)
        score.insert(0, harmony_part)
        
        # Write to file
        score.write(format, file_name)
        print(f"Saved combined music to {file_name}")
        
        return score
    
    def generate_music(self, melody_seed, harmony_seed, num_steps, 
                      max_sequence_length, temperature_melody=0.7, temperature_harmony=0.7):
        """
        Generate both melody and harmony
        
        :param melody_seed: seed string for melody
        :param harmony_seed: seed string for harmony  
        :param num_steps: number of steps to generate
        :param max_sequence_length: max sequence length
        :param temperature_melody: temperature for melody generation
        :param temperature_harmony: temperature for harmony generation
        :return: tuple of (melody, harmony)
        """
        print("Generating melody...")
        melody = self.generate_melody(melody_seed, num_steps, max_sequence_length, temperature_melody)
        
        print("Generating harmony...")
        harmony = self.generate_harmony(harmony_seed, num_steps, max_sequence_length, temperature_harmony)
        
        return melody, harmony


if __name__ == '__main__':
    mg = MusicGenerator()
    
    # Example seeds (adjust based on your actual data)
    melody_seed = "62 53 _ _ _ r _ _ _ _ _ _  74 r 62 _ _ _ _ _ _ _ 74 _ _ _ _ _ _ _ _ 71 _ _ _ _ _ _ _ _ 53 _ _ _ r "
    harmony_seed = "[50] [41] _ _ _ r _ _ _ _ _ _ [29,41,62,68,71] r [71] _ _ _ _ _ _ _ _ [68] _ _ _ _ _ _ _ _ [41]"
    
    # Generate both melody and harmony
    print("=" * 50)
    print("GENERATING MUSIC")
    print("=" * 50)
    
    melody, harmony = mg.generate_music(
        melody_seed=melody_seed,
        harmony_seed=harmony_seed,
        num_steps=500,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature_melody=0.8,
        temperature_harmony=0.7
    )
    
    print(f"\nGenerated melody length: {len(melody)}")
    print(f"Generated harmony length: {len(harmony)}")
    
    # Save separately
    print("\nSaving files...")
    mg.save_melody(melody, file_name='generated_melody.mid')
    mg.save_harmony(harmony, file_name='generated_harmony.mid')
    
    # Save combined (melody + harmony in one file)
    score = mg.save_combined(melody, harmony, file_name='generated_combined.mid')
    
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE!")
    print("=" * 50)
    print("Files created:")
    print("  - generated_melody.mid")
    print("  - generated_harmony.mid")
    print("  - generated_combined.mid (melody + harmony together)")