import os
import music21 as m21
import json
import numpy as np

MAESTRO_DIR = "D:/maestro-v3.0.0-midi/2017"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "merged_dataset"
MAPPING_PATH = "mapping.json"

ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]
SEQUENCE_LENGTH = 64
NUM_MEASURES = 20

def load_songs_in_midi(dataset_path, limit=None):
    """Load MIDI files from dataset"""
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                try:
                    song = m21.converter.parse(os.path.join(path, file))
                    songs.append(song)
                    
                    if limit and len(songs) >= limit:
                        return songs
                        
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
                    
    return songs

def extract_first_n_measures(song, n_measures=6):
    """Extract only the first n measures from a song"""
    try:
        excerpt = song.measures(0, n_measures)
        return excerpt
    except:
        print(f"Warning: Could not extract {n_measures} measures")
        return song

def has_acceptable_durations(song, accept_durations):
    """Check if all notes have acceptable durations"""
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in accept_durations:
            return False
    return True

def transpose(song):
    """Transpose song to C major or A minor"""
    # Get key from the song
    try:
        key = song.analyze("key")
    except:
        print("Warning: Could not analyze key, skipping transpose")
        return song
    
    # Get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    else:
        return song
            
    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song

def separate_melody_harmony(song):
    """
    Separate melody (highest notes) from harmony (lower notes/chords)
    Returns two separate streams
    """
    melody_stream = m21.stream.Stream()
    harmony_stream = m21.stream.Stream()
    
    # Get all notes and chords with their offsets
    for element in song.flat.notesAndRests:
        if isinstance(element, m21.note.Note):
            # Single notes - check if it's likely melody (higher register)
            if element.pitch.midi >= 60:  # Middle C and above = melody
                melody_stream.insert(element.offset, element)
            else:
                harmony_stream.insert(element.offset, element)
                
        elif isinstance(element, m21.chord.Chord):
            # Chords - split into melody (highest note) and harmony (rest)
            pitches = element.pitches
            highest = max(pitches, key=lambda p: p.midi)
            
            # Melody: highest note
            melody_note = m21.note.Note(highest)
            melody_note.duration = element.duration
            melody_stream.insert(element.offset, melody_note)
            
            # Harmony: remaining notes as chord
            if len(pitches) > 1:
                harmony_pitches = [p for p in pitches if p != highest]
                harmony_chord = m21.chord.Chord(harmony_pitches)
                harmony_chord.duration = element.duration
                harmony_stream.insert(element.offset, harmony_chord)
                
        elif isinstance(element, m21.note.Rest):
            # Add rests to both streams
            melody_stream.insert(element.offset, element)
            harmony_stream.insert(element.offset, element)
    
    return melody_stream, harmony_stream

def encode_stream(stream, temp_step=0.25):
    """
    Encode a stream (melody or harmony) into time series representation
    For harmony with chords, we'll encode as comma-separated MIDI numbers
    """
    encoded = []
    
    for event in stream.notesAndRests:
        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = str(event.pitch.midi)
        # Handle chords (harmony)
        elif isinstance(event, m21.chord.Chord):
            # Encode chord as comma-separated MIDI numbers
            pitches = sorted([p.midi for p in event.pitches])
            symbol = f"[{','.join(map(str, pitches))}]"
        # Handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        else:
            continue
            
        # Convert note/rest/chord into time series notation
        steps = int(event.duration.quarterLength / temp_step)
        for step in range(steps):
            if step == 0:
                encoded.append(symbol)
            else:
                encoded.append('_')
                
    # Cast encoded to string
    encoded_str = " ".join(encoded)
    return encoded_str

def encode_song_with_melody_harmony(song, temp_step=0.25):
    """
    Encode song with separate melody and harmony tracks
    Returns a tuple of (melody_encoding, harmony_encoding)
    """
    melody_stream, harmony_stream = separate_melody_harmony(song)
    
    melody_encoded = encode_stream(melody_stream, temp_step)
    harmony_encoded = encode_stream(harmony_stream, temp_step)
    
    return melody_encoded, harmony_encoded

def preprocess(dataset_path, limit=None):
    """Preprocess MAESTRO dataset"""
    # Load songs
    print("Loading songs...")
    songs = load_songs_in_midi(dataset_path, limit=limit)
    print(f'Loaded {len(songs)} songs.')
    
    processed_count = 0
    
    for i, song in enumerate(songs):
        print(f"Processing song {i+1}/{len(songs)}...")
        
        # Extract first 6 measures
        song = extract_first_n_measures(song, NUM_MEASURES)
        
        # Filter out songs with non-acceptable durations
        # if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
        #     print(f"  Skipped: Non-acceptable durations")
        #     continue
            
        # Transpose to C-major / A-minor
        song = transpose(song)
        
        # Encode song with melody and harmony separation
        melody_encoded, harmony_encoded = encode_song_with_melody_harmony(song)
        
        # Save both melody and harmony to separate files
        melody_path = os.path.join(SAVE_DIR, f"{i}_melody.txt")
        harmony_path = os.path.join(SAVE_DIR, f"{i}_harmony.txt")
        
        with open(melody_path, 'w') as fp:
            fp.write(melody_encoded)
        with open(harmony_path, 'w') as fp:
            fp.write(harmony_encoded)
            
        processed_count += 1
    
    print(f"Successfully processed {processed_count} songs.")
    return processed_count

def load(file_path):
    """Load encoded song from file"""
    with open(file_path, 'r') as fp:
        song = fp.read()
        return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Create merged dataset files for melody and harmony separately"""
    new_song_delimiter = "/ " * sequence_length
    
    melody_songs = ""
    harmony_songs = ""
    
    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        melody_files = sorted([f for f in files if f.endswith('_melody.txt')])
        
        for melody_file in melody_files:
            # Get corresponding harmony file
            base_name = melody_file.replace('_melody.txt', '')
            harmony_file = f"{base_name}_harmony.txt"
            
            melody_path = os.path.join(path, melody_file)
            harmony_path = os.path.join(path, harmony_file)
            
            if os.path.exists(harmony_path):
                melody_song = load(melody_path)
                harmony_song = load(harmony_path)
                
                melody_songs += melody_song + " " + new_song_delimiter
                harmony_songs += harmony_song + " " + new_song_delimiter
    
    # Remove final empty space
    melody_songs = melody_songs[:-1]
    harmony_songs = harmony_songs[:-1]
    
    # Save merged datasets
    os.makedirs(file_dataset_path, exist_ok=True)
    
    melody_save_path = os.path.join(file_dataset_path, "merged_melody.txt")
    harmony_save_path = os.path.join(file_dataset_path, "merged_harmony.txt")
    
    with open(melody_save_path, 'w') as fp:
        fp.write(melody_songs)
    with open(harmony_save_path, 'w') as fp:
        fp.write(harmony_songs)
        
    return melody_songs, harmony_songs

def create_mapping(melody_songs, harmony_songs, mapping_path):
    """Create vocabulary mapping for both melody and harmony"""
    # Combine all symbols from both melody and harmony
    all_symbols = melody_songs.split() + harmony_songs.split()
    vocabulary = list(set(all_symbols))
    
    # Create mappings
    mappings = {}
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # Save vocab to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    
    return mappings

def convert_songs_to_int(songs, mappings):
    """Convert song symbols to integers"""
    int_songs = []
    songs_list = songs.split()
    
    for symbol in songs_list:
        int_songs.append(mappings[symbol])
        
    return int_songs

def generate_training_sequences(sequence_length):
    """
    Generate training sequences for both melody and harmony
    Returns: (melody_inputs, melody_targets, harmony_inputs, harmony_targets)
    """
    # Load mappings
    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)
    
    # Load melody and harmony songs
    melody_songs = load(os.path.join(SINGLE_FILE_DATASET, "merged_melody.txt"))
    harmony_songs = load(os.path.join(SINGLE_FILE_DATASET, "merged_harmony.txt"))
    
    # Convert to integers
    int_melody = convert_songs_to_int(melody_songs, mappings)
    int_harmony = convert_songs_to_int(harmony_songs, mappings)
    
    # Generate sequences for melody
    melody_inputs = []
    melody_targets = []
    num_sequences = len(int_melody) - sequence_length
    
    for i in range(num_sequences):
        melody_inputs.append(int_melody[i:i+sequence_length])
        melody_targets.append(int_melody[i+sequence_length])
    
    # Generate sequences for harmony
    harmony_inputs = []
    harmony_targets = []
    num_sequences = len(int_harmony) - sequence_length
    
    for i in range(num_sequences):
        harmony_inputs.append(int_harmony[i:i+sequence_length])
        harmony_targets.append(int_harmony[i+sequence_length])
    
    # One-hot encode
    vocabulary_size = len(mappings)
    
    # Melody one-hot encoding
    melody_inputs = np.array(melody_inputs)
    melody_targets = np.array(melody_targets)
    
    # Harmony one-hot encoding
    harmony_inputs = np.array(harmony_inputs)
    harmony_targets = np.array(harmony_targets)
    
    print(f"Generated {len(melody_inputs)} melody sequences")
    print(f"Generated {len(harmony_inputs)} harmony sequences")
    print(f"Vocabulary size: {vocabulary_size}")
    
    return melody_inputs, melody_targets, harmony_inputs, harmony_targets, vocabulary_size

def main():
    """Main preprocessing pipeline"""
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Preprocess all songs
    preprocess(MAESTRO_DIR, limit=1)  # Start with 50 songs for testing
    
    # Create merged datasets
    melody_songs, harmony_songs = create_single_file_dataset(
        SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH
    )
    
    # Create vocabulary mapping
    create_mapping(melody_songs, harmony_songs, MAPPING_PATH)
    
    # Generate training sequences
    melody_inputs, melody_targets, harmony_inputs, harmony_targets, vocab_size = \
        generate_training_sequences(SEQUENCE_LENGTH)
    
    print("\nPreprocessing complete!")
    print(f"Melody input shape: {melody_inputs.shape}")
    print(f"Harmony input shape: {harmony_inputs.shape}")

if __name__ == "__main__":
    # Test on a single song first
    print("Testing on single song...")
    songs = load_songs_in_midi(MAESTRO_DIR, limit=1)
    
    if len(songs) > 0:
        song = songs[0]
        print(f"Loaded song with {len(song.flat.notesAndRests)} elements")
        
        # Extract first 6 measures
        excerpt = extract_first_n_measures(song, NUM_MEASURES)
        print(f"Excerpt has {len(excerpt.flat.notesAndRests)} elements")
        
        # Transpose
        transposed = transpose(excerpt)
        
        # Encode
        melody, harmony = encode_song_with_melody_harmony(transposed)
        print(f"\nMelody encoding (first 200 chars):\n{melody[:200]}")
        print(f"\nHarmony encoding (first 200 chars):\n{harmony[:200]}")
        
        # Uncomment to view in MuseScore
        # from your earlier code
        # def view_excerpt(song, start_measure=0, end_measure=8):
        #     excerpt = song.measures(start_measure, end_measure)
        #     excerpt.write('musicxml', f'excerpt_{start_measure}-{end_measure}.musicxml')
        #     print(f"Saved excerpt to: excerpt_{start_measure}-{end_measure}.musicxml")
        #     return excerpt
        # view_excerpt(song, 0, 6)
    
    # Run full preprocessing when ready
    main()