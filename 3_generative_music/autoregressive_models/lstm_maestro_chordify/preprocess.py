import os
import music21 as m21
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAESTRO_DIR = "D:/maestro-v3.0.0-midi/2017"
SAVE_DIR = os.path.join(SCRIPT_DIR, "dataset")
SINGLE_FILE_DATASET = os.path.join(SCRIPT_DIR, "merged_dataset")
MAPPING_PATH = os.path.join(SCRIPT_DIR, "mapping.json")

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
NUM_MEASURES = 16  # Number of measures to extract

def load_songs_in_midi(dataset_path, limit=None):
    """Load MIDI files from dataset"""
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                try:
                    song = m21.converter.parse(os.path.join(path, file))
                    songs.append(song)
                    print(f"Loaded: {file}")
                    
                    if limit and len(songs) >= limit:
                        return songs
                        
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
                    
    return songs

def extract_first_n_measures(song, n_measures=16):
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

def encode_song(song, temp_step=0.25):
    """
    Encode entire piano score using chordify to capture simultaneity
    """
    encoded = []
    
    try:
        # Chordify combines all simultaneous notes into chords
        chordified = song.chordify()
        elements = chordified.flatten().notesAndRests
    except Exception as e:
        print(f"Warning: Chordify failed ({e}), using flatten")
        elements = song.flatten().notesAndRests
    
    for element in elements:
        # Handle single notes
        if isinstance(element, m21.note.Note):
            symbol = str(element.pitch.midi)
        
        # Handle chords (multiple simultaneous notes)
        elif isinstance(element, m21.chord.Chord):
            pitches = sorted([p.midi for p in element.pitches])
            symbol = f"[{','.join(map(str, pitches))}]"
        
        # Handle rests
        elif isinstance(element, m21.note.Rest):
            symbol = "r"
        
        else:
            continue
            
        # Convert to time series notation
        steps = int(element.duration.quarterLength / temp_step)
        for step in range(steps):
            if step == 0:
                encoded.append(symbol)
            else:
                encoded.append('_')
                
    encoded_str = " ".join(encoded)
    return encoded_str

def preprocess(dataset_path, limit=None):
    """Preprocess MAESTRO dataset - full piano score"""
    # Load songs
    print("Loading songs...")
    songs = load_songs_in_midi(dataset_path, limit=limit)
    print(f'Loaded {len(songs)} songs.')
    
    processed_count = 0
    
    for i, song in enumerate(songs):
        print(f"Processing song {i+1}/{len(songs)}...")
        
        # Extract first N measures
        song = extract_first_n_measures(song, NUM_MEASURES)
        
        # Filter out songs with non-acceptable durations
        # if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
        #     print(f"  Skipped: Non-acceptable durations")
        #     continue
            
        # Transpose to C-major / A-minor
        song = transpose(song)
        
        # Encode entire piano score
        encoded_song = encode_song(song)
        
        # Save to file
        save_path = os.path.join(SAVE_DIR, f"{i}.txt")
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)
            
        processed_count += 1
    
    print(f"Successfully processed {processed_count} songs.")
    return processed_count

def load(file_path):
    """Load encoded song from file"""
    with open(file_path, 'r') as fp:
        song = fp.read()
        return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Create merged dataset file"""
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    
    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        # Sort files numerically
        files = sorted([f for f in files if f.endswith('.txt')], 
                      key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
        
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter
    
    # Remove final empty space
    songs = songs[:-1]
    
    # Save merged dataset
    os.makedirs(file_dataset_path, exist_ok=True)
    save_path = os.path.join(file_dataset_path, "merged_data.txt")
    
    with open(save_path, 'w') as fp:
        fp.write(songs)
        
    return songs

def create_mapping(songs, mapping_path):
    """Create vocabulary mapping"""
    # Identify vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # Create mappings
    mappings = {}
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # Save vocab to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    
    print(f"Vocabulary size: {len(mappings)}")
    return mappings

def convert_songs_to_int(songs, mappings):
    """Convert song symbols to integers"""
    int_songs = []
    songs_list = songs.split()
    
    for symbol in songs_list:
        if symbol in mappings:
            int_songs.append(mappings[symbol])
        else:
            print(f"Warning: Unknown symbol '{symbol}' - skipping")
        
    return int_songs

def generate_training_sequences(sequence_length):
    """
    Generate training sequences for full piano score
    Returns: (inputs, targets, vocabulary_size)
    """
    # Load mappings
    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)
    
    # Load songs
    songs = load(os.path.join(SINGLE_FILE_DATASET, "merged_data.txt"))
    
    # Convert to integers
    int_songs = convert_songs_to_int(songs, mappings)
    
    # Generate sequences
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)
    vocabulary_size = len(mappings)
    
    print(f"Generated {len(inputs)} sequences")
    print(f"Vocabulary size: {vocabulary_size}")
    
    return inputs, targets, vocabulary_size

def main():
    """Main preprocessing pipeline"""
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Preprocess all songs
    preprocess(MAESTRO_DIR, limit=50)  # Adjust limit as needed
    
    # Create merged dataset
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    
    # Create vocabulary mapping
    create_mapping(songs, MAPPING_PATH)
    
    # Generate training sequences
    inputs, targets, vocab_size = generate_training_sequences(SEQUENCE_LENGTH)
    
    print("\nPreprocessing complete!")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")

if __name__ == "__main__":
    # Test on a single song first
    # print("Testing on single song...")
    # songs = load_songs_in_midi(MAESTRO_DIR, limit=1)
    
    # if len(songs) > 0:
    #     song = songs[0]
        
    #     print(f"Number of parts: {len(song.parts)}")
        
    #     # Check each part
    #     for i, part in enumerate(song.parts):
    #         print(f"\nPart {i}:")
    #         print(f"  Name: {part.partName}")
    #         notes = part.flatten().notesAndRests
    #         print(f"  Number of elements: {len(notes)}")
    #         print(f"  First 5 elements: {list(notes[:5])}")
        
    #     # Check flattened version
    #     flat_notes = song.flat.notesAndRests
    #     print(f"\nFlattened song:")
    #     print(f"  Total elements: {len(flat_notes)}")
    #     print(f"  First 10 elements: {list(flat_notes[:10])}")
        
    #     # Check chordified version
    #     try:
    #         chordified = song.chordify()
    #         chord_notes = chordified.flat.notesAndRests
    #         print(f"\nChordified song:")
    #         print(f"  Total elements: {len(chord_notes)}")
    #         print(f"  First 10 elements: {list(chord_notes[:10])}")
    #     except Exception as e:
    #         print(f"Chordify failed: {e}")
    
    # Run full preprocessing when ready
    main()