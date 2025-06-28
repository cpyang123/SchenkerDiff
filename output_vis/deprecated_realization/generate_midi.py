import pretty_midi

def parse_generated_file(file_path):
    """
    Parses a file that contains one or more graphs.
    
    Each graph is expected to have the following structure:
    
      N=<number>
      X:
      <one line of space-separated integers>
      E:
      <N lines of space-separated integers>
      R:
      <N lines of space-separated floats>
      
    Returns:
      A list of tuples, each tuple containing (N, X, E, R) for one graph.
    """
    with open(file_path, 'r') as f:
        # Remove blank lines and strip whitespace
        lines = [line.strip() for line in f if line.strip()]
    
    graphs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("N="):
            # Parse the number of nodes
            N = int(lines[i].split('=')[1])
            i += 1
            
            # Parse the X list (one line after the "X:" marker)
            if i < len(lines) and lines[i] == "X:":
                X = [int(num) for num in lines[i+1].split()]
                i += 2
            else:
                raise ValueError("Expected 'X:' marker after N= line")
            
            # Parse the adjacency matrix E (N lines following "E:")
            if i < len(lines) and lines[i] == "E:":
                E = []
                for j in range(N):
                    E.append([int(num) for num in lines[i+1+j].split()])
                i += 1 + N
            else:
                raise ValueError("Expected 'E:' marker")
            
            # Parse the rhythm matrix R (N lines following "R:")
            if i < len(lines) and lines[i] == "R:":
                R = []
                for j in range(N):
                    R.append([float(num) for num in lines[i+1+j].split()])
                i += 1 + N
            else:
                raise ValueError("Expected 'R:' marker")
            
            graphs.append((N, X, E, R))
        else:
            i += 1
    return graphs

def create_midi_from_graph(X, R, scale_degrees, scale_degree_to_midi, tempo_multiplier = 1.0, output_file='output.mid'):
    """
    Creates a MIDI file from the note (X) and rhythm (R) data.
    
    Parameters:
      X: list of integers, where each integer indexes into scale_degrees.
      R: list of lists of floats, where each row corresponds to a note event.
         The second-to-last value in each row is the note duration, and the last value is the offset (start time).
      scale_degrees: list of strings representing the available scale degrees.
      scale_degree_to_midi: dict mapping each scale degree string to a MIDI pitch number.
      output_file: the file name for the generated MIDI file.
    """
    # Create a PrettyMIDI object and a piano instrument (program number 0)
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    offset = 0
    for i, scale_index in enumerate(X):
        # Map the integer in X to a scale degree, then to a MIDI pitch.
        note_degree = scale_degrees[scale_index]
        pitch = scale_degree_to_midi[note_degree]
        
        # The last two numbers in each row of R correspond to duration and offset.
        duration = R[i][-2]
        offset_norm = R[i][-1]
        if i > 0:
            if offset_norm != R[i - 1][-1]:
                offset += R[i-1][-2]
        start_time = offset
        end_time = offset + duration
        
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)
    
    pm.instruments.append(instrument)
    pm.write(output_file)
    print(f"MIDI file written to {output_file}")

if __name__ == '__main__':
    # Define the scale degrees and mapping to MIDI pitches.
    scale_degrees = [
        'A2', 'm2', 'P8', 'A6',
        'm7', 'M2', 'm6', 'M7',
        'm3', 'M3', 'P5', 'd7',
        'P4', 'M6', 'A4', 'd5'
    ]
    scale_degree_to_midi = {
        'P8': 60,    # tonic at middle C
        'm2': 60 + 1,  # 61
        'M2': 60 + 2,  # 62
        'm3': 60 + 3,  # 63
        'A2': 60 + 3,  # 63 (augmented 2nd, enharmonically equal to m3)
        'M3': 60 + 4,  # 64
        'P4': 60 + 5,  # 65
        'A4': 60 + 6,  # 66
        'd5': 60 + 6,  # 66 (diminished 5th, same as A4)
        'P5': 60 + 7,  # 67
        'm6': 60 + 8,  # 68
        'M6': 60 + 9,  # 69
        'd7': 60 + 9,  # 69 (diminished 7th, same as M6)
        'm7': 60 + 10, # 70
        'A6': 60 + 10, # 70 (augmented 6th, same as m7)
        'M7': 60 + 11, # 71
    }
    
    # Parse the file (which may contain multiple graphs)
    file_path = '../generated_samples1.txt'
    graphs = parse_generated_file(file_path)
    print(f"Found {len(graphs)} graph(s) in the file.")
    
    # Process each graph and create a separate MIDI file for each.
    for idx, (N, X, E, R) in enumerate(graphs):
        print(f"\nGraph {idx+1}:")
        print("N =", N)
        print("X =", X)
        print("E =", E)
        print("R =", R)
        output_file = f'output_graph_{idx+1}.mid'
        create_midi_from_graph(X, R, scale_degrees, scale_degree_to_midi, output_file=output_file, tempo_multiplier=3.0)
