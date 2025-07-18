from music21 import stream, note, meter, key, tempo, interval
from rule_guidance import SCALE_DEGREES, SCALE_DEGREE_TO_C, SCALE_DEGREE_TO_CLASS, C_BASED_0, CLASS_TO_SCALE_DEGREE
import numpy as np

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
                X = [int(num) for num in lines[i + 1].split()]
                i += 2
            else:
                raise ValueError("Expected 'X:' marker after N= line")

            # Parse the adjacency matrix E (N lines following "E:")
            if i < len(lines) and lines[i] == "E:":
                E = []
                for j in range(N):
                    E.append([int(num) for num in lines[i + 1 + j].split()])
                i += 1 + N
            else:
                raise ValueError("Expected 'E:' marker")

            # Parse the rhythm matrix R (N lines following "R:")
            if i < len(lines) and lines[i] == "R:":
                R = []
                for j in range(N):
                    R.append([float(num) for num in lines[i + 1 + j].split()])
                i += 1 + N
            else:
                raise ValueError("Expected 'R:' marker")

            graphs.append((N, X, E, R))
        else:
            i += 1
    return graphs


def realization(X, R, tempo_multiplier=1.0, output_file='output.mid'):
    """
       duration    = R[i][6]
       offset_norm = R[i][7]
       voice_low   = R[i][-2]
    """

    # Create the score and parts
    score = stream.Score()
    treble_part = stream.Part(id='Treble')
    bass_part = stream.Part(id='Bass')

    # Optional: Set metadata
    treble_part.append(tempo.MetronomeMark(number=80))
    treble_part.append(key.KeySignature(0))

    bass_part.append(tempo.MetronomeMark(number=80))
    bass_part.append(key.KeySignature(0))

    # Initialize previous note trackers
    treble_center = "C5"
    bass_center = "C3"
    prev_treble = "C5"
    prev_bass = "C3"

    possible_octaves = [2, 3, 4, 5, 6]
    for note_idx, note_row in enumerate(R):
        # check if note is not bass or treble
        if note_row[8] not in [0.0, 1.0]:
            continue

        # match duration to quarterlength
        duration = note_row[6] * 2
        is_treble = bool(note_row[8])

        # convert note class to pitchclass based on tonic C
        note_class = X[note_idx]
        pitchclass = SCALE_DEGREE_TO_C[CLASS_TO_SCALE_DEGREE[note_class]]

        # check if it's a step away from previous note
        curr_pitch_int = C_BASED_0[pitchclass]
        prev_pitch_int = C_BASED_0[prev_treble[:-1] if is_treble else prev_bass[:-1]]
        if 0 <= abs(curr_pitch_int - prev_pitch_int) <= 2:
            candidate_notes, candidate_absolute_distances, candidate_actual_distances, indices_absolute = \
                find_closest_notes(
                    prev_treble if is_treble else prev_bass,
                    pitchclass,
                    possible_octaves
                )
        else:
            candidate_notes, candidate_absolute_distances, candidate_actual_distances, indices_absolute = \
                    find_closest_notes(
                        treble_center if is_treble else bass_center,
                        pitchclass,
                        possible_octaves
                    )

        nearest_note = candidate_notes[indices_absolute[0]]
        nearest_note.quarterLength = duration
        if is_treble:
            treble_part.append(nearest_note)
            prev_treble = nearest_note.nameWithOctave
        else:
            bass_part.append(nearest_note)
            prev_bass = nearest_note.nameWithOctave

    score.insert(0, treble_part)
    score.insert(0, bass_part)

    score.write("musicxml", fp=output_file)



def find_closest_notes(from_note_string, to_pitchclass, possible_octaves):
    from_note = note.Note(from_note_string)
    candidate_notes = []
    candidate_absolute_distances = []
    candidate_actual_distances = []
    for octave in possible_octaves:
        candidate_note_string = f'{to_pitchclass}{octave}'
        candidate_note = note.Note(candidate_note_string)

        candidate_interval = interval.Interval(from_note, candidate_note)

        candidate_notes.append(candidate_note)
        candidate_absolute_distances.append(abs(candidate_interval.semitones))
        candidate_actual_distances.append(candidate_interval.semitones)

    indices_absolute = np.argsort(candidate_absolute_distances)
    return candidate_notes, candidate_absolute_distances, candidate_actual_distances, indices_absolute


if __name__ == '__main__':
    # Parse the file (which may contain multiple graphs)
    file_path = './generated_samples1.txt'
    graphs = parse_generated_file(file_path)
    print(f"Found {len(graphs)} graph(s) in the file.")

    # Process each graph and create a separate MIDI file for each.
    for idx, (N, X, E, R) in enumerate(graphs):
        print(f"\nGraph {idx + 1}:")
        print("N =", N)
        print("X =", X)
        print("E =", E)
        print("R =", R)
        output_file = f'output_graph_{idx + 1}.xml'
        realization(X, R, output_file=output_file, tempo_multiplier=3.0)

