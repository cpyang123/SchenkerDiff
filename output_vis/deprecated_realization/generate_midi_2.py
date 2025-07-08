import math
import pretty_midi
import random

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






def create_midi_from_graph_octaves(X, R, scale_degrees, scale_degree_to_midi,
                                   tempo_multiplier=1.0,
                                   output_file='output.mid'):
    """
    Like before, but uses these R‐columns:
       duration    = R[i][6]
       offset_norm = R[i][7]
       voice_low   = R[i][-2]
    and spaces voices by octaves, centers each voice on its own octave,
    tracks prev_pitch and timing per-voice, and downshifts an octave if any note >127.
    """
    # map each distinct low_to_high norm → a voice index
    low_norms = sorted({row[-2] for row in R})
    voice_to_idx = {
        norm: idx - math.floor(len(low_norms) / 2)
        for idx, norm in enumerate(low_norms)
    }

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    # per-voice timing and state
    offset_by_voice = {}  # cumulative time per voice
    last_offset_norm = {}  # previous offset_norm per voice
    last_duration = {}  # previous duration per voice
    prev_pitch_by_voice = {}  # previous pitch per voice
    central_pitch = 60  # middle‐C reference

    for i, scale_idx in enumerate(X):
        degree = scale_degrees[scale_idx]
        base = scale_degree_to_midi[degree]

        # unpack R‐fields
        duration = R[i][6]
        offset_norm = R[i][7]
        low_norm = R[i][-2]

        # voice index for this note
        v_idx = voice_to_idx[low_norm]
        prev = prev_pitch_by_voice.get(v_idx)

        # octave‐shift base by voice index
        base += 12 * v_idx

        # determine raw pitch near prev
        if prev is None:
            raw = base
        else:
            # two candidates around prev
            low_cand = base + 12 * math.floor((prev - base) / 12)
            high_cand = low_cand + 12

            # choose candidate closer to prev (or +- octave center)
            prev_best = low_cand if abs(low_cand - prev) <= abs(high_cand - prev) else high_cand
            voice_center = central_pitch + 12 * v_idx
            cent_best = low_cand if abs(low_cand - voice_center) <= abs(high_cand - voice_center) else high_cand

            # mostly stick to voice‐center if it aligns, else small random bias
            raw = cent_best if (prev_best == cent_best or random.random() < 0.9) else prev_best

        pitch = int(round(raw))

        # --- timing per voice ---
        # initialize if first time we see this voice
        if v_idx not in offset_by_voice:
            offset_by_voice[v_idx] = 0.0
            last_offset_norm[v_idx] = None
            last_duration[v_idx] = 0.0

        # if offset_norm changed on this voice, advance its time by its last duration
        if last_offset_norm[v_idx] is not None and offset_norm != last_offset_norm[v_idx]:
            offset_by_voice[v_idx] += last_duration[v_idx]

        start = offset_by_voice[v_idx]
        end = start + duration

        # append note
        inst.notes.append(pretty_midi.Note(
            velocity=40,
            pitch=pitch,
            start=start,
            end=end
        ))

        # update per-voice trackers
        prev_pitch_by_voice[v_idx] = raw
        last_offset_norm[v_idx] = offset_norm
        last_duration[v_idx] = duration

    # ensure no pitch >127 by dropping full octaves
    # for n in inst.notes:
    #     while n.pitch > 127:
    #         n.pitch -= 12
    #     n.pitch = max(0, n.pitch)

    pm.instruments.append(inst)
    pm.write(output_file)
    print(f"MIDI file written to {output_file}")


if __name__ == '__main__':
    # Define the scale degrees and mapping to MIDI pitches (for a base octave, with tonic at middle C).
    scale_degrees = [
        'A2', 'm2', 'P8', 'A6', 'A1', 'A5',
        'm7', 'M2', 'm6', 'M7',
        'm3', 'M3', 'P5', 'd7',
        'P4', 'M6', 'A4', 'd5'
    ]

    scale_degree_to_midi = {
        'P8': 60,  # tonic at middle C
        'A1': 60 + 1,
        'm2': 60 + 1,  # 61
        'M2': 60 + 2,  # 62
        'm3': 60 + 3,  # 63
        'A2': 60 + 3,  # 63 (augmented 2nd, enharmonically equal to m3)
        'M3': 60 + 4,  # 64
        'P4': 60 + 5,  # 65
        'A4': 60 + 6,  # 66
        'd5': 60 + 6,  # 66 (diminished 5th, same as A4)
        'P5': 60 + 7,  # 67
        'A5': 60 + 8,
        'm6': 60 + 8,  # 68
        'M6': 60 + 9,  # 69
        'd7': 60 + 9,  # 69 (diminished 7th, same as M6)
        'm7': 60 + 10,  # 70
        'A6': 60 + 10,  # 70 (augmented 6th, same as m7)
        'M7': 60 + 11,  # 71
    }

    # Parse the file (which may contain multiple graphs)
    file_path = '../generated_samples1.txt'
    graphs = parse_generated_file(file_path)
    print(f"Found {len(graphs)} graph(s) in the file.")

    # Process each graph and create a separate MIDI file for each.
    for idx, (N, X, E, R) in enumerate(graphs):
        print(f"\nGraph {idx + 1}:")
        print("N =", N)
        print("X =", X)
        print("E =", E)
        print("R =", R)
        output_file = f'output_graph_{idx + 1}.mid'
        create_midi_from_graph_octaves(X, R, scale_degrees, scale_degree_to_midi, output_file=output_file,
                                       tempo_multiplier=3.0)
