def create_midi_from_graph(X, R, scale_degrees, scale_degree_to_midi, tempo_multiplier=1.0, output_file='output.mid'):
    """
    Creates a MIDI file from the note (X) and rhythm (R) data.

    Parameters:
      X: list of integers, where each integer indexes into scale_degrees.
      R: list of lists of floats, where each row corresponds to a note event.
         The second-to-last value in each row is the note duration, and the last value is the offset (start time).
      scale_degrees: list of strings representing the available scale degrees.
      scale_degree_to_midi: dict mapping each scale degree string to a MIDI pitch number (for a base octave).
      output_file: the file name for the generated MIDI file.

    Note:
      Instead of using an absolute mapping, this version maps each scale degree to the octave that is closest
      to the previous note, assuming we are in the key of C.

      For scale degrees whose base pitch is larger than a major third (i.e. > MIDI pitch for 'E'),
      the note is mapped with a higher probability to the candidate note that is closest to the central octave
      (here defined as the octave around MIDI pitch 60) rather than to the one closest to the previous note.
      For scale degrees less than or equal to a major third, we always map to the candidate closest to the previous note.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    offset = 0
    prev_pitch = None
    # Define the reference central octave (Middle C)
    central_pitch = 60

    for i, scale_index in enumerate(X):
        # Get the target scale degree and its base MIDI pitch.
        note_degree = scale_degrees[scale_index]
        base_pitch = scale_degree_to_midi[note_degree]

        if i == 0 or prev_pitch is None:
            pitch = base_pitch
        else:
            # Compute two candidates: one by lowering to the nearest octave and one by raising.
            candidate_low = base_pitch + 12 * math.floor((prev_pitch - base_pitch) / 12)
            candidate_high = candidate_low + 12

            # Check if this scale degree is larger than a major third.
            # Assuming we are in C major, we compare to the MIDI pitch for 'E'.
            if base_pitch > scale_degree_to_midi.get('E', 64):
                # Candidate closest to the previous note.
                candidate_prev = candidate_low if abs(candidate_low - prev_pitch) <= abs(
                    candidate_high - prev_pitch) else candidate_high
                # Candidate closest to the central octave reference.
                candidate_central = candidate_low if abs(candidate_low - central_pitch) <= abs(
                    candidate_high - central_pitch) else candidate_high

                # If both candidates are the same, just use that candidate.
                if candidate_prev == candidate_central:
                    pitch = candidate_prev
                else:
                    # With higher probability (70%), choose the candidate closer to the central octave.
                    if random.random() < 0.8:
                        pitch = candidate_central
                    else:
                        pitch = candidate_prev
            else:
                # For scale degrees less than or equal to a major third, always use the candidate closest to the previous note.
                pitch = candidate_low if abs(candidate_low - prev_pitch) <= abs(
                    candidate_high - prev_pitch) else candidate_high

        prev_pitch = pitch

        # Retrieve duration and offset from R.
        duration = R[i][-3]
        offset_norm = R[i][-2]
        if i > 0 and offset_norm != R[i - 1][-2]:
            offset += R[i - 1][-3]
        start_time = offset
        end_time = offset + duration

        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_file)
    print(f"MIDI file written to {output_file}")

# def create_midi_from_graph(X, R, scale_degrees, scale_degree_to_midi, tempo_multiplier = 1.0, output_file='output.mid'):
#     """
#     Creates a MIDI file from the note (X) and rhythm (R) data.

#     Parameters:
#       X: list of integers, where each integer indexes into scale_degrees.
#       R: list of lists of floats, where each row corresponds to a note event.
#          The second-to-last value in each row is the note duration, and the last value is the offset (start time).
#       scale_degrees: list of strings representing the available scale degrees.
#       scale_degree_to_midi: dict mapping each scale degree string to a MIDI pitch number (for a base octave).
#       output_file: the file name for the generated MIDI file.

#     Note:
#       Instead of using an absolute mapping, this version maps each scale degree to the octave that is closest
#       to the previous note, assuming we are in the key of C.
#     """
#     pm = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=0)

#     offset = 0
#     prev_pitch = None
#     for i, scale_index in enumerate(X):
#         # Get the target scale degree and its base MIDI pitch.
#         note_degree = scale_degrees[scale_index]
#         base_pitch = scale_degree_to_midi[note_degree]

#         if i == 0 or prev_pitch is None:
#             pitch = base_pitch
#         else:
#             # Compute two candidates: one by lowering to the nearest octave and one by raising.
#             candidate_low = base_pitch + 12 * math.floor((prev_pitch - base_pitch) / 12)
#             candidate_high = candidate_low + 12
#             # Choose the candidate that is closest to the previous note.
#             if abs(candidate_low - prev_pitch) <= abs(candidate_high - prev_pitch):
#                 pitch = candidate_low
#             else:
#                 pitch = candidate_high

#         prev_pitch = pitch

#         # The last two values in each row of R correspond to duration and offset.
#         duration = R[i][-3]
#         offset_norm = R[i][-2]
#         if i > 0:
#             if offset_norm != R[i - 1][-2]:
#                 offset += R[i-2][-3]
#         start_time = offset
#         end_time = offset + duration

#         note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
#         instrument.notes.append(note)

#     pm.instruments.append(instrument)
#     pm.write(output_file)
#     print(f"MIDI file written to {output_file}")
