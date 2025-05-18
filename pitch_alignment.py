import json
from music21 import pitch, key
from pyScoreParser.musicxml_parser.mxp import MusicXMLDocument


def parse_pitch(pitch_name, key_sig, sharps, flats, naturals, index):
    if pitch_name == "_":
        return -1
    p = pitch.Pitch(pitch_name)

    step_alterations = key_sig.alteredPitches
    for altered_pitch in step_alterations:
        if p.name == altered_pitch.name[0]:
            p.accidental = altered_pitch.accidental

    if index in sharps:
        p.accidental = pitch.Accidental('sharp')
    elif index in flats:
        p.accidental = pitch.Accidental('flat')
    elif index in naturals:
        p.accidental = pitch.Accidental('natural')

    return p.midi


def extract_midi_pitches(note_data, key_sig):
    midi_pitches = []
    for i, note in enumerate(note_data['pitchNames']):
        midi_pitch = parse_pitch(note, key_sig, note_data['sharps'], note_data['flats'], note_data['naturals'], i)
        midi_pitches.append(midi_pitch)
    return midi_pitches


def convert_key_signature(key_str):
    if key_str == '0':
        return key.KeySignature(0)
    elif key_str[-1] == 's':
        sharps = int(key_str[:-1])
        return key.KeySignature(sharps)
    elif key_str[-1] == 'f':
        flats = -int(key_str[:-1])
        return key.KeySignature(flats)
    else:
        raise ValueError("Invalid key signature format")


def json_to_midi_pitch(filepath):
    with open(filepath, "r") as file:
        data_dict = json.load(file)

    note_categories = {
        'treble': data_dict["trebleNotes"],
        'innerTreble': data_dict["innerTrebleNotes"],
        'innerBass': data_dict["innerBassNotes"],
        'bass': data_dict["bassNotes"]
    }

    key_sig_str = data_dict["key"]
    key_sig = convert_key_signature(key_sig_str)

    t = extract_midi_pitches(note_categories['treble'], key_sig)
    it = extract_midi_pitches(note_categories['innerTreble'], key_sig)
    ib = extract_midi_pitches(note_categories['innerBass'], key_sig)
    b = extract_midi_pitches(note_categories['bass'], key_sig)
    return t, it, ib, b


def get_xml_note_list(xml_file_path):
    XMLDocument = MusicXMLDocument(str(xml_file_path))
    pyscoreparser_notes = XMLDocument.get_notes()
    ret = []
    for i in range(len(pyscoreparser_notes)):
        ret.append(pyscoreparser_notes[i].pitch[1])
    return ret


def align(xml, json):
    xml_notes = get_xml_note_list(xml)
    t, it, ib, b = json_to_midi_pitch(json)

    def find_first(lists):
        for lst in lists:
            if lst[0] != -1: return lst[0]
        raise Exception("No non-zero elements found")

    offset = find_first([t, it, ib, b]) - xml_notes[0]  # in case of transposition
    xml_notes = [p + offset for p in xml_notes]

    depths_to_global = {}
    j = -1

    def is_different(element, index, current_voice):
        if current_voice == 0:
            return True
        voices = [t, it, ib, b]
        for v_idx in range(current_voice):
            if voices[v_idx][index] == element:
                return False
        return True

    for i in range(len(t)):
        for voice_index, voice_list in enumerate([t, it, ib, b]):
            element = voice_list[i]
            if element == -1:
                continue

            if is_different(element, i, voice_index):
                if xml_notes[j + 1] == element:
                    j += 1
                    depths_to_global[(voice_index, i)] = j
                else:
                    print("File:", xml, "Alignment failed at index", j, "; voice", voice_index)
                    raise Exception("ALIGNMENT FAILED")
            else:
                depths_to_global[(voice_index, i)] = j

    return depths_to_global


if __name__ == "__main__":
    xml_filepath = 'schenkerian_clusters\\Primi_1\\Primi_1.xml'
    json_filepath = 'schenkerian_clusters\\Primi_1\\Primi_1.json'
    alignment = align(xml_filepath, json_filepath)
    dir = "visualization_playground\\depth_to_global_primi_1.pkl"
    import pickle
    with open(dir, 'wb') as f:
        pickle.dump(alignment,f)
    print(alignment)
