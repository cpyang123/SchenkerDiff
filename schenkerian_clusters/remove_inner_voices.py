from music21 import converter
from pathlib import Path
import json

def keep_soprano_and_bass_only(file_path: Path):
    try:
        score = converter.parse(str(file_path))

        # Get all parts
        parts = score.parts.stream()

        if len(parts) < 2:
            print(f"⚠️ Not enough parts in {file_path.name} (found {len(parts)}), skipping.")
            return

        # Find soprano and bass parts by name
        soprano = None
        bass = None
        
        for part in parts:
            part_name = ""
            if hasattr(part, 'partName') and part.partName:
                part_name = part.partName.lower()
            elif hasattr(part, 'instrumentName') and part.instrumentName:
                part_name = part.instrumentName.lower()
            elif len(parts) == 4:  # Fallback to index for standard SATB
                part_index = list(parts).index(part)
                if part_index == 0:
                    part_name = "soprano"
                elif part_index == 3:
                    part_name = "bass"
            
            # Check if this part is soprano
            if any(keyword in part_name for keyword in ['soprano', 'treble', 's ', 's.', 'cantus']):
                soprano = part
            # Check if this part is bass
            elif any(keyword in part_name for keyword in ['bass', 'b ', 'b.', 'bassus']):
                bass = part

        # If we couldn't find by name and have exactly 4 parts, use traditional SATB ordering
        if soprano is None and bass is None and len(parts) == 4:
            print(f"⚠️ Could not identify parts by name in {file_path.name}, using SATB ordering (S=0, B=3)")
            soprano = parts[0]
            bass = parts[3]
        elif soprano is None or bass is None:
            # Try to find highest and lowest parts by pitch range
            if soprano is None and bass is None:
                # Find the parts with highest and lowest average pitches
                part_avg_pitches = []
                for part in parts:
                    pitches = []
                    for element in part.flatten():
                        if hasattr(element, 'pitch') and element.pitch:
                            pitches.append(element.pitch.midi)
                    if pitches:
                        avg_pitch = sum(pitches) / len(pitches)
                        part_avg_pitches.append((part, avg_pitch))
                
                if len(part_avg_pitches) >= 2:
                    part_avg_pitches.sort(key=lambda x: x[1])  # Sort by average pitch
                    bass = part_avg_pitches[0][0]  # Lowest
                    soprano = part_avg_pitches[-1][0]  # Highest
                    print(f"📝 Identified parts by pitch range in {file_path.name}")

        if soprano is None or bass is None:
            print(f"⚠️ Could not identify soprano and bass parts in {file_path.name}, skipping.")
            return

        new_score = score.__class__()  # new Score
        new_score.insert(0, soprano)
        new_score.insert(0, bass)

        new_score.write('musicxml', fp=str(file_path))
        print(f"✅ Removed inner voices in: {file_path.name}")

    except Exception as e:
        print(f"❌ Failed to process {file_path.name}: {e}")


def process_all_musicxml_remove_inner(root_folder: str):
    root_path = Path(root_folder)
    for file in root_path.rglob("*.xml"):
        keep_soprano_and_bass_only(file)

def remove_json_inner(file):
    with open(file, 'r') as f:
        analysis = json.load(f)
    num_verticalities = len(analysis['innerTrebleNotes']['pitchNames'])
    for voice in ['innerTrebleNotes', 'innerBassNotes']:
        analysis[voice]['pitchNames'] = ['_' for _ in range(num_verticalities)]
        for info in ['depths']:
            analysis[voice][info] = [0 for _ in range(num_verticalities)]
        for info in ['scaleDegree']:
            max_depth = 13
            analysis[voice][info] = [[0 for _ in range(num_verticalities)] for _ in range(max_depth)]
        for info in ['selected', 'ursatz', 'flagged', 'sharps', 'flats', 'naturals', 'parenthetical']:
            analysis[voice][info] = []
    with open(file, 'w') as f:
        json.dump(analysis, f)



def process_all_json_remove_inner(root_folder: str):
    root_path = Path(root_folder)
    for file in root_path.rglob("*.json"):
        remove_json_inner(file)


if __name__ == "__main__":
    process_all_musicxml_remove_inner("../schenkerian_clusters/")
    ("../schenkerian_clusters/")
