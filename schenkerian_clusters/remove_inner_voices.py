from music21 import converter
from pathlib import Path
import json

def keep_soprano_and_bass_only(file_path: Path):
    try:
        score = converter.parse(str(file_path))

        # Get parts by index (assuming SATB: 0=S, 1=A, 2=T, 3=B)
        parts = score.parts.stream()

        if len(parts) < 4:
            print(f"⚠️ Not enough parts in {file_path.name} (found {len(parts)}), skipping.")
            return

        soprano = parts[0]
        bass = parts[3]

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
    process_all_json_remove_inner("../schenkerian_clusters/")