from music21 import converter, stream
from pathlib import Path
import json

def keep_SATB_voices(file_path: Path):
    try:
        score = converter.parse(str(file_path))
        parts = list(score.parts)  # list of Part objects

        if len(parts) < 4:
            print(f"⚠️ Not enough parts in {file_path.name} (found {len(parts)}), skipping.")
            return

        # container for the four voices
        satb = {'soprano': None, 'alto': None, 'tenor': None, 'bass': None}

        # first pass: try to identify by explicit partName or instrumentName
        for part in parts:
            name = ""
            if hasattr(part, 'partName') and part.partName:
                name = part.partName.lower()
            elif hasattr(part, 'instrumentName') and part.instrumentName:
                name = part.instrumentName.lower()

            if any(k in name for k in ['soprano', 'cantus', 'treble']):
                satb['soprano'] = part
            elif any(k in name for k in ['alto', 'discantus', 'decant']):
                satb['alto'] = part
            elif 'tenor' in name:
                satb['tenor'] = part
            elif any(k in name for k in ['bass', 'bassus']):
                satb['bass'] = part

        # fallback: if none were found by name, assume standard SATB order S=0,A=1,T=2,B=3
        if any(v is None for v in satb.values()):
            print(f"⚠️ Could not identify all voices by name in {file_path.name}, using SATB ordering (0–1–2–3).")
            satb['soprano'], satb['alto'], satb['tenor'], satb['bass'] = parts[:4]

        # ensure we actually have all four
        missing = [voice for voice, part in satb.items() if part is None]
        if missing:
            print(f"⚠️ Missing voices {missing} in {file_path.name}, skipping.")
            return

        # build a new score with just the SATB parts
        new_score = stream.Score()
        # copy over metadata if you like:
        if score.metadata:
            new_score.insert(0, score.metadata)

        # insert in the traditional order
        for voice in ('soprano', 'alto', 'tenor', 'bass'):
            new_score.insert(0, satb[voice])

        # overwrite the original file
        new_score.write('musicxml', fp=str(file_path))
        print(f"✅ Kept SATB voices in: {file_path.name}")

    except Exception as e:
        print(f"❌ Failed to process {file_path.name}: {e}")


def process_all_musicxml_keep_SATB(root_folder: str):
    root_path = Path(root_folder)
    for file in root_path.rglob("*.xml"):
        keep_SATB_voices(file)


# If you also want to stop blanking-out your inner voices in the JSON,
# just remove or comment out the `process_all_json_remove_inner` calls
# (or leave that function untouched if you still need it elsewhere).

if __name__ == "__main__":
    process_all_musicxml_keep_SATB("../schenkerian_clusters/")
