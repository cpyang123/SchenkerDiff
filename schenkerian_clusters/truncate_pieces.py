#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET

def truncate_musicxml(filepath: str, max_bars: int = 20):
    """
    Parse a MusicXML file and keep only the first `max_bars` measures
    in each <part>. Overwrites the original file.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    # MusicXML typically uses a default namespace; '{*}' matches any.
    part_tag    = './/{*}part'
    measure_tag = '{*}measure'

    for part in root.findall(part_tag):
        measures = part.findall(measure_tag)
        # remove all measures after the first max_bars
        for m in measures[max_bars:]:
            part.remove(m)

    # write back, preserving XML declaration
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

def main(root_dir: str, max_bars: int = 20):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.xml'):
                fullpath = os.path.join(dirpath, fname)
                try:
                    truncate_musicxml(fullpath, max_bars)
                    print(f"Truncated {fullpath} → first {max_bars} bars")
                except Exception as e:
                    print(f"⚠️  Failed on {fullpath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python truncate_musicxml.py <root_dir> [max_bars]")
        sys.exit(1)

    root_directory = sys.argv[1]
    bars_to_keep   = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(root_directory, bars_to_keep)
