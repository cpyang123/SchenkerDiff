#!/usr/bin/env python3
"""
find_large_xml.py

Recursively search a directory for .xml files and print those larger than a given size.

Usage:
    python find_large_xml.py <directory> <size_in_bytes>
"""
import sys
from pathlib import Path

def parse_args():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory> <size_in_bytes>")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Error: {root} is not a directory.")
        sys.exit(1)
    try:
        threshold = int(sys.argv[2])
    except ValueError:
        print("Error: size_in_bytes must be an integer.")
        sys.exit(1)
    return root, threshold

def main():
    root_dir, threshold = parse_args()
    # rglob will recurse and match any file ending with .xml (case-insensitive)
    for xml_path in root_dir.rglob("*.xml"):
        try:
            size = xml_path.stat().st_size
        except OSError as e:
            print(f"Warning: could not stat {xml_path}: {e}", file=sys.stderr)
            continue
        if size > threshold:
            print(f"{xml_path}  ({size} bytes)")

if __name__ == "__main__":
    main()
