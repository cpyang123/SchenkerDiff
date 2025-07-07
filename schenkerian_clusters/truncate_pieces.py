#!/usr/bin/env python3
"""
Split MusicXML files at fermatas in the soprano (first part) and export each segment as a new XML (MusicXML) file,
while preserving the active key and time signatures at each segment’s start and correct bar lines.
Deletes the original file on successful processing.
"""
import os
import sys
import argparse
import copy
from music21 import converter, stream, expressions, key, meter


def find_fermata_positions(soprano_part: stream.Part):
    """
    Find all fermata positions in the soprano part.
    Returns a list of tuples (measure_number, offset_in_quarter_length).
    """
    positions = []
    for measure in soprano_part.getElementsByClass(stream.Measure):
        for n in measure.notes:
            if any(isinstance(expr, expressions.Fermata) for expr in n.expressions):
                positions.append((measure.measureNumber, n.offset))
                break
    return positions


def create_segment(score: stream.Score, start_measure: int, end_measure: int, fermata_offset: float):
    """
    Build a new Score containing measures from start_measure..end_measure (inclusive).
    In the end_measure, truncate notes after fermata_offset.
    Preserves the active key and time signatures at the segment’s start.
    """
    segment = stream.Score()
    if score.metadata:
        segment.metadata = score.metadata

    for original_part in score.parts:
        new_part = stream.Part()
        new_part.id = original_part.id

        # Determine active key and time at the segment's start
        start_m = original_part.measure(start_measure)
        ks = start_m.getContextByClass(key.KeySignature)
        ts = start_m.getContextByClass(meter.TimeSignature)
        if ks:
            new_part.insert(0, copy.deepcopy(ks))
        if ts:
            new_part.insert(0, copy.deepcopy(ts))

        # Append measures, truncating the last one at the fermata offset
        for measure in original_part.getElementsByClass(stream.Measure):
            m_num = measure.measureNumber
            if m_num < start_measure or (end_measure is not None and m_num > end_measure):
                continue
            m_clone = copy.deepcopy(measure)
            if end_measure is not None and m_num == end_measure and fermata_offset is not None:
                for n in list(m_clone.notesAndRests):
                    if hasattr(n, 'offset') and n.offset > fermata_offset:
                        m_clone.remove(n)
            new_part.append(m_clone)

        segment.append(new_part)
    return segment


def split_file(filepath: str, output_dir: str):
    """
    Parse file, split at fermatas, write segments as XML (MusicXML).
    Returns True if at least one fermata was found and segments created.
    """
    score = converter.parse(filepath)
    soprano = score.parts[0]
    fermatas = find_fermata_positions(soprano)
    if not fermatas:
        print(f"⚠️  No fermatas found in {filepath}")
        return False

    last_measure = soprano.getElementsByClass(stream.Measure)[-1].measureNumber
    boundaries = []
    start = 1
    for m_num, offset in fermatas:
        boundaries.append((start, m_num, offset))
        start = m_num + 1
    if start <= last_measure:
        boundaries.append((start, None, None))

    base = os.path.splitext(os.path.basename(filepath))[0]
    for idx, (s, e, off) in enumerate(boundaries, start=1):
        segment = create_segment(score, s, e, off)
        fname = f"{base}_segment_{idx:02d}.xml"
        outpath = os.path.join(output_dir, fname)
        segment.write('xml', fp=outpath)
        rng = f"measures {s}-{e or last_measure}"
        print(f"✅ Created segment: {outpath}  ({rng})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split MusicXML files at fermatas into segments, then delete originals"
    )
    parser.add_argument('root_dir', help="Directory to search for XML files")
    parser.add_argument(
        '--segments-dir', action='store_true',
        help="Place outputs in a 'segments' subdirectory"
    )
    args = parser.parse_args()

    for dirpath, _, filenames in os.walk(args.root_dir):
        for fname in filenames:
            if not fname.lower().endswith(('.xml', '.musicxml')):
                continue
            full_path = os.path.join(dirpath, fname)
            out_dir = dirpath
            if args.segments_dir:
                out_dir = os.path.join(dirpath, 'segments')
                os.makedirs(out_dir, exist_ok=True)
            try:
                success = split_file(full_path, out_dir)
                if success:
                    os.remove(full_path)
                    print(f"🗑️  Deleted original: {full_path}")
            except Exception as e:
                print(f"❌ Failed on {full_path}: {e}")

if __name__ == '__main__':
    main()
