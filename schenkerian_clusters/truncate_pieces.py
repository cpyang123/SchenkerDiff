#!/usr/bin/env python3
"""
Split MusicXML files at fermatas in the soprano (first part) and export each segment as a new MusicXML file,
while preserving key and time signatures and correct bar lines.
"""
import os
import sys
import argparse
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
    Preserves key/time signatures and barlines.
    """
    segment = stream.Score()
    if score.metadata:
        segment.metadata = score.metadata

    for original_part in score.parts:
        new_part = stream.Part()
        new_part.id = original_part.id
        # Preserve the part's key and time signature at the start of the segment
        ks_list = original_part.getElementsByClass(key.KeySignature)
        if ks_list:
            new_part.insert(0, ks_list[0].clone())
        ts_list = original_part.getElementsByClass(meter.TimeSignature)
        if ts_list:
            new_part.insert(0, ts_list[0].clone())

        for measure in original_part.getElementsByClass(stream.Measure):
            m_num = measure.measureNumber
            if m_num < start_measure:
                continue
            if end_measure is not None and m_num > end_measure:
                continue
            m_clone = measure.clone()
            if end_measure is not None and m_num == end_measure and fermata_offset is not None:
                # Remove notes and chords starting after the fermata offset
                for n in list(m_clone.notesAndRests):
                    if hasattr(n, 'offset') and n.offset > fermata_offset:
                        m_clone.remove(n)
            new_part.append(m_clone)

        segment.append(new_part)
    return segment


def split_file(filepath: str, output_dir: str):
    """
    Parse file, find fermatas in soprano, split into segments, and write them out.
    """
    score = converter.parse(filepath)
    soprano = score.parts[0]
    fermatas = find_fermata_positions(soprano)
    if not fermatas:
        print(f"⚠️  No fermatas found in {filepath}")
        return

    last_measure = soprano.getElementsByClass(stream.Measure)[-1].measureNumber
    boundaries = []
    start = 1
    for m_num, offset in fermatas:
        boundaries.append((start, m_num, offset))
        start = m_num + 1
    # final segment if any measures remain
    if start <= last_measure:
        boundaries.append((start, None, None))

    base = os.path.splitext(os.path.basename(filepath))[0]
    for idx, (s, e, off) in enumerate(boundaries, start=1):
        segment = create_segment(score, s, e, off)
        fname = f"{base}_segment_{idx:02d}.musicxml"
        outpath = os.path.join(output_dir, fname)
        segment.write('musicxml', fp=outpath)
        rng = f"measures {s}-{e or last_measure}"
        print(f"✅ Created segment: {outpath}  ({rng})")


def main():
    parser = argparse.ArgumentParser(
        description="Split MusicXML files at fermatas in soprano into separate segments"
    )
    parser.add_argument('root_dir', help="Root directory to search for XML files")
    parser.add_argument(
        '--segments-dir', action='store_true',
        help="Place outputs in a 'segments' subdirectory alongside each file"
    )
    args = parser.parse_args()

    for dirpath, _, filenames in os.walk(args.root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.xml', '.musicxml')):
                full = os.path.join(dirpath, fname)
                if args.segments_dir:
                    out_dir = os.path.join(dirpath, 'segments')
                    os.makedirs(out_dir, exist_ok=True)
                else:
                    out_dir = dirpath
                try:
                    split_file(full, out_dir)
                except Exception as e:
                    print(f"❌ Failed on {full}: {e}")

if __name__ == '__main__':
    main()
