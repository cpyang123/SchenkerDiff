#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def find_fermata_measures_in_soprano(filepath: str):
    """
    Parse a MusicXML file and find all measures that contain fermatas in the highest voice (soprano).
    Returns a list of measure numbers (1-based) where fermatas occur in the soprano part.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    fermata_measures = set()
    
    # Find all parts
    parts = root.findall('.//{*}part')
    if not parts:
        return []
    
    # Assume the first part is the soprano (highest voice)
    soprano_part = parts[0]
    measures = soprano_part.findall('{*}measure')
    
    for i, measure in enumerate(measures):
        measure_number = i + 1  # 1-based measure numbering
        
        # Check for fermata in notations
        fermatas = measure.findall('.//{*}fermata')
        if fermatas:
            fermata_measures.add(measure_number)
            continue
        
        # Check for fermata in articulations
        articulations = measure.findall('.//{*}articulations')
        for articulation in articulations:
            if articulation.find('{*}fermata') is not None:
                fermata_measures.add(measure_number)
                break
    
    return sorted(list(fermata_measures))

def split_measure_at_fermata(measure):
    """
    Split a measure at the fermata point, returning notes before+including fermata 
    and notes after fermata.
    Returns (notes_up_to_fermata, notes_after_fermata)
    """
    all_elements = list(measure)
    fermata_found = False
    fermata_index = -1
    
    # Find all note elements and their positions
    for i, element in enumerate(all_elements):
        if element.tag.endswith('note'):
            # Check if this note has a fermata
            has_fermata = False
            
            # Check in notations
            notations = element.find('{*}notations')
            if notations is not None:
                fermata = notations.find('{*}fermata')
                if fermata is not None:
                    has_fermata = True
            
            # Check in articulations within notations
            if not has_fermata and notations is not None:
                articulations = notations.find('{*}articulations')
                if articulations is not None:
                    fermata = articulations.find('{*}fermata')
                    if fermata is not None:
                        has_fermata = True
            
            if has_fermata:
                fermata_found = True
                fermata_index = i
                break
    
    if not fermata_found:
        return all_elements, []
    
    # Split elements at fermata point
    elements_up_to_fermata = all_elements[:fermata_index + 1]
    elements_after_fermata = all_elements[fermata_index + 1:]
    
    return elements_up_to_fermata, elements_after_fermata

def truncate_measure_at_fermata(measure):
    """
    Remove all elements that come after a fermata in the given measure.
    """
    elements_up_to_fermata, elements_after_fermata = split_measure_at_fermata(measure)
    
    # Remove all elements after fermata
    for element in elements_after_fermata:
        measure.remove(element)

def create_partial_measure_from_fermata(original_measure, measure_number):
    """
    Create a new measure containing only the elements that come after the fermata.
    """
    elements_up_to_fermata, elements_after_fermata = split_measure_at_fermata(original_measure)
    
    if not elements_after_fermata:
        return None
    
    # Create new measure with same attributes
    new_measure = ET.Element(original_measure.tag, original_measure.attrib)
    
    # Copy measure number (increment it slightly to indicate it's a partial measure)
    if 'number' in original_measure.attrib:
        new_measure.set('number', str(measure_number))
    
    # Add elements that come after fermata
    for element in elements_after_fermata:
        new_measure.append(element)
    
    return new_measure

def create_segments_at_fermatas(filepath: str, output_dir: str = None):
    """
    Parse a MusicXML file and create separate segments ending at each fermata.
    Each segment contains measures from the beginning (or previous fermata) up to and including the fermata,
    with no notes after the fermata.
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    fermata_measures = find_fermata_measures_in_soprano(filepath)
    
    if not fermata_measures:
        print(f"⚠️  No fermatas found in soprano voice of {filepath}")
        return
    
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Create segments ending at each fermata
    base_filename = Path(filepath).stem
    start_measure = 1
    
    for seg_idx, fermata_measure in enumerate(fermata_measures):
        end_measure = fermata_measure
        
        # Create a copy of the original tree for this segment
        segment_tree = ET.parse(filepath)
        segment_root = segment_tree.getroot()
        
        # Process each part in the segment
        for part_idx, part in enumerate(segment_root.findall('.//{*}part')):
            measures = part.findall('{*}measure')
            
            # Remove measures outside the current segment
            measures_to_remove = []
            for i, measure in enumerate(measures):
                measure_number = i + 1  # 1-based
                if measure_number < start_measure or measure_number > end_measure:
                    measures_to_remove.append(measure)
                elif measure_number == end_measure:
                    # This is the fermata measure - remove notes after fermata
                    truncate_measure_at_fermata(measure)
            
            for measure in measures_to_remove:
                part.remove(measure)
        
        # Save the segment
        segment_filename = f"{base_filename}_segment_{seg_idx + 1:02d}_measures_{start_measure}-{end_measure}.xml"
        segment_filepath = os.path.join(output_dir, segment_filename)
        
        segment_tree.write(segment_filepath, encoding='utf-8', xml_declaration=True)
        print(f"✅ Created segment: {segment_filename} (measures {start_measure}-{end_measure}, ends at fermata)")
        
        # Next segment starts after this fermata
        start_measure = fermata_measure + 1
    
    print(f"📊 Created {len(fermata_measures)} segments from fermatas in soprano voice")

def main(root_dir: str, create_segments_dir: bool = False):
    """
    Process all XML files in the directory and create segments at fermatas in soprano voice.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.xml'):
                fullpath = os.path.join(dirpath, fname)
                
                # Create a segments subdirectory if requested
                if create_segments_dir:
                    segments_dir = os.path.join(dirpath, "segments")
                    os.makedirs(segments_dir, exist_ok=True)
                    output_dir = segments_dir
                else:
                    output_dir = dirpath
                
                try:
                    create_segments_at_fermatas(fullpath, output_dir)
                    # Delete the original file after successful segmentation
                    os.remove(fullpath)
                    print(f"📁 Processed and deleted original: {fullpath}")
                except Exception as e:
                    print(f"❌ Failed on {fullpath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python truncate_pieces.py <root_dir> [create_segments_dir]")
        print("  root_dir: Directory containing XML files to process")
        print("  create_segments_dir: Create 'segments' subdirectory (default: false)")
        print("  ")
        print("This script creates segments of musical pieces, where each segment")
        print("ends at a fermata found in the soprano (highest) voice.")
        sys.exit(1)

    root_directory = sys.argv[1]
    create_segments_subdir = sys.argv[2].lower() != 'false' if len(sys.argv) > 2 else False
    
    main(root_directory, create_segments_subdir)
