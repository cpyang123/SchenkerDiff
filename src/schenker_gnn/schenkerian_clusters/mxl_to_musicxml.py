from music21 import *
import os

def convert_mxl_to_musicxml(mxl_path, output_dir=None):
    # Parse the MXL file
    score = converter.parse(mxl_path)

    # Determine the output file path
    if output_dir:
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mxl_path))[0] + ".xml")
    else:
        output_path = os.path.splitext(mxl_path)[0] + ".xml"

    # Write the score as MusicXML
    score.write('musicxml', fp=output_path)

    return output_path


def convert_all_mxl_to_musicxml(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mxl"):
            mxl_path = os.path.join(input_dir, filename)
            convert_mxl_to_musicxml(mxl_path, output_dir)


if __name__ == "__main__":
    # Example usage:
    input_directory = "mxls"
    output_directory = "xmls"

    convert_all_mxl_to_musicxml(input_directory, output_directory)
    print("Conversion completed.")