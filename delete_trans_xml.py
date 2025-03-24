#!/usr/bin/env python3
import os
import sys

def delete_trans_xml_files(folder_path):
    """
    Recursively delete all .xml files in folder_path that contain '_trans_' in their filename.
    """
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.xml') and '_trans_' in filename:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_trans_xml.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
    
    delete_trans_xml_files(folder_path)
