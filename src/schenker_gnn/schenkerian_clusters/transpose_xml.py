import os

from music21 import converter, interval, pitch


def transpose(source_file, tonic="C", major=True, key_known=True):
    # Load the MusicXML file
    score = converter.parse(source_file)

    if not key_known:
        k = score.analyze("key")
        tonic = k.tonic.name
        major = k.mode == "major"

    if major:
        same_mode_keys = ["C", "G", "D", "A", "E", "B", "F#", "C#", "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F"]
    else:
        same_mode_keys = ["a", "e", "b", "f#", "c#", "g#", "d#", "a#", "ab", "eb", "bb", "f", "c", "g", "d"]


    # Transpose the piece to each key of the same mode and write to new MusicXML files
    for key in same_mode_keys:
        if key.upper() == tonic.upper(): continue
        i = interval.Interval(pitch.Pitch(tonic), pitch.Pitch(key))
        transposed_score = score.transpose(i)
        transposed_score.write('musicxml', f'{source_file[:-4]}_trans_{key}.xml')




def transpose_WTC():
    parent_directory = "C:\\Users\\88ste\\PycharmProjects\\forks\\gnn-music-analysis\\schenkerian_clusters"
    for item in os.listdir(parent_directory):
        print(item)
        item_path = os.path.join(parent_directory, item)
        if not os.path.isdir(item_path) or item in ["mxls", "xmls"] or item[:3] != "WTC": continue
        split_item = item.split("_")
        try:
            if split_item[3] == "min":
                transpose(f"./{item}/{item}.xml", tonic=split_item[2], major=False)
            elif split_item[3] == "maj":
                transpose(f"./{item}/{item}.xml", tonic=split_item[2], major=True)
        except FileNotFoundError as e:
            print(e)

def transpose_Pachelbel():
    parent_directory = "C:\\Users\\88ste\\PycharmProjects\\forks\\gnn-music-analysis\\schenkerian_clusters"
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        split_item = item.split("_")
        if not os.path.isdir(item_path) or item in ["mxls", "xmls"] or split_item[0] not in ["Primi","Secundi","Tertii","Quarti","Quinti","Sexti","Septimi","Octavi"]: continue
        print(item)
        try:
            transpose(f"./{item}/{item}.xml", key_known=False)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    # transpose("./WTC_II_C_maj/WTC_II_C_maj.xml", "C", major=True)
    transpose_WTC()
    # transpose_Pachelbel()
