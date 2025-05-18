import json
import os
import pickle
from pitch_alignment import align

def process_raw(filepath):
    with open(filepath, "r") as file:
        data_dict = json.load(file)

    note_categories = {
        'treble': data_dict["trebleNotes"],
        'innerTreble': data_dict["innerTrebleNotes"],
        'innerBass': data_dict["innerBassNotes"],
        'bass': data_dict["bassNotes"]
    }

    depths = {key: [] for key in note_categories}
    idx = 0

    for i in range(len(data_dict["trebleNotes"]["pitchNames"])):
        for j, (key, notes) in enumerate(note_categories.items()):
            pitch_name = notes["pitchNames"][i]
            if pitch_name != "_":
                depths[key].append(notes["depths"][i])
                idx += 1
            else:
                depths[key].append(-1)
    return depths['treble'], depths['bass']


def depth_to_edges(depth_list, v, depths_to_global, start_idx_global):
    all_edges = []

    #global_start = int(start_idx_global + v/3)
    #global_end = global_start + 2

    while any(depth >= 0 for depth in depth_list):
        indices = [i for i, depth in enumerate(depth_list) if depth >= 0]
        mapped_indices = [depths_to_global[(v, i)] for i in indices]

        if len(mapped_indices) > 1:
            edges_at_depth = []
            for i in range(len(mapped_indices) - 1):
                edges_at_depth.append([mapped_indices[i], mapped_indices[i + 1]])
            #start_idx = edges_at_depth[0][0]
            #end_idx = edges_at_depth[-1][1]
            #edges_at_depth.append([start_idx, global_start])
            #edges_at_depth.append([end_idx, global_end])
            all_edges.append(edges_at_depth)
        depth_list = [depth - 1 for depth in depth_list]
    return all_edges


def main():
    base_path = "schenkerian_clusters"

    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                json_filepath = os.path.join(subdir, file)
                xml_filepath = json_filepath.replace('.json', '.xml')

                try:
                    t_depth, b_depth = process_raw(json_filepath)
                    depths_to_global = align(xml_filepath, json_filepath)

                    start_idx_global = max(depths_to_global.values()) + 1

                    t_edges = depth_to_edges(t_depth, 0, depths_to_global, start_idx_global)
                    b_edges = depth_to_edges(b_depth, 3, depths_to_global, start_idx_global)
                    
                    pickle_path = os.path.join(subdir, file.replace('.json', '.pkl'))
                    with open(pickle_path, 'wb') as pkl_file:
                        pickle.dump({'t_edges': t_edges, 'b_edges': b_edges}, pkl_file)
                    
                    #print(f"Processed and saved: {pickle_path}")

                except Exception as e:
                    print(f"Skipping file {file} due to error: {e}")
                    pass


if __name__ == "__main__":
    main()