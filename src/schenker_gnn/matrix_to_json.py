import numpy as np
import pickle
import json


def read_file(file_path):
    with open(file_path, 'rb') as file:
        matrices = pickle.load(file)
    return matrices


def update_depths(matrices):
    n_rows = matrices[0].shape[0]
    cluster_to_index = {i: i for i in range(n_rows)}
    depths = [0] * n_rows

    for matrix in matrices:
        for c in range(matrix.shape[1]):
            rows_with_ones = np.where(matrix[:, c] == 1)[0]
            if len(rows_with_ones) > 0:
                r = rows_with_ones[0]
                note_index = cluster_to_index[r]
                depths[note_index] += 1
                del cluster_to_index[r]
                cluster_to_index[c] = note_index

    return depths


def convert_to_format(depths, global_to_depths):
    n_verticalities = max(value[1] for value in global_to_depths.values()) + 1
    depths_format = [[-1] * n_verticalities for _ in range(4)]
    for i in range(len(depths)):
        voice, verticality = global_to_depths[i]
        depths_format[voice][verticality] = depths[i]
    return depths_format


def process_raw(filepath):  # I cannot find the json_to_cluster_refactor.py so I just put it here. Almost the same code for processing raw json.
    with open(filepath, "r") as file:
        data_dict = json.load(file)
        t = data_dict["trebleNotes"]
        b = data_dict["bassNotes"]
        ti = data_dict["innerTrebleNotes"]
        bi = data_dict["innerBassNotes"]
        idx = 0
        t_depth = []
        b_depth = []
        ti_depth = []
        bi_depth = []
        global_to_depths = {}
        for i in range(len(t["pitchNames"])):
            if t["pitchNames"][i] != "_":
                t_depth.append(t["depths"][i])
                global_to_depths[idx] = (0, i)
                idx += 1
            else:
                t_depth.append(-1)
            if b["pitchNames"][i] != "_" and b["pitchNames"][i] != t["pitchNames"][i]:
                b_depth.append(b["depths"][i])
                global_to_depths[idx] = (3, i)
                idx += 1
            else:
                b_depth.append(-1)
            if ti["pitchNames"][i] != "_" and ti["pitchNames"][i] != t["pitchNames"][i] and ti["pitchNames"][i] != \
                    b["pitchNames"][i]:
                ti_depth.append(ti["depths"][i])
                global_to_depths[idx] = (1, i)
                idx += 1
            else:
                ti_depth.append(-1)
            if bi["pitchNames"][i] != "_" and bi["pitchNames"][i] != t["pitchNames"] and bi["pitchNames"][i] != \
                    b["pitchNames"][i] and bi["pitchNames"][i] != ti["pitchNames"][i]:
                bi_depth.append(bi["depths"][i])
                global_to_depths[idx] = (2, i)
                idx += 1
            else:
                bi_depth.append(-1)
    return t_depth, b_depth, ti_depth, bi_depth, global_to_depths


def generate_paths(piece):
    base_path = 'schenkerian_clusters'
    ground_truth_path = f'{base_path}/{piece}/{piece}.json'
    file_path = f'{base_path}/{piece}/{piece}.pkl'
    return ground_truth_path, file_path


if __name__ == "__main__":
    piece = 'Primi_1'
    ground_truth_path, file_path = generate_paths(piece)
    t_depth, b_depth, ti_depth, bi_depth, global_to_positions = process_raw(ground_truth_path)
    print('\n')
    print("--------------------Ground Truth--------------------")
    print(t_depth)
    print(ti_depth)
    print(bi_depth)
    print(b_depth)
    print('\n')
    print("--------------------Generated-----------------------")
    matrices = read_file(file_path)
    depths = update_depths(matrices)
    depths_format = convert_to_format(depths, global_to_positions)
    for i in range(len(depths_format)):
        print(depths_format[i])
