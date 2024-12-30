import json
import numpy as np


def process_raw(filepath):
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
        index_to_global = {}

        for i in range(len(t["pitchNames"])):
            if t["pitchNames"][i] != "_":
                t_depth.append(t["depths"][i])
                index_to_global[(0, i)] = idx
                idx += 1
            else:
                t_depth.append(-1)

            if b["pitchNames"][i] != "_" and b["pitchNames"][i] != t["pitchNames"][i]:
                b_depth.append(b["depths"][i])
                index_to_global[(3, i)] = idx
                idx += 1
            else:
                b_depth.append(-1)

            if ti["pitchNames"][i] != "_" and ti["pitchNames"][i] != t["pitchNames"][i] and ti["pitchNames"][i] != b["pitchNames"][i]:
                ti_depth.append(ti["depths"][i])
                index_to_global[(1, i)] = idx
                idx += 1
            else:
                ti_depth.append(-1)

            if bi["pitchNames"][i] != "_" and bi["pitchNames"][i] != t["pitchNames"] and bi["pitchNames"][i] != b["pitchNames"][i] and bi["pitchNames"][i] != ti["pitchNames"][i]:
                bi_depth.append(bi["depths"][i])
                index_to_global[(2, i)] = idx
                idx += 1
            else:
                bi_depth.append(-1)

    return t_depth, b_depth, ti_depth, bi_depth, index_to_global


def find_cluster_idx_outer(depth_list, i):
    for left_idx in range(i, -1, -1):
        if depth_list[left_idx] > 0:
            return left_idx

    for right_idx in range(i, len(depth_list)):
        if depth_list[right_idx] > 0:
            return right_idx
    
    return None


def find_closest_inner_to_outer(arr, idx, left=True): #helper for find_cluster_idx_inner
    range_func = range(idx-1, -1, -1) if left else range(idx+1, len(arr))
    for j in range_func:
        if arr[j] > 0:
            return j
    return None

def find_cluster_idx_inner(t_depth, b_depth, i_depth, i):
    for left_idx in range(i, -1, -1):
        if i_depth[left_idx] > 0:
            return [left_idx]
    for right_idx in range(i, len(i_depth)):
        if i_depth[right_idx] > 0:
            return [right_idx]

    ret = [i if t_depth[i] > 0 else None, i if b_depth[i] > 0 else None]
    if ret[0] is not None and ret[1] is not None:
        return ret
    
    if ret[0] is None and ret[1] is not None:
        ret[0] = find_cluster_idx_outer(t_depth, i)
        return ret
    if ret[1] is None and ret[0] is not None:
        ret[1] = find_cluster_idx_outer(b_depth, i)
        return ret
    
    #If both are None, cluster them to the outer.
    t1 = find_closest_inner_to_outer(t_depth, i, left=True)
    b1 = find_closest_inner_to_outer(b_depth, i, left=False)
    t2 = find_closest_inner_to_outer(t_depth, i, left=False)
    b2 = find_closest_inner_to_outer(b_depth, i, left=True)

    pairs = [(t1, b1), (t2, b2)]
    valid_pairs = [(t, b) for t, b in pairs if t is not None and b is not None]
    # print(valid_pairs)
    return min(valid_pairs, key=lambda x: min(abs(x[0]-i),abs(x[1]-i)))


def handle_inner_0_depth(t_depth, b_depth, v_depth, i, cluster_dict, note_global_idx, inner_to_outer_dict, voice_idx, index_to_global):
    cluster_to_idx = find_cluster_idx_inner(t_depth, b_depth, v_depth, i)
    if len(cluster_to_idx) == 1:
        cluster_dict[note_global_idx] = index_to_global[(voice_idx, cluster_to_idx[0])]
    else:
        t_idx = index_to_global[(0, cluster_to_idx[0])]
        b_idx = index_to_global[(3, cluster_to_idx[1])]
        inner_to_outer_dict[note_global_idx] = (t_idx, b_idx)
    return cluster_dict


def handle_outer_0_depth(voice_idx, voice_depth, i, index_to_global, cluster_dict, num_cluster, note_global_idx):
    cluster_to_idx = index_to_global[(voice_idx, find_cluster_idx_outer(voice_depth, i))]
    if cluster_to_idx in list(cluster_dict.keys()):
        cluster_dict[note_global_idx] = cluster_dict[cluster_to_idx]
    else:
        cluster_dict[cluster_to_idx] = num_cluster
        num_cluster += 1
        cluster_dict[note_global_idx] = cluster_dict[cluster_to_idx]
    return cluster_dict, num_cluster


def cluster_voice(t_depth, b_depth, voice_depth, voice_idx, cluster_dict, num_cluster, index_to_global, is_inner, inner_to_outer_dict):
    for i in range(len(voice_depth)):
        if voice_depth[i] == -1:
            continue
        note_global_idx = index_to_global[(voice_idx, i)]
        if voice_depth[i] > 0:
            if note_global_idx in list(cluster_dict.keys()):
                continue
            cluster_dict[note_global_idx] = num_cluster
            num_cluster += 1
        else:
            if is_inner:
                cluster_dict = handle_inner_0_depth(
                    t_depth, b_depth, voice_depth, i,
                    cluster_dict, note_global_idx, inner_to_outer_dict,
                    voice_idx, index_to_global
                )
            else:
                cluster_dict, num_cluster = handle_outer_0_depth(
                    voice_idx, voice_depth, i,
                    index_to_global, cluster_dict, num_cluster, note_global_idx
                )

    return cluster_dict, num_cluster


def get_matrix(t_depth, b_depth, ti_depth, bi_depth, index_to_global):
    voice_depths = {"t": t_depth, "b": b_depth, "ti": ti_depth, "bi": bi_depth}
    voice_indeces = {"t": 0, "b": 3, "ti": 1, "bi": 2}
    num_cluster = 0
    cluster_dict = {}

    inner_to_outer_dict = {}

    for voice_name in voice_depths.keys():
        is_inner = len(voice_name) == 2
        cluster_dict, num_cluster = cluster_voice(
            t_depth, b_depth, voice_depths[voice_name], voice_indeces[voice_name],
            cluster_dict, num_cluster, index_to_global, is_inner, inner_to_outer_dict
        )

    cluster_lists = [[] for _ in range(num_cluster)]

    for key, value in cluster_dict.items():
        cluster_lists[value].append(key)

    sorted_clusters = sorted(cluster_lists, key=min)
    matrix = np.zeros((max(cluster_dict.keys()) + 1, num_cluster), dtype=float)
    for i in range(len(sorted_clusters)):
        for j in sorted_clusters[i]:
            matrix[j, i] = 1

    for key, (a, b) in inner_to_outer_dict.items():
        c1 = np.where(matrix[a] == 1)[0]
        c2 = np.where(matrix[b] == 1)[0]
        matrix[key, c1[0]] = 0.5
        matrix[key, c2[0]] = 0.5

    index_to_global_new = {}
    for voice_name in voice_depths.keys():
        v_depth = [max(x - 1, -1) for x in voice_depths[voice_name]]

        for i in range(len(v_depth)):
            if v_depth[i] > -1:
                index_to_global_new[(voice_indeces[voice_name], i)] = np.where(
                    matrix[index_to_global[(voice_indeces[voice_name], i)]] == 1
                )[0][0]
        voice_depths[voice_name] = v_depth

    return matrix, voice_depths["t"], voice_depths["b"], voice_depths["ti"], voice_depths["bi"], index_to_global_new


def print_matrices(json_file):
    t_depth, b_depth, ti_depth, bi_depth, index_to_global = process_raw(json_file)
    while True:
        try:
            matrix, t_depth, b_depth, ti_depth, bi_depth, index_to_global = get_matrix(
                t_depth, b_depth, ti_depth,bi_depth, index_to_global
            )
            print(matrix.shape)
            #matrix_str = np.array_repr(matrix).replace('\n', '')
            #print(matrix_str)
            print(matrix)

        except Exception as e:
            print(e.with_traceback())
            break


def get_clusters(json_filepath):
    clusters = []
    t_depth, b_depth, ti_depth, bi_depth, index_to_global = process_raw(json_filepath)
    while True:
        try:
            matrix, t_depth, b_depth, ti_depth, bi_depth, index_to_global = get_matrix(
                t_depth, b_depth, ti_depth, bi_depth, index_to_global
            )
            # print(matrix.shape)
            # print(matrix)
            clusters.append(matrix)
        except KeyError as e:
            break
    return clusters


if __name__ == "__main__":
    #json_file = "data_processing\WTCII_F#m_Fugue_Subject.json"
    #json_file = "WTCI_cm_subject.json"
    #json_file = "./schenkerian_clusters\WTC_I_F_maj\WTC_I_F_maj.json"
    #json_file = "./schenkerian_clusters\WTC_I_G_min\WTC_I_G_min.json"
    json_file = ".\WTC_I_C#_min\WTC_I_C#_min.json"
    clusters = get_clusters(json_file)
    print(clusters)