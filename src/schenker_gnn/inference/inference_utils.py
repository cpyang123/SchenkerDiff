import math
import os
import random
import pickle
from pathlib import Path
import heapq
from typing import Callable, Any, List

import torch

from src.schenker_gnn.config import *

import numpy as np

import numpy as np


def load_model(
        model,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        device,
        gnn_save_path,
        lp_treble_save_path,
        lp_bass_save_path,
        vp_save_path,
        reward_model=None,
        reward_save_path=None
):

    latest_gnn_file = get_latest_save_file(gnn_save_path, "gnn")
    latest_lp_treble_file = get_latest_save_file(lp_treble_save_path, "lp")
    latest_lp_bass_file = get_latest_save_file(lp_bass_save_path, "lp")
    latest_vp_file = get_latest_save_file(vp_save_path, "vp")

    latest_reward_file = None
    if reward_model is not None:
        latest_reward_file = get_latest_save_file(reward_save_path, "reward")

    if latest_gnn_file and latest_lp_treble_file and latest_lp_bass_file and latest_vp_file:
        model.load_state_dict(torch.load(latest_gnn_file, map_location=device))
        link_predictor_treble.load_state_dict(torch.load(latest_lp_treble_file, map_location=device))
        link_predictor_bass.load_state_dict(torch.load(latest_lp_bass_file, map_location=device))
        voice_predictor.load_state_dict(torch.load(latest_vp_file, map_location=device))

        all_to_device(model, link_predictor_treble, link_predictor_bass, voice_predictor, device=device)

        if latest_reward_file:
            reward_model.load_state_dict(torch.load(latest_reward_file, map_location=device))
            reward_model.to(device)
        return model, link_predictor_treble, link_predictor_bass, voice_predictor, reward_model
    else:
        raise FileNotFoundError("No saved model found")


def get_latest_save_file(save_path, identifier):
    files = [f for f in os.listdir(save_path) if f.startswith(identifier)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)
    latest_file = os.path.join(save_path, files[0]) if files else None
    return latest_file


def all_to_device(*args, device=DEVICE):
    for arg in args:
        arg.to(device)


def get_inference_all_depths(name, edge_prob_dict, voice_idx, beam_width):
    file_name = name.split('\\')[-1]
    with open(Path(f'.\\voice_classification_inferred\\{file_name}.pkl'), 'rb') as f:
        node_list = pickle.load(f)
    node_list = node_list[voice_idx]
    start_idx = max(max(key) for key in edge_prob_dict.keys()) - 2
    node_list.append(-1)
    node_list.append(start_idx + 2)
    edge_prob_dict_new = {}
    for key, value in edge_prob_dict.items():
        new_key = tuple(-1 if x == start_idx else x for x in key)
        if start_idx in key:
            new_key = new_key[::-1]
        edge_prob_dict_new[new_key] = value
    edge_prob_dict = {key: value for key, value in edge_prob_dict_new.items()
                      if key[0] in node_list and key[1] in node_list}

    def filter_dict(edge_prob_dict, start, end):
        edge_prob_dict_filtered = {key: value for key, value in edge_prob_dict.items()
                                   if not ((key[0] < start and start < key[1] < end)
                                           or (start < key[0] < end and end < key[1]))}
        edge_prob_dict_filtered.pop((start, end))
        return edge_prob_dict_filtered

    inferences = []

    # def pick_largest_items(d, k):
    #     largest_prob_edges = heapq.nlargest(k, d.items(), key=lambda item: item[1])
    #     return largest_prob_edges

    def pick_largest_items(d, k):
        if k == 0:
            return []

        selected_items = []
        current_dict = d.copy()

        for _ in range(k):
            if not current_dict:
                break

            largest_prob_element = max(current_dict.items(), key=lambda item: item[1])
            selected_items.append(largest_prob_element)

            start, end = largest_prob_element[0]

            filtered_out_dict = {key: value for key, value in current_dict.items()
                                 if (key[0] < start and start < key[1] < end)
                                 or (start < key[0] < end and end < key[1])}

            current_dict = filtered_out_dict

        return selected_items

    def select_edge_recursion(edge_prob_dict, beam_width, current_instance):
        if edge_prob_dict:
            edges_and_probs = pick_largest_items(edge_prob_dict, beam_width)
            for edge_prob in edges_and_probs:
                next_instance = current_instance
                next_instance.append(edge_prob)
                edge_prob_dict_filtered = filter_dict(edge_prob_dict, edge_prob[0][0], edge_prob[0][1])
                select_edge_recursion(edge_prob_dict_filtered, beam_width, next_instance)
        else:
            inferences.append(sorted(current_instance))

    select_edge_recursion(edge_prob_dict, beam_width, [])

    instance_cleaned = []
    for instance in inferences:
        instance = instance
        edges = []
        total_prob = 0
        for edge_prob in instance:
            edges.append(edge_prob[0])
            total_prob += edge_prob[1]
        instance_cleaned.append((edges, total_prob / len(edges)))
    return max(instance_cleaned, key=lambda x: x[1])[0]


def remove_consecutive_note_edges(edge_prob_dict, node_list):
    new_prob_dict = {}
    for edge, prob in edge_prob_dict.items():
        try:
            if abs(node_list.index(edge[0]) - node_list.index(edge[1])) != 1 \
                    or edge[1] == max(node_list) \
                    or edge[0] == max(node_list) - 1:
                new_prob_dict[edge] = prob
        except ValueError as e:
            continue
    return new_prob_dict

def remove_impossible_note_edges(edge_prob_dict, node_list):
    new_prob_dict = {}
    for edge, prob in edge_prob_dict.items():
        if edge[0] in node_list and edge[1] in node_list:
            new_prob_dict[edge] = prob
    return new_prob_dict


def sample_analysis(node_list, treble_prob_dict, bass_prob_dict, k, remove_consecutive=True, reward_model=False):
    treble = sample_analysis_voice("treble", node_list[0], treble_prob_dict, k, remove_consecutive=remove_consecutive, reward_model=reward_model)
    bass = sample_analysis_voice("bass", node_list[1], bass_prob_dict, k, remove_consecutive=remove_consecutive, reward_model=reward_model)
    return treble, bass


def get_node_list(voice_preds, version="threshold", threshold=0.4):
    treble, bass, inner = [], [], []
    if version == "threshold":
        for i, node in enumerate(voice_preds):
            if node[0].item() > threshold:
                treble.append(i)
            if node[1].item() > threshold:
                bass.append(i)
            if node[2].item() > threshold:
                inner.append(i)
    elif version == "probabilistic":
        while len(treble) == 0 or len(bass) == 0:
            treble, bass = [], []
            for i, node in enumerate(voice_preds[:-4]):
                if np.random.binomial(1, node[0].item(), 1)[0]:
                    treble.append(i)
                if np.random.binomial(1, node[1].item(), 1)[0]:
                    bass.append(i)
    # treble.append(len(voice_preds)-4)
    # treble.append(len(voice_preds)-3)
    # bass.append(len(voice_preds)-2)
    # bass.append(len(voice_preds)-1)
    # print("node_list:", [treble, bass, inner])
    return [treble, bass, inner]


def sample_analysis_voice(voice, node_list, edge_prob_dict, k, name=None, remove_consecutive=True, reward_model=False):
    # print("edge_prob_dict:", edge_prob_dict)
    # print("node_list:", node_list)
    if name:
        with open(Path(f'inference\\voice_classification_inferred\\{name}.pkl'), 'rb') as f:
            node_list = pickle.load(f)
    if voice == "treble":
        start_idx = max(max(key) for key in edge_prob_dict.keys()) - 3
        if reward_model:
            start_idx += 2
    elif voice == "bass":
        start_idx = max(max(key) for key in edge_prob_dict.keys()) - 1
    if remove_consecutive:
        edge_prob_dict = remove_consecutive_note_edges(edge_prob_dict, node_list)
    else:
        edge_prob_dict = remove_impossible_note_edges(edge_prob_dict, node_list)

    # Order edge tuples based on the global start node. global start node becomes -1
    edge_prob_dict_new = {}
    for edge_tuple, probability in edge_prob_dict.items():
        new_key = (-1, edge_tuple[0]) if start_idx in edge_tuple else edge_tuple
        edge_prob_dict_new[new_key] = probability

    # edge_prob_dict = {key: value for key, value in edge_prob_dict_new.items()
    #                   if key[0] in node_list and key[1] in node_list}

    # Remove edges that conflict with a new edge, which starts at start and ends at end
    def filter_dict(edge_prob_dict, new_edge_start, new_edge_end):
        edge_prob_dict_filtered = {
            edge_tuple: probability
            for edge_tuple, probability in edge_prob_dict.items()
            if not ((edge_tuple[0] < new_edge_start < edge_tuple[1] < new_edge_end)
                    or (new_edge_start < edge_tuple[0] < new_edge_end < edge_tuple[1]))}
        edge_prob_dict_filtered.pop((new_edge_start, new_edge_end), None)
        return edge_prob_dict_filtered

    pick_k_largest_items = lambda dictionary: heapq.nlargest(k, dictionary.items(), key=lambda item: item[1])

    analysis = []
    current_dict = edge_prob_dict_new.copy()

    while current_dict:
        edges_and_probs = pick_k_largest_items(current_dict)
        selected_edge_prob = random.choice(edges_and_probs)[0]
        analysis.append(selected_edge_prob)
        start, end = selected_edge_prob
        current_dict = filter_dict(current_dict, start, end)

    final_analysis = []
    for edge_tuple in analysis:
        final_analysis.append(
            (edge_tuple[1], start_idx) if -1 in edge_tuple else edge_tuple
        )
    return final_analysis


def get_inference(name, pred, edge_index, pos_edge_index, inference_method, print_info: bool = True,
                  voice: str = "bass"):
    if inference_method in ['lr', 'mle_lr']:
        next_dict, prev_dict, global_start_idx = convert_to_dicts(edge_index, pred, name)
        if inference_method == 'lr':
            inferred_edges = infer_edges_beam_lr(next_dict, global_start_idx, beam_width=5)
        elif inference_method == 'mle_lr':
            inferred_edges = infer_edges_beam_mle_lr(next_dict, prev_dict, global_start_idx, beam_width=5)
    else:
        edge_dict, global_start_idx, concurrent_edges = convert_to_dicts_mle(edge_index, pred, name)
        inferred_edges = infer_edges_mle(edge_dict, global_start_idx, concurrent_edges, False)
    print("---------------", name, "---------------")
    positive_edge_list = torch.unique(pos_edge_index).tolist()
    positive_edge_list = [
        idx for idx in positive_edge_list
        if idx not in [global_start_idx, global_start_idx + 2]
    ]

    if print_info:
        print("---------------", name, "---------------")
        print(f"---{voice.upper()} GROUND TRUTH---")
        print(positive_edge_list)
        print(f"-----{voice.upper()} INFERRED-----")
        print(inferred_edges)

    return inferred_edges, (positive_edge_list == inferred_edges)


def convert_to_dicts(edge_index, preds, name):
    dict_1 = {}
    dict_2 = {}
    file_name = name.split('\\')[-1]

    with open(Path(f'../inference/overlapping_edges/{file_name}.pkl'), 'rb') as f:
        forbidden_edges = pickle.load(f)

    global_start_idx = torch.max(edge_index).item() - 2

    for i in range(edge_index.shape[1]):
        node_1 = edge_index[0, i].item()
        node_2 = edge_index[1, i].item()

        if (node_1, node_2) in forbidden_edges:
            continue

        log_probability = math.log(preds[i].item())

        if node_2 == global_start_idx:
            node_1, node_2 = node_2, node_1

        if node_1 not in dict_1:
            dict_1[node_1] = []
        if node_2 not in dict_2:
            dict_2[node_2] = []

        dict_1[node_1].append((node_2, log_probability))
        dict_2[node_2].append((node_1, log_probability))
    return dict_1, dict_2, global_start_idx


def convert_to_dicts_mle(edge_index, preds, name):
    edge_dict = {}

    file_name = name.split('\\')[-1]
    with open(Path(f'../inference/overlapping_edges/{file_name}.pkl'), 'rb') as f:
        forbidden_edges = pickle.load(f)

    global_start_idx = torch.max(edge_index).item() - 2

    for i in range(edge_index.shape[1]):
        node_1 = edge_index[0, i].item()
        node_2 = edge_index[1, i].item()

        if (node_1, node_2) in forbidden_edges:
            continue

        log_probability = math.log(preds[i].item())

        if node_2 == global_start_idx:
            node_1, node_2 = node_2, node_1

            node_1 = -1

        edge_dict[(node_1, node_2)] = log_probability
    return edge_dict, global_start_idx, forbidden_edges


def infer_edges(next_dict, prev_dict):
    inferred_edges = []

    def find_largest_edge(edge_dict):
        largest_value = -float('inf')
        best_key = None
        best_value = None
        for key, value in edge_dict.items():
            if value > largest_value:
                largest_value = value
                best_value = key
        return best_value

    start_key = None
    start_value = None
    largest_value = -float('inf')
    for key, values in next_dict.items():
        for value in values:
            if value[1] > largest_value:
                largest_value = value[1]
                start_key = key
                start_value = value[0]

    inferred_edges.append((start_key, start_value))
    left_node = start_key
    right_node = start_value

    while left_node in prev_dict:
        next_edges = {key: value for key, value in prev_dict[left_node]}
        if not next_edges:
            break
        best_value = find_largest_edge(next_edges)
        inferred_edges.append((best_value, left_node))
        left_node = best_value

    while right_node in next_dict:
        next_edges = {key: value for key, value in next_dict[right_node]}
        if not next_edges:
            break
        best_value = find_largest_edge(next_edges)
        inferred_edges.append((right_node, best_value))
        right_node = best_value

    nodes = set()
    for edge in inferred_edges:
        nodes.update(edge)
    return sorted(nodes)


def infer_edges_beam_lr(next_dict, global_start_idx, beam_width):
    all_possible_paths = []

    def find_largest_edges(edge_list, beam_width):
        sorted_edges = sorted(edge_list, key=lambda x: x[1], reverse=True)
        return sorted_edges[:beam_width]

    def build_path(path, log_prob):
        next_edges = find_largest_edges(next_dict[max(path)], beam_width)
        for next_node, log_prob_next in next_edges:
            if next_node == global_start_idx + 2:
                all_possible_paths.append((path, log_prob / len(path)))
                return
            build_path(path + [next_node], log_prob + log_prob_next)

    initial_edges = find_largest_edges(next_dict[global_start_idx], beam_width)
    for start_node, start_prob in initial_edges:
        build_path([start_node], start_prob)

    most_likely_path = max(all_possible_paths, key=lambda x: x[1])
    return most_likely_path[0]


def infer_edges_beam_mle_lr(next_dict, prev_dict, global_start_idx, beam_width):
    all_possible_paths = []

    def find_largest_edges(edge_list, beam_width):
        sorted_edges = sorted(edge_list, key=lambda x: x[1], reverse=True)
        return sorted_edges[:beam_width]

    def build_path(path, log_prob):
        next_edges = find_largest_edges(next_dict[max(path)], beam_width)
        for next_node, log_prob_next in next_edges:
            if next_node == global_start_idx + 2:
                prev_edges = find_largest_edges(prev_dict[min(path)], beam_width)
                for prev_node, log_prob_prev in prev_edges:
                    if prev_node == global_start_idx:
                        all_possible_paths.append((path, log_prob / len(path)))
                        return
                    build_path(path + [prev_node], log_prob + log_prob_prev)
            else:
                build_path(path + [next_node], log_prob + log_prob_next)

    edge_list = [(start, end, log_prob)
                 for start, edges in next_dict.items()
                 for end, log_prob in edges]
    sorted_edges = sorted(edge_list, key=lambda x: x[2], reverse=True)[:beam_width]
    starting_list = []
    for start, end, log_prob in sorted_edges:
        if start == global_start_idx:
            starting_list.append(([end], log_prob))
        elif end == global_start_idx + 2:
            starting_list.append(([start], log_prob))
        else:
            starting_list.append(([start, end], log_prob))

    for path, log_prob in starting_list:
        build_path(path, log_prob)

    most_likely_path = max(all_possible_paths, key=lambda x: x[1])
    return sorted(most_likely_path[0])


def infer_edges_mle(edge_dict, global_start_idx, concurrent_edges, random_choice):
    global_end_idx = global_start_idx + 2

    def filter_edges(edge_dict, edge):
        node_1 = edge[0]
        node_2 = edge[1]
        for forbidden_edge in concurrent_edges:
            if forbidden_edge[0] == node_2:
                edge_dict = {k: v for k, v in edge_dict.items() if not (k[0] == forbidden_edge[1])}
            if forbidden_edge[1] == node_1:
                edge_dict = {k: v for k, v in edge_dict.items() if not (k[1] == forbidden_edge[0])}
        filtered_dict = {key: value for key, value in edge_dict.items()
                         if not (key[0] < node_1 < key[1]
                                 or key[0] < node_2 < key[1]
                                 or node_1 < key[0] < node_2
                                 or node_1 < key[1] < node_2)}
        return filtered_dict

    def continue_search(inferred_list):
        if not inferred_list: return True
        inferred_list.sort(key=lambda x: x[0])
        for i in range(len(inferred_list) - 1):
            if inferred_list[i][1] != inferred_list[i + 1][0]:
                return True
        return False

    def choose_key_random(log_prob_dict):
        keys = list(log_prob_dict.keys())
        log_probs = np.array(list(log_prob_dict.values()))
        probs = np.exp(log_probs)
        probs /= probs.sum()
        chosen_key = np.random.choice(len(keys), p=probs)
        return keys[chosen_key]

    search_start = True
    search_end = True

    inferred_list = []

    while search_start or search_end or continue_search(inferred_list):
        if random_choice:
            edge_to_add = choose_key_random(edge_dict)
        else:
            edge_to_add = max(edge_dict, key=edge_dict.get)
        inferred_list.append(edge_to_add)
        edge_dict = filter_edges(edge_dict, edge_to_add)
        edge_dict.pop(edge_to_add, None)
        search_start = (edge_to_add[0] != -1) and search_start
        search_end = (edge_to_add[1] != global_end_idx) and search_end

    inferred_list.sort(key=lambda x: x[0])
    return [inferred_list[i][1] for i in range(len(inferred_list) - 1)]


def create_edge_dict(pos_edge_index):
    nodes = torch.cat((pos_edge_index[0], pos_edge_index[1])).unique()
    edge_dict = {(start.item(), end.item()): 0 for start in nodes for end in nodes}
    for start, end in zip(pos_edge_index[0], pos_edge_index[1]):
        edge_dict[(start.item(), end.item())] = 1
    return edge_dict


def add_noise(edge_dict, method, noise_scale=0.2, flip_probability=0.1):
    noisy_dict = {}

    if method == 'laplace':
        for edge, value in edge_dict.items():
            noise = torch.distributions.Laplace(0, noise_scale).sample().item()
            noisy_value = max(min(value + noise, 1), 0)
            noisy_dict[edge] = noisy_value

    elif method == 'dirichlet':
        keys, vals = edge_dict.items()
        edge_keys = list(keys)
        edge_values = torch.tensor(list(vals), dtype=torch.float)
        alpha = noise_scale * torch.ones_like(edge_values) + edge_values
        dirichlet_noise = torch.distributions.Dirichlet(alpha).sample()

        for i, edge in enumerate(edge_keys):
            noisy_dict[edge] = min(dirichlet_noise[i].item(), 1)

    elif method == 'flip':
        for edge, value in edge_dict.items():
            noisy_value = value
            if random.random() < flip_probability:
                noisy_value = 1 - value
            noisy_dict[edge] = noisy_value

    return noisy_dict


def get_noisy_ground_truth(
        pos_edge_index_treble,
        pos_edge_index_bass,
        noise_scale=0.1,
        method='laplace',
        flip_probability=0.1
):
    edge_dict_treble = create_edge_dict(pos_edge_index_treble)
    edge_dict_bass = create_edge_dict(pos_edge_index_bass)

    edge_dict_treble_noisy = add_noise(edge_dict_treble, noise_scale, method, flip_probability=flip_probability)
    edge_dict_bass_noisy = add_noise(edge_dict_bass, noise_scale, method, flip_probability=flip_probability)

    return edge_dict_treble_noisy, edge_dict_bass_noisy


if __name__ == "__main__":
    from pprint import pprint
    edge_prob_dict = {(9, 12): 1.0, (9, 13): 1.0, (8, 9): 0.979, (3, 10): 0.973, (3, 9): 0.947, (5, 7): 0.946,
                      (9, 11): 0.946, (1, 10): 0.941, (6, 7): 0.937, (1, 3): 0.861, (3, 7): 0.845, (5, 6): 0.817,
                      (6, 9): 0.78, (2, 3): 0.756, (6, 8): 0.715, (1, 2): 0.713, (7, 8): 0.665, (3, 5): 0.54,
                      (7, 11): 0.455, (6, 11): 0.432, (7, 9): 0.396, (3, 6): 0.391, (5, 8): 0.356, (8, 11): 0.338,
                      (8, 12): 0.249, (1, 7): 0.186, (1, 5): 0.162, (3, 8): 0.129, (3, 11): 0.077, (1, 6): 0.068,
                      (2, 7): 0.058, (5, 9): 0.049, (5, 11): 0.047, (2, 5): 0.026, (2, 6): 0.012, (1, 11): 0.011,
                      (8, 13): 0.01, (1, 9): 0.005, (1, 8): 0.003, (0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, (0, 4): 0.0,
                      (0, 5): 0.0, (0, 6): 0.0, (0, 7): 0.0, (0, 8): 0.0, (0, 9): 0.0, (0, 10): 0.0, (0, 11): 0.0,
                      (0, 12): 0.0, (0, 13): 0.0, (1, 4): 0.0, (1, 12): 0.0, (1, 13): 0.0, (2, 4): 0.0, (2, 8): 0.0,
                      (2, 9): 0.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0, (3, 4): 0.0, (3, 12): 0.0,
                      (3, 13): 0.0, (4, 5): 0.0, (4, 6): 0.0, (4, 7): 0.0, (4, 8): 0.0, (4, 9): 0.0, (4, 10): 0.0,
                      (4, 11): 0.0, (4, 12): 0.0, (4, 13): 0.0, (5, 10): 0.0, (5, 12): 0.0, (5, 13): 0.0, (6, 10): 0.0,
                      (6, 12): 0.0, (6, 13): 0.0, (7, 10): 0.0, (7, 12): 0.0, (7, 13): 0.0, (8, 10): 0.0, (9, 10): 0.0}
    noisy = add_noise(edge_prob_dict, method='laplace', noise_scale=0.5, flip_probability=0.5)
    pprint(noisy)
    # node_list = [1, 5, 8, 9, 10, 11]
    # sample_analysis_voice("bass", node_list, edge_prob_dict, 3)
    # remove_consecutive_note_edges(edge_prob_dict, node_list)


# OLD METHOD FOR WITHOUT GLOBAL NODES
# def infer_edges_beam(next_dict, prev_dict, beam_width=3):
#     all_possible_paths = []

#     def find_largest_edges(edge_list, beam_width=3):
#         sorted_edges = sorted(edge_list, key=lambda x: x[1], reverse=True)
#         return sorted_edges[:beam_width]

#     def build_path(path, log_prob):
#         left_node = min(path)
#         right_node = max(path)

#         if left_node not in prev_dict and right_node not in next_dict:
#             all_possible_paths.append((tuple(sorted(path)), log_prob)) //wrong

#         while left_node in prev_dict:
#             next_edges = prev_dict[left_node]
#             best_values = find_largest_edges(next_edges, beam_width)
#             for best_value, prob in best_values:
#                 new_path = [best_value] + path
#                 new_log_prob = log_prob + prob
#                 build_path(new_path, new_log_prob)
#             break

#         while right_node in next_dict:
#             next_edges = next_dict[right_node]
#             best_values = find_largest_edges(next_edges, beam_width)
#             for best_value, prob in best_values:
#                 new_path = path + [best_value]
#                 new_log_prob = log_prob + prob
#                 build_path(new_path, new_log_prob)
#             break

#     start_points = []
#     for key, values in next_dict.items():
#         for value in values:
#             start_points.append((key, value[0], value[1]))
#     start_points = sorted(start_points, key=lambda x: x[2], reverse=True)[:beam_width]

#     for start_key, start_value, start_log_prob in start_points:
#         initial_path = [start_key, start_value]
#         build_path(initial_path, start_log_prob)

#     best_path = max(all_possible_paths, key=lambda x: x[1])[0]
#     return list(best_path)
