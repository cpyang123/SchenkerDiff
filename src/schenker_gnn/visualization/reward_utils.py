import os
import pickle
import random

from inference.inference_utils import sample_analysis, add_noise


class MismatchedDepthException(Exception):
    def __init__(self, message):
        super().__init__(message)


def create_synthetic_reward_training_data():
    directory = "../schenkerian_clusters"
    items = os.listdir(directory)
    # Filter out only the directories
    folders = [
        item for item in items
        if os.path.isdir(os.path.join(directory, item))
           and item not in ['mxls', '__pycache__']
    ]
    for folder_name in folders:
        try:
            treble_edges, bass_edges, node_voices = gather_ground_truth_analysis(folder_name, directory)
        except (FileNotFoundError, MismatchedDepthException):
            continue
        treble_prob_dict, bass_prob_dict = convert_analysis_to_prob_dict(treble_edges, bass_edges, node_voices)
        analysis_treble1, analysis_bass1 = sample_analysis(
            node_voices, treble_prob_dict, bass_prob_dict, k=1, remove_consecutive=False, reward_model=True
        )

        treble_prob_dict_noisy = add_noise(treble_prob_dict, 'flip', noise_scale=0.5, flip_probability=0.8)
        bass_prob_dict_noisy = add_noise(bass_prob_dict, 'flip', noise_scale=0.5, flip_probability=0.8)
        node_voices_noisy = get_noisy_node_voices(node_voices)
        analysis_treble2, analysis_bass2 = sample_analysis(
            node_voices_noisy, treble_prob_dict_noisy, bass_prob_dict_noisy, k=3, remove_consecutive=False, reward_model=True
        )
        reward_pair = [{
            'analysis_treble1': analysis_treble1,
            'analysis_bass1': analysis_bass1,
            'analysis_treble2': analysis_treble2,
            'analysis_bass2': analysis_bass2,
            'node_list1': node_voices,
            'node_list2': node_voices.copy(),
            'preference': 1
        }]
        with open(f"./reward_model_preference_data/{folder_name}_randid{random.randint(1, 999999)}.pkl", 'wb') as f:
            pickle.dump(reward_pair, f)


def get_noisy_node_voices(node_voices):
    treble = node_voices[0][-2:]
    bass = node_voices[1][-2:]
    inner = []

    max_note = max(treble) - 2
    treble += random.sample(range(max_note), random.randint(max(2, max_note-4), max_note))
    bass += random.sample(range(max_note), random.randint(max(2, max_note-4), max_note))

    return [sorted(treble), sorted(bass), inner]


def convert_analysis_to_prob_dict(treble_edges, bass_edges, node_voices) -> tuple[dict[tuple:float], dict[tuple:float]]:
    treble_nodes = node_voices[0]
    bass_nodes = node_voices[1]
    treble_prob_dict = {
        (node_i, node_j): 0.0
        for node_i in range(bass_nodes[-1] - 3)
        for node_j in range(node_i, bass_nodes[-1] + 1)
        if node_j > node_i
    }
    bass_prob_dict = {
        (node_i, node_j): 0.0
        for node_i in range(bass_nodes[-1] - 3)
        for node_j in range(node_i, bass_nodes[-1] + 1)
        if node_j > node_i
    }
    for edge in treble_edges:
        treble_prob_dict[edge] = 1.0
    for edge in bass_edges:
        bass_prob_dict[edge] = 1.0
    return treble_prob_dict, bass_prob_dict


def gather_ground_truth_analysis(folder_name, directory):
    path = os.path.join(directory, folder_name, folder_name + '.pkl')
    with open(path, 'rb') as f:
        raw_edges = pickle.load(f)

    node_voices = gather_node_voices(raw_edges)
    treble_edges, bass_edges = gather_treble_bass_edges(raw_edges, node_voices)

    return treble_edges, bass_edges, node_voices


def gather_treble_bass_edges(raw_edges, node_voices):
    global_treble_start, global_treble_end = max(node_voices[0]) - 1, max(node_voices[0])
    global_bass_start, global_bass_end = max(node_voices[1]) - 1, max(node_voices[1])
    highest_depth_treble = len(raw_edges['t_edges']) - 1
    highest_depth_bass = len(raw_edges['b_edges']) - 1

    if highest_depth_bass != highest_depth_treble:
        raise MismatchedDepthException("Bass and treble depth don't match")

    treble_flattened = [tuple(edge) for depth_edges in raw_edges['t_edges'] for edge in depth_edges]
    treble_flattened += [
        (global_treble_start, raw_edges['t_edges'][depth][0][0])
        for depth in range(len(raw_edges['t_edges']))
    ] + [
        (raw_edges['t_edges'][depth][-1][-1], global_treble_end)
        for depth in range(len(raw_edges['t_edges']))
    ]
    treble_edges = list(set(treble_flattened))

    bass_flattened = [tuple(edge) for depth_edges in raw_edges['b_edges'] for edge in depth_edges]
    bass_flattened += [
        (global_bass_start, raw_edges['b_edges'][depth][0][0])
        for depth in range(len(raw_edges['b_edges']))
    ] + [
        (raw_edges['b_edges'][depth][-1][-1], global_bass_end)
        for depth in range(len(raw_edges['t_edges']))
    ]
    bass_edges = list(set(bass_flattened))

    return treble_edges, bass_edges


def gather_node_voices(raw_edges) -> list[list[int]]:
    treble_nodes = set([idx for depth_edges in raw_edges['t_edges'] for edge in depth_edges for idx in edge])
    bass_nodes = set([idx for depth_edges in raw_edges['b_edges'] for edge in depth_edges for idx in edge])

    # Add global nodes
    final_node = max(max(treble_nodes), max(bass_nodes))
    treble_nodes.update([final_node+1, final_node+2])
    bass_nodes.update([final_node+3, final_node+4])

    node_voices = [list(treble_nodes), list(bass_nodes), []]

    return node_voices


if __name__ == "__main__":
    from pprint import pprint
    create_synthetic_reward_training_data()

