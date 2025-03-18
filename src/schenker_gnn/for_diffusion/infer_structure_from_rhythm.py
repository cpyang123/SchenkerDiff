import heapq
import random

import numpy as np
import torch

from src.schenker_gnn.config import *
from src.schenker_gnn.data_processing import HeteroGraphData
from src.schenker_gnn.inference.inference_utils import load_model
from src.schenker_gnn.model.LinkPredictor import LinkPredictor
from src.schenker_gnn.model.RewardModel import RewardGNN
from src.schenker_gnn.model.VoicePredictor import VoicePredictor
from src.schenker_gnn.model.schenker_GNN_model import SchenkerGNN
from src.pyScoreParser.musicxml_parser import MusicXMLDocument
from src.schenker_gnn.utils import prepare_model


def load_all_models(include_reward=False):
    gnn, optimizer_gnn, scheduler_gnn = prepare_model(
        model_class=SchenkerGNN,
        num_feature=NUM_FEATURES,
        embedding_dim=GNN_EMB_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        output_dim=GNN_OUTPUT_DIM,
        num_layers=GNN_NUM_LAYERS,
        dropout=DROPOUT_PERCENT,
        diff_dim=DIFF_POOL_EMB_DIM,
        diff_node_num=DIFF_NODE_NUM,
        device=DEVICE,
        ablations=ABLATIONS
    )
    link_predictor_treble, optimizer_lp_treble, scheduler_lp_treble = prepare_model(
        model_class=LinkPredictor,
        in_channels=GNN_OUTPUT_DIM + VOICE_PRED_NUM_CLASSES if ABLATIONS['voice_concat'] else GNN_OUTPUT_DIM,
        hidden_channels=LINK_PRED_HIDDEN_DIM,
        out_channels=1,
        num_layers=LINK_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )
    link_predictor_bass, optimizer_lp_bass, scheduler_lp_bass = prepare_model(
        model_class=LinkPredictor,
        in_channels=GNN_OUTPUT_DIM + VOICE_PRED_NUM_CLASSES if ABLATIONS['voice_concat'] else GNN_OUTPUT_DIM,
        hidden_channels=LINK_PRED_HIDDEN_DIM,
        out_channels=1,
        num_layers=LINK_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )
    voice_predictor, optimizer_voice, scheduler_voice = prepare_model(
        model_class=VoicePredictor,
        in_channels=GNN_OUTPUT_DIM,
        hidden_channels=VOICE_PRED_HIDDEN_DIM,
        out_channels=3,
        num_layers=VOICE_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )

    reward_predictor, optimizer_reward, scheduler_reward = None, None, None
    if include_reward:
        reward_predictor, optimizer_reward, scheduler_reward = prepare_model(
            model_class=RewardGNN,
            learning_rate=LEARNING_RATE,
            num_feature=NUM_FEATURES,
            embedding_dim=REWARD_EMB_DIM,
            hidden_dim=REWARD_HIDDEN_DIM,
            output_dim=REWARD_OUT_DIM,
            num_layers=REWARD_NUM_LAYERS,
            dropout=DROPOUT_PERCENT,
            diff_dim=DIFF_POOL_EMB_DIM,
            diff_node_num=DIFF_NODE_NUM,
            device=DEVICE
        )

    gnn_save_path = f"{'../../../SchenkerDiff/src/schenker_gnn/' if not include_reward else ''}saved_models/GNN_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    lp_treble_save_path = f"{'../../../SchenkerDiff/src/schenker_gnn/' if not include_reward else ''}saved_models/LP_treble_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    lp_bass_save_path = f"{'../../../SchenkerDiff/src/schenker_gnn/' if not include_reward else ''}saved_models/LP_bass_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    vp_save_path = f"{'../../../SchenkerDiff/src/schenker_gnn/' if not include_reward else ''}saved_models/VP_input{GNN_OUTPUT_DIM}_hidden{VOICE_PRED_HIDDEN_DIM}_layers{VOICE_PRED_NUM_LAYERS}"

    reward_save_path = None
    if include_reward:
        reward_save_path = f"saved_models/reward_emb{REWARD_EMB_DIM}_hidden{REWARD_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{REWARD_NUM_LAYERS}"
    # print(DEVICE)
    gnn, link_predictor_treble, link_predictor_bass, voice_predictor, reward_predictor = load_model(
        gnn,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        DEVICE,
        gnn_save_path,
        lp_treble_save_path,
        lp_bass_save_path,
        vp_save_path,
        reward_model=reward_predictor,
        reward_save_path=reward_save_path
    )

    return gnn, link_predictor_treble, link_predictor_bass, voice_predictor, reward_predictor, \
           optimizer_gnn, optimizer_lp_treble, optimizer_lp_bass, optimizer_voice, optimizer_reward, \
           scheduler_gnn, scheduler_lp_treble, scheduler_lp_bass, scheduler_voice, scheduler_reward

def predict_link_probabilities_visualization(gnn, lp_treble, lp_bass, voice_model, xml_filename, edge_indices):
    gnn.eval()
    lp_bass.eval()
    lp_treble.eval()

    data = HeteroGraphData.process_file_for_GUI(xml_filename, include_global_nodes=INCLUDE_GLOBAL_NODES)
    name = data['name']
    graph = data['data'].to(DEVICE)
    if ABLATIONS['voice_given']:
        voice = data['voice']
    else:
        voice = None

    return predict_link_probabilities(name, graph, voice, gnn, lp_treble, lp_bass, voice_model, edge_indices)


def predict_link_probabilities(name, graph, voice, gnn, lp_treble, lp_bass, voice_model, edge_indices):
    final_embedding = gnn(graph)
    # print(final_embedding.shape)

    if ABLATIONS['voice_given']:
        voice_pred = voice
    else:
        voice_model.eval()
        voice_pred = voice_model(final_embedding)
    human_readable_voice = torch.round(voice_pred, decimals=2)

    if ABLATIONS['voice_concat']:
        final_embedding = torch.cat((final_embedding, voice_pred), dim=1)

    all_pred_treble = lp_treble(final_embedding[edge_indices[0]],
                                final_embedding[edge_indices[1]])
    all_pred_bass = lp_bass(final_embedding[edge_indices[0]],
                            final_embedding[edge_indices[1]])

    human_readable_treble = {
        (from_node.item(), to_node.item()): round(all_pred_treble[i].item(), 3)
        for i, (from_node, to_node) in enumerate(zip(edge_indices[0], edge_indices[1]))
    }
    human_readable_treble = dict(sorted(human_readable_treble.items(), key=lambda item: item[1], reverse=True))

    human_readable_bass = {
        (from_node.item(), to_node.item()): round(all_pred_bass[i].item(), 3)
        for i, (from_node, to_node) in enumerate(zip(edge_indices[0], edge_indices[1]))
    }
    human_readable_bass = dict(sorted(human_readable_bass.items(), key=lambda item: item[1], reverse=True))
    # print("human treble:", human_readable_treble)
    # pprint(human_readable_bass)
    return name, human_readable_treble, human_readable_bass, human_readable_voice


def load_graph_edges(pyscoreparser_notes):
    edge_index = [[], []]
    for i, note_i in enumerate(pyscoreparser_notes):
        for j, note_j in enumerate(pyscoreparser_notes[i + 1:], start=i + 1):
            if note_i.state_fixed.time_position != note_j.state_fixed.time_position:
                edge_index[0].append(i)
                edge_index[1].append(j)
        # Include all global edges
        for global_i in range(len(pyscoreparser_notes), len(pyscoreparser_notes) + 4):
            edge_index[0].append(i)
            edge_index[1].append(global_i)
    edge_index = torch.tensor(edge_index, device=DEVICE)
    return edge_index


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


def sample_analysis_voice(voice, node_list, edge_prob_dict, k, name=None, remove_consecutive=True, reward_model=False):
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


def load_score(xml_filename, node_sample_method="threshold"):
    # Define nodes and edges
    # xml_filename = f"../schenkerian_clusters/{music_name}/{music_name}.xml"
    xml = MusicXMLDocument(xml_filename)
    pyscoreparser_notes = xml.get_notes()

    # Load all models
    gnn, lp_treble, lp_bass, voice_model, *_ = load_all_models()
    edge_index = load_graph_edges(pyscoreparser_notes)

    # Make predictions
    name, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities_visualization(
        gnn, lp_treble, lp_bass, voice_model, xml_filename, edge_index
    )

    # Sample voice indices
    node_list1 = get_node_list(human_readable_voice, version=node_sample_method)
    # Sample edges based on voice indices
    analysis_treble1, analysis_bass1 = sample_analysis(node_list1, human_readable_treble, human_readable_bass, k=3, remove_consecutive=False)


    return analysis_treble1, analysis_bass1, node_list1


def extract_structure_adjacency_matrix(analysis_treble, analysis_bass, node_list):
    max_idx = max([i for l in node_list for i in l])
    n = max_idx - 3
    adjacency_mat = torch.zeros((n, n))
    for edge in analysis_treble:
        if edge[0] >= n or edge[1] >= n: continue
        # if abs(edge[0] - edge[1]) <= 1: continue
        adjacency_mat[edge[0], edge[1]] += 1
    for edge in analysis_bass:
        if edge[0] >= n or edge[1] >= n: continue
        # if abs(edge[0] - edge[1]) <= 1: continue
        if adjacency_mat[edge[0], edge[1]] == 1: continue
        adjacency_mat[edge[0], edge[1]] += 1
    adjacency_mat = adjacency_mat + adjacency_mat.T
    return adjacency_mat

def extract_structure_sparse(analysis_treble, analysis_bass, node_list):
    max_idx = max(i for l in node_list for i in l)
    n = max_idx - 3
    directed_edges = {}

    for edge in analysis_treble:
        if edge[0] >= n or edge[1] >= n:
            continue
        key = (edge[0], edge[1])
        directed_edges[key] = directed_edges.get(key, 0) + 1

    for edge in analysis_bass:
        if edge[0] >= n or edge[1] >= n:
            continue
        key = (edge[0], edge[1])
        # Only add bass edge if there hasn't already been a treble edge recorded.
        if directed_edges.get(key, 0) == 1:
            continue
        directed_edges[key] = directed_edges.get(key, 0) + 1

    edge_index_list = []
    edge_attr_list = []
    for (i, j), weight in directed_edges.items():
        if weight > 0:
            edge_index_list.append([i, j])
            edge_attr_list.append(weight)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return edge_index, edge_attr


if __name__ == "__main__":
    piece = "Primi_1"
    analysis_treble, analysis_bass, node_list = load_score(f"../schenkerian_clusters/{piece}/{piece}.xml")
    extract_structure_adjacency_matrix(analysis_treble, analysis_bass, node_list)
    print(analysis_treble)
    print(analysis_bass)
    print(node_list)

