from config import *
from inference.inference_utils import get_node_list, sample_analysis_voice, sample_analysis


def get_all_edge_index(voice_gt):
    edge_index = [[],[]]
    num_nodes = voice_gt.shape[0] - 4
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
                edge_index[0].append(i)
                edge_index[1].append(j)
        # Include all global edges
        for global_i in range(num_nodes, num_nodes+4):
            edge_index[0].append(i)
            edge_index[1].append(global_i)
    edge_index = torch.tensor(edge_index)
    return edge_index

def ppo_predict_link_probabilities(name, data, gnn, lp_treble, lp_bass, voice_model, edge_index_treble, edge_index_bass):
    final_embedding = gnn(data)

    voice_model.eval()
    voice_pred = voice_model(final_embedding)
    human_readable_voice = torch.round(voice_pred, decimals=2)

    final_embedding = torch.cat((final_embedding, voice_pred), dim=1)

    # Double check edge indecies
    all_pred_treble = lp_treble(final_embedding[edge_index_treble[0]],
                                final_embedding[edge_index_treble[1]])

    all_pred_bass = lp_bass(final_embedding[edge_index_bass[0]],
                            final_embedding[edge_index_bass[1]])

    human_readable_treble = {
        (from_node.item(), to_node.item()): round(all_pred_treble[i].item(), 3)
        for i, (from_node, to_node) in enumerate(zip(edge_index_treble[0], edge_index_treble[1]))
    }
    human_readable_treble = dict(sorted(human_readable_treble.items(), key=lambda item: item[1], reverse=True))

    human_readable_bass = {
        (from_node.item(), to_node.item()): round(all_pred_bass[i].item(), 3)
        for i, (from_node, to_node) in enumerate(zip(edge_index_bass[0], edge_index_bass[1]))
    }
    human_readable_bass = dict(sorted(human_readable_bass.items(), key=lambda item: item[1], reverse=True))

    print(human_readable_treble)

    # Sample voice indices and generate graph
    node_list = get_node_list(human_readable_voice)

    analysis_treble, analysis_bass = sample_analysis(node_list, human_readable_treble, human_readable_bass, k=3)

    # [TODO] : Double check the datatypes of the analysis, then output the list of probs and edges
    # Use get_node_list from visualization utils to get node list
    # then pass into sample_analysis to get the final graph
    # Then use this to get probabilities from the human readable version
    # the pass these back to get the probabilities
    # also pass them into the reward model

    return name, analysis_treble, analysis_bass


def ppo_sample_analysis(node_list, treble_prob_dict, bass_prob_dict, k):
    if node_list[0] == []:
        treble = [((0,0), 0)]
    else:
        treble = sample_analysis_voice("treble", node_list[0], treble_prob_dict, k)
    if node_list[1] == []:
        bass = [((0,0), 0)]
    else:
        bass = sample_analysis_voice("bass", node_list[1], bass_prob_dict, k)
    return treble, bass



