import os
import sys

import numpy as np

from config import *
from data_processing import HeteroGraphData
from inference.inference_utils import load_model, get_inference, get_inference_all_depths, sample_analysis
from model.LinkPredictor import LinkPredictor
from model.VoicePredictor import VoicePredictor
from model.schenker_GNN_model import SchenkerGNN
from utils import prepare_model, prepare_data_loader, calculate_loss_lp, predict_layer


def test_loop(
        model,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        samples
):
    model.eval()
    link_predictor_treble.eval()
    link_predictor_bass.eval()
    same_inference = 0
    total_inference = 0
    with torch.no_grad():
        test_loss = []
        for name, data, \
            pos_edge_index_treble, neg_edge_index_treble, \
            pos_edge_index_bass, neg_edge_index_bass, \
            all_edges_index_treble, all_edges_index_bass in zip(
            samples['name'],
            samples['data'],
            samples['positive_edge_indices_treble'],
            samples['negative_edge_indices_treble'],
            samples['positive_edge_indices_bass'],
            samples['negative_edge_indices_bass'],
            samples['all_edges_indices_treble'],
            samples['all_edges_indices_bass']
        ):
            final_embedding = model(data)
            try:
                voice_pred = voice_predictor(final_embedding)
                final_embedding = torch.cat((final_embedding, voice_pred), dim=1)
                pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass = predict_layer(
                    final_embedding,
                    link_predictor_treble, link_predictor_bass,
                    pos_edge_index_treble, neg_edge_index_treble,
                    pos_edge_index_bass, neg_edge_index_bass
                )

                all_pred_treble = link_predictor_treble(final_embedding[all_edges_index_treble[0]],
                                                        final_embedding[all_edges_index_treble[1]])
                all_pred_bass = link_predictor_bass(final_embedding[all_edges_index_bass[0]],
                                                    final_embedding[all_edges_index_bass[1]])

                human_readable_treble = {
                    (from_node.item(), to_node.item()): round(all_pred_treble[i].item(), 3)
                    for i, (from_node, to_node) in enumerate(zip(all_edges_index_treble[0], all_edges_index_treble[1]))
                }
                human_readable_treble = dict(
                    sorted(human_readable_treble.items(), key=lambda item: item[1], reverse=True))
                human_readable_bass = {
                    (from_node.item(), to_node.item()): round(all_pred_bass[i].item(), 3)
                    for i, (from_node, to_node) in enumerate(zip(all_edges_index_bass[0], all_edges_index_bass[1]))
                }
                human_readable_bass = dict(sorted(human_readable_bass.items(), key=lambda item: item[1], reverse=True))

                print(name)
                print(human_readable_treble)
                print(human_readable_bass)
                if not ALL_DEPTHS_AT_ONCE:
                    inference_method = 'mle'
                    treble_inference, treble_match = get_inference(
                        name, all_pred_treble, all_edges_index_treble, pos_edge_index_treble, inference_method,
                        print_info=True, voice="treble"
                    )
                    bass_inference, bass_match = get_inference(
                        name, all_pred_bass, all_edges_index_bass, pos_edge_index_bass, inference_method,
                        print_info=True, voice="bass"
                    )

                    same_inference += int(treble_match) + int(bass_match)
                    total_inference += 2
                else:
                    all_treble = get_inference_all_depths(name, human_readable_treble, 0, 3)
                    all_bass = get_inference_all_depths(name, human_readable_bass, 1, 1)
            except IndexError as e:
                continue
            loss = calculate_loss_lp(pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass)
            test_loss.append(loss.item())
    return np.mean(test_loss), same_inference, total_inference


if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)

    model, _, _ = prepare_model(
        model_class=SchenkerGNN,
        num_features=NUM_FEATURES,
        embedding_dim=GNN_EMB_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        output_dim=GNN_OUTPUT_DIM,
        num_layers=GNN_NUM_LAYERS,
        dropout=DROPOUT_PERCENT,
        diff_dim=DIFF_POOL_EMB_DIM,
        diff_node_num=DIFF_NODE_NUM,
        device=DEVICE
    )
    link_predictor_treble = LinkPredictor(
        in_channels=GNN_OUTPUT_DIM + VOICE_PRED_NUM_CLASSES,
        hidden_channels=LINK_PRED_HIDDEN_DIM,
        out_channels=1,
        num_layers=LINK_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )
    link_predictor_bass = LinkPredictor(
        in_channels=GNN_OUTPUT_DIM + VOICE_PRED_NUM_CLASSES,
        hidden_channels=LINK_PRED_HIDDEN_DIM,
        out_channels=1,
        num_layers=LINK_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )
    voice_predictor = VoicePredictor(
        in_channels=GNN_OUTPUT_DIM,
        hidden_channels=VOICE_PRED_HIDDEN_DIM,
        out_channels=3,
        num_layers=VOICE_PRED_NUM_LAYERS,
        dropout=DROPOUT_PERCENT
    )

    test_samples = prepare_data_loader(
        "../" + TEST_NAMES, TEST_SAVE_FOLDER, 0, HeteroGraphData, test_mode=True
        # TEST_NAMES, TEST_SAVE_FOLDER, 0, HeteroGraphData, test_mode=True
    )

    gnn_save_path = f"../saved_models/GNN_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    lp_treble_save_path = f"../saved_models/LP_treble_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    lp_bass_save_path = f"../saved_models/LP_bass_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    vp_save_path = f"../saved_models/VP_input{GNN_OUTPUT_DIM}_hidden{VOICE_PRED_HIDDEN_DIM}_layers{VOICE_PRED_NUM_LAYERS}"

    # model, link_predictor = load_model(model, link_predictor, DEVICE, "../" + gnn_save_path, "../" + lp_save_path)
    model, link_predictor_treble, link_predictor_bass, voice_predictor = load_model(
        model,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        DEVICE,
        gnn_save_path,
        lp_treble_save_path,
        lp_bass_save_path,
        vp_save_path
    )

    test_loss, same_inference, total_inference = test_loop(
        model, link_predictor_treble, link_predictor_bass, voice_predictor, test_samples
    )
    print(f'Test Loss: {test_loss:.4f}')
    print(f"Hit the same inference", same_inference, "times out of a total of", total_inference)
