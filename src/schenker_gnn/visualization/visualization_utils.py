from copy import deepcopy

from config import *
from data_processing import HeteroGraphData
from inference.inference_utils import load_model, get_node_list, sample_analysis
from model.LinkPredictor import LinkPredictor
from model.RewardModel import RewardGNN
from model.VoicePredictor import VoicePredictor
from model.schenker_GNN_model import SchenkerGNN
from pyScoreParser.musicxml_parser.mxp import MusicXMLDocument
from utils import prepare_model
from visualization.styles import *


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

    gnn_save_path = f"{'../' if not include_reward else ''}saved_models/GNN_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    lp_treble_save_path = f"{'../' if not include_reward else ''}saved_models/LP_treble_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    lp_bass_save_path = f"{'../' if not include_reward else ''}saved_models/LP_bass_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    vp_save_path = f"{'../' if not include_reward else ''}saved_models/VP_input{GNN_OUTPUT_DIM}_hidden{VOICE_PRED_HIDDEN_DIM}_layers{VOICE_PRED_NUM_LAYERS}"

    reward_save_path = None
    if include_reward:
        reward_save_path = f"saved_models/reward_emb{REWARD_EMB_DIM}_hidden{REWARD_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{REWARD_NUM_LAYERS}"

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
    graph = data['data']
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


def load_graph_nodes(xml_filename: str, id=''):
    xml = MusicXMLDocument(xml_filename)
    pyscoreparser_notes = xml.get_notes()
    nodes = [
        {
            'data': {
                'id': str(i) + id,
                'label': f'{note.pitch[0]} ({i})',
                'midi': note.pitch[1],
                'time': note.state_fixed.time_position
            },
            'position': {
                'x': 100 * note.state_fixed.time_position,
                'y': HEIGHT - 10 * note.pitch[1]
            },
            'grabbable': False
        }
        for i, note in enumerate(pyscoreparser_notes)
    ]
    return nodes, pyscoreparser_notes


def load_graph_edges(pyscoreparser_notes):
    edges = []
    edge_index = [[], []]
    for i, note_i in enumerate(pyscoreparser_notes):
        for j, note_j in enumerate(pyscoreparser_notes[i + 1:], start=i + 1):
            if note_i.state_fixed.time_position != note_j.state_fixed.time_position:
                edges.append({
                    'data': {'id': f'{str(i)}_{str(j)}_treble', 'source': str(i), 'target': str(j), 'isCheckable': True}
                })
                edges.append({
                    'data': {'id': f'{str(i)}_{str(j)}_bass', 'source': str(i), 'target': str(j), 'isCheckable': True}
                })
                edge_index[0].append(i)
                edge_index[1].append(j)
        # Include all global edges
        for global_i in range(len(pyscoreparser_notes), len(pyscoreparser_notes) + 4):
            edge_index[0].append(i)
            edge_index[1].append(global_i)
    edge_index = torch.tensor(edge_index)
    print(edge_index)
    return edges, edge_index


def load_graph_edges_from_sampled_analysis(analysis_treble, analysis_bass, num_notes, id):
    has_global_idx = lambda x: x[0] not in range(num_notes) or x[1] not in range(num_notes)
    edges = []
    for edge in analysis_treble:
        if has_global_idx(edge):
            continue
        edges.append({
            'data': {
                'id': f'{edge[0]}_{edge[1]}_treble', 'source': f'{edge[0]}{id}', 'target': f'{edge[1]}{id}'
            }
        })
        edges.append({
            'data': {
                'id': f'{edge[0]}_{edge[1]}_fake_treble', 'source': f'{edge[0]}{id}', 'target': f'{edge[1]}{id}'
            }
        })
        DEFAULT_STYLE_SHEET_RLHF.append({
            'selector': f'edge[id="{edge[0]}_{edge[1]}_fake_treble"]',
            'style': {'opacity': 0.0}
        })
    for edge in analysis_bass:
        if has_global_idx(edge):
            continue
        edges.append({
            'data': {
                'id': f'{edge[0]}_{edge[1]}_fake_bass', 'source': f'{edge[0]}{id}', 'target': f'{edge[1]}{id}'
            }
        })
        edges.append({
            'data': {
                'id': f'{edge[0]}_{edge[1]}_bass', 'source': f'{edge[0]}{id}', 'target': f'{edge[1]}{id}'
            }
        })
        DEFAULT_STYLE_SHEET_RLHF.append({
            'selector': f'edge[id="{edge[0]}_{edge[1]}_fake_bass"]',
            'style': {'opacity': 0.0}
        })
    return edges


def append_voice_probabilities(nodes, voice_preds):
    for i, node in enumerate(nodes):
        node["treble prob"] = round(voice_preds[i, 0].item(), 3)
        node["bass prob"] = round(voice_preds[i, 1].item(), 3)
        node["inner prob"] = round(voice_preds[i, 2].item(), 3)
    return nodes


def append_edge_probabilities(edges, treble_preds, bass_preds):
    for edge in edges:
        translated_key = (int(edge['data']['source']), int(edge['data']['target']))
        if 'treble' in edge['data']['id']:
            edge['data']['label'] = str(treble_preds[translated_key])
        elif 'bass' in edge['data']['id']:
            edge['data']['label'] = str(bass_preds[translated_key])
    return edges


def create_node_grid_df(nodes):
    columnDefs = [
        {"field": col}
        for col in ["Identifier", "Note Name", "Midi", "Time", "Treble Pred", "Bass Pred", "Inner Pred"]
    ]
    columnDefs.append({
        "field": "Voice",
        "editable": True,
        "cellEditor": "agSelectCellEditor",
        "cellEditorParams": {
            "values": ["Undecided", "Treble", "Inner", "Bass", "Both"]
        },
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value === 'Treble'",
                    "style": {"backgroundColor": NODE_BORDER_COLOR_TREBLE},
                },
                {
                    "condition": "params.value === 'Bass'",
                    "style": {"backgroundColor": NODE_BORDER_COLOR_BASS},
                },
                {
                    "condition": "params.value === 'Both'",
                    "style": {"backgroundColor": NODE_BORDER_COLOR_BOTH}
                },
                {
                    "condition": "params.value === 'Inner'",
                    "style": {"backgroundColor": NODE_BORDER_COLOR_INNER}
                }
            ],
            "defaultStyle": {"backgroundColor": NODE_BORDER_COLOR_UNDECIDED}
        }
    })

    rowData = []

    for node in nodes:
        data = node['data']
        rowData.append({
            "Identifier": data['id'],
            "Note Name": data['label'],
            "Midi": data['midi'],
            "Time": data['time'],
            "Voice": "Undecided",
            "Treble Pred": node['treble prob'],
            "Bass Pred": node['bass prob'],
            "Inner Pred": node['inner prob']
        })

    return columnDefs, rowData


def create_edge_grid_df(edges):
    columnDefs = [
        {
            "field": "Identifier",
            "checkboxSelection": {
                "function": "params['data']['isCheckable'].toString() === 'true'"
            }
        },
        {"field": "Source", "filter": "agTextColumnFilter"},
        {"field": "Target", "filter": "agTextColumnFilter"},
        {"field": "Voice"},
        {"field": "Prediction Probability"},
        {"field": "isCheckable"}
    ]

    rowData = []
    for edge in edges:
        data = edge['data']
        voice = "treble" if "treble" in data['id'] else "bass"
        rowData.append({
            "Identifier": data['id'],
            "Source": data['source'],
            "Target": data['target'],
            "Voice": voice,
            "Prediction Probability": float(data['label'])
        })
    return columnDefs, rowData


def add_voice_info_to_nodes(node_list, id, stylesheet=DEFAULT_STYLE_SHEET_RLHF):
    max_idx = max([j for i in node_list for j in i])
    for i in range(max_idx + 1):
        if i in node_list[0] and i in node_list[1]:
            color = NODE_BORDER_COLOR_BOTH
        elif i in node_list[0]:
            color = NODE_BORDER_COLOR_TREBLE
        elif i in node_list[1]:
            color = NODE_BORDER_COLOR_BASS
        elif i in node_list[2]:
            color = NODE_BORDER_COLOR_INNER
        else:
            color = NODE_BORDER_COLOR_UNDECIDED
        stylesheet.append({
            'selector': f'node[id="{i}{id}"]',
            'style': {
                'border-color': color,
                "border-opacity": "1",
                "border-width": "2px"
            }
        })


def load_score(music_name, node_sample_method="threshold"):
    # Define nodes and edges
    xml_filename = f"../schenkerian_clusters/{music_name}/{music_name}.xml"
    nodes1, pyscoreparser_notes = load_graph_nodes(xml_filename, 'left')
    nodes2, _ = load_graph_nodes(xml_filename, 'right')

    # Load all models
    gnn, lp_treble, lp_bass, voice_model, *_ = load_all_models()
    _, edge_index = load_graph_edges(pyscoreparser_notes)

    # Make predictions
    name, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities_visualization(
        gnn, lp_treble, lp_bass, voice_model, xml_filename, edge_index
    )

    # Sample voice indices
    node_list1 = get_node_list(human_readable_voice, version=node_sample_method)
    node_list2 = get_node_list(human_readable_voice, version=node_sample_method)
    # Sample edges based on voice indices
    analysis_treble1, analysis_bass1 = sample_analysis(node_list1, human_readable_treble, human_readable_bass, k=3)
    analysis_treble2, analysis_bass2 = deepcopy(analysis_treble1), deepcopy(analysis_bass1)
    while set(analysis_treble1) == set(analysis_treble2) and set(analysis_bass1) == set(analysis_bass2):
        analysis_treble2, analysis_bass2 = sample_analysis(node_list2, human_readable_treble, human_readable_bass, k=3)

    edges1 = load_graph_edges_from_sampled_analysis(analysis_treble1, analysis_bass1, len(pyscoreparser_notes), 'left')
    edges2 = load_graph_edges_from_sampled_analysis(analysis_treble2, analysis_bass2, len(pyscoreparser_notes), 'right')

    stylesheet1 = DEFAULT_STYLE_SHEET_RLHF.copy()
    stylesheet2 = DEFAULT_STYLE_SHEET_RLHF.copy()

    add_voice_info_to_nodes(node_list1, 'left', stylesheet1)
    add_voice_info_to_nodes(node_list2, 'right', stylesheet2)

    return nodes1, edges1, \
           nodes2, edges2, \
           analysis_treble1, analysis_bass1, node_list1, \
           analysis_treble2, analysis_bass2, node_list2, \
           stylesheet1, stylesheet2
