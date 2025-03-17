import copy
import os
import pickle
import random
import re

import matplotlib.pyplot as plt
import music21.pitch
import numpy as np
from torch_geometric.data import Dataset

import src.schenker_gnn.data_processing as data_processing
from src.schenker_gnn.config import *
from src.schenker_gnn.data_processing import HeteroGraphData


def prepare_train_valid_data_loaders(
        train_names,
        save_folder,
        depth,
        dataset_class=data_processing.HeteroGraphData,
        train_percent=TRAIN_PERCENT,
        include_depth_edges=True
):
    samples = prepare_data_loader(
        train_names,
        save_folder,
        depth,
        dataset_class,
        include_depth_edges=include_depth_edges,
        test_mode=False
    )

    num_samples = len(samples['name'])
    permutation = np.random.permutation(range(num_samples))
    for sample_key, sample_list in samples.items():
        samples[sample_key] = [samples[sample_key][i] for i in permutation]

    num_training_samples = round(train_percent * num_samples)

    training_samples = {
        k: v[:num_training_samples]
        for k, v in samples.items()
    }
    validation_samples = {
        k: v[num_training_samples:]
        for k, v in samples.items()
    }

    return training_samples, validation_samples

def prepare_test_data_loaders(train_names,
        save_folder,
        depth,
        dataset_class=data_processing.HeteroGraphData,
        include_depth_edges=True):
    
    samples = prepare_data_loader(
        train_names,
        save_folder,
        depth,
        dataset_class,
        include_depth_edges=include_depth_edges,
        test_mode=True
    )    
    num_samples = len(samples['name'])
    permutation = np.random.permutation(range(num_samples))
    for sample_key, sample_list in samples.items():
        samples[sample_key] = [samples[sample_key][i] for i in permutation]
    
    test_samples = {
        k: v for k, v in samples.items()
    }
    
    return test_samples


def prepare_data_loaders_reward(comparison_directory):
    samples = {
        "name": [],
        "graph_pair": []
    }

    for pickle_file in os.listdir(comparison_directory):
        name = re.match(r'(.*?)(_rand)', pickle_file).group(1)
        cluster_name = f"schenkerian_clusters/{name}/{name}.xml"

        try:
            graph1 = HeteroGraphData.process_file_for_GUI(cluster_name)['data']
        except data_processing.EnharmonicError:
            continue

        samples['name'].append(name)
        graph2 = copy.deepcopy(graph1)

        pickle_path = os.path.join(comparison_directory, pickle_file)

        with open(pickle_path, "rb") as file:
            data = pickle.load(file)

        graph1_edges = {
            "schenker_treble": torch.tensor(data[0]['analysis_treble1']).T,
            "schenker_bass": torch.tensor(data[0]['analysis_bass1']).T
        }
        graph2_edges = {
            "schenker_treble": torch.tensor(data[0]['analysis_treble2']).T,
            "schenker_bass": torch.tensor(data[0]['analysis_bass2']).T
        }

        preference = data[0]['preference']

        for edge_type in ['schenker_treble', 'schenker_bass']:
            for graph, graph_edges in zip([graph1, graph2], [graph1_edges, graph2_edges]):
                graph[('note', edge_type, 'note')].edge_index = graph_edges[edge_type]

                num_edges = graph[('note', edge_type, 'note')].edge_index.shape[1]
                edge_weights = torch.ones(num_edges)
                graph[('note', edge_type, 'note')].edge_attr = edge_weights

        if preference == 1:
            samples['graph_pair'].append((graph1, graph2))
        else:
            samples['graph_pair'].append((graph2, graph1))

    indices = list(range(len(samples['name'])))
    random.shuffle(indices)
    training_samples = {
        "name": [samples['name'][i] for i in indices[:-20]],
        "graph_pair": [samples['graph_pair'][i] for i in indices[:-20]]
    }
    validation_samples = {
        "name": [samples['name'][i] for i in indices[-20:]],
        "graph_pair": [samples['graph_pair'][i] for i in indices[-20:]]
    }
    return training_samples, validation_samples


def data_loader_all_depths(dataset, depth, include_global_nodes, samples):
    for data in dataset:
        name = data['name']
        voice = data['voice']
        data = data['data']

        for training_voice in ['treble', 'bass']:
            pos_edge_index, important_indices = _get_all_positive_edge_indices(
                data, training_voice, include_global_nodes
            )
            neg_edge_index = _get_negative_edge_index(
                data, training_voice, depth, important_indices, include_global_nodes, pos_edge_index, 'all'
            )

            samples[f'positive_edge_indices_{training_voice}'].append(pos_edge_index)
            samples[f'negative_edge_indices_{training_voice}'].append(neg_edge_index)

            all_edges_index = _get_all_pairs_edge_index(data, important_indices, include_global_nodes)
            samples[f'all_edges_indices_{training_voice}'].append(all_edges_index)

        for d in range(NUM_DEPTHS):
            del data[('note', f'treble_depth{d}', 'note')]
            del data[('note', f'bass_depth{d}', 'note')]

        samples['name'].append(name)
        samples['data'].append(data)
        samples['voice'].append(voice)

    return samples


def prepare_data_loader(
        file_names,
        save_folder,
        depth,
        dataset_class=data_processing.HeteroGraphData,
        test_mode=False,
        include_depth_edges=True,
        include_global_nodes=INCLUDE_GLOBAL_NODES
):
    with open(file_names, "r") as file:
        file_names = file.readlines()
    file_names = [line.strip() for line in file_names if line[0] != "#"]

    dataset: Dataset = dataset_class(
        root=save_folder,
        train_names=file_names,
        test_mode=test_mode,
        include_depth_edges=include_depth_edges
    )

    samples = {
        "name": [],
        "data": [],
        "positive_edge_indices_treble": [],
        "negative_edge_indices_treble": [],
        "positive_edge_indices_bass": [],
        "negative_edge_indices_bass": [],
        "all_edges_indices_treble": [],
        "all_edges_indices_bass": [],
        "voice": []
    }
    return data_loader_all_depths(dataset, depth, include_global_nodes, samples)


def _get_all_pairs_edge_index(data, important_indices, include_global_nodes):
    num_nodes = data.num_nodes - 4
    from_nodes = []
    to_nodes = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            from_nodes.append(i)
            to_nodes.append(j)

    all_edges = torch.stack([torch.tensor(from_nodes), torch.tensor(to_nodes)])

    if include_global_nodes:
        for i in range(data.num_nodes - 4):
            all_edges = torch.cat((all_edges, torch.tensor([[i], [important_indices["grand_start"]]])), dim=1)
            all_edges = torch.cat((all_edges, torch.tensor([[i], [important_indices["grand_end"]]])), dim=1)

    return all_edges


def _get_negative_edge_index(data, training_voice, depth, important_indices, include_global_nodes, pos_edge_index,
                             method):
    if method == 'all':
        positive_edges = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
        num_nodes = data.num_nodes - 4
        from_nodes = []
        to_nodes = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) not in positive_edges:
                    from_nodes.append(i)
                    to_nodes.append(j)

        neg_edge_index = torch.stack([torch.tensor(from_nodes), torch.tensor(to_nodes)])
    else:
        neg_edge_index = random_path_sampling(
            edge_index=data[('note', f'{training_voice}_depth{depth}', 'note')].edge_index,
            num_nodes=data.num_nodes - 4
        )
    if include_global_nodes:
        for i in range(data.num_nodes - 4):
            if i != important_indices["node_start"]:
                neg_edge_index = torch.cat((neg_edge_index, torch.tensor([[i], [important_indices["grand_start"]]])),
                                           dim=1)
            if i != important_indices["node_end"]:
                neg_edge_index = torch.cat((neg_edge_index, torch.tensor([[i], [important_indices["grand_end"]]])),
                                           dim=1)
    return neg_edge_index


def _get_all_positive_edge_indices(data, training_voice, include_global_nodes):
    pos_edge_index_list = []
    important_indices = None
    for d in range(NUM_DEPTHS):
        try:
            pos_edge_index = copy.deepcopy(data[('note', f'{training_voice}_depth{d}', 'note')].edge_index)
            if include_global_nodes:
                pos_edge_index, important_indices = _get_positive_global_edges(
                    data, training_voice, pos_edge_index
                )
            pos_edge_index_list.append(pos_edge_index)
        except AttributeError:
            break
    pos_edge_index = torch.cat(pos_edge_index_list, dim=1)
    # pos_edge_index = torch.unique(pos_edge_index, dim=1)
    return pos_edge_index, important_indices


def _get_positive_global_edges(data, training_voice, pos_edge_index):
    important_indices = {
        "grand_start": data.num_nodes - int(training_voice == 'treble')*2 - 2,
        "grand_end": data.num_nodes - int(training_voice == 'treble')*2 - 1,
        "node_start": pos_edge_index[0][0].item(),
        "node_end": pos_edge_index[1][-1].item()
    }

    start_edge = torch.tensor([important_indices["node_start"], important_indices["grand_start"]])
    end_edge = torch.tensor([important_indices["node_end"], important_indices["grand_end"]])
    pos_edge_index = torch.cat((pos_edge_index, torch.stack((start_edge, end_edge), dim=1)), dim=1)
    return pos_edge_index, important_indices


def _get_positive_edge_index(data, training_voice, depth, include_global_nodes):
    pos_edge_index = copy.deepcopy(data[('note', f'{training_voice}_depth{depth}', 'note')].edge_index)

    important_indices = None
    if include_global_nodes:
        pos_edge_index, important_indices = _get_positive_global_edges(data, training_voice, pos_edge_index)
    return pos_edge_index, important_indices


def random_path_sampling(
        edge_index,
        num_nodes,
        num_paths=10,
        max_attempts=50,
        max_path_attempts=10,
        max_unique_attemps=100
):
    positive_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    def sample_path():
        path_attempts = 0
        path = [0]
        current_node = 0
        while current_node != num_nodes - 1:
            next_node = current_node
            attempts = 0
            while next_node == current_node or (current_node, next_node) in positive_edges and attempts < max_attempts:
                next_node = torch.randint(current_node + 1, num_nodes, (1,)).item()
                attempts += 1
            if attempts >= max_attempts:
                path = [1]
                current_node = 1
                path_attempts += 1
                if path_attempts >= max_path_attempts:
                    return [float('nan')]
                continue
            path.append(next_node)
            current_node = next_node
        return path

    unique_paths = set()
    unique_edges = set()

    unique_path_attempts = 0
    while len(unique_paths) < num_paths and unique_path_attempts < max_unique_attemps:
        path = tuple(sample_path())
        if float('nan') in path:
            break
        unique_paths.add(path)
        unique_path_attempts += 1

    for path in unique_paths:
        for i in range(len(path) - 1):
            unique_edges.add((path[i], path[i + 1]))

    edges_tensor = torch.tensor(list(unique_edges), dtype=torch.long).t()

    return edges_tensor


def prepare_model(
        model_class,
        *args,
        learning_rate=LEARNING_RATE,
        **kwargs,

):
    model = model_class(
        *args,
        **kwargs
        # num_features,
        # embedding_dim,
        # hidden_dim,
        # output_dim,
        # num_layers,
        # dropout,
        # diff_dim,
        # diff_node_num,
        # device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # adamW
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
    return model, optimizer, scheduler


def save_pretrained_model(model, link_predictor_treble, link_predictor_bass, voice_predictor, epoch, valid_loss):
    # gnn_save_path = f"./saved_models/GNN_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    # lp_save_path = f"./saved_models/LP_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    gnn_save_path = f"saved_models/GNN_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    lp_treble_save_path = f"saved_models/LP_treble_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    lp_bass_save_path = f"saved_models/LP_bass_input{GNN_OUTPUT_DIM}_hidden{LINK_PRED_HIDDEN_DIM}_layers{LINK_PRED_NUM_LAYERS}"
    vp_save_path = f"saved_models/VP_input{GNN_OUTPUT_DIM}_hidden{VOICE_PRED_HIDDEN_DIM}_layers{VOICE_PRED_NUM_LAYERS}"
    os.makedirs(gnn_save_path, exist_ok=True)
    os.makedirs(lp_treble_save_path, exist_ok=True)
    os.makedirs(lp_bass_save_path, exist_ok=True)
    os.makedirs(vp_save_path, exist_ok=True)

    gnn_save_number = len(os.listdir(gnn_save_path))
    lp_save_number = len(os.listdir(lp_treble_save_path))
    vp_save_number = len(os.listdir(vp_save_path))

    torch.save(model.state_dict(),
               f"{gnn_save_path}/gnn{gnn_save_number}_epoch{epoch}_loss{np.round(valid_loss, decimals=2)}")
    torch.save(link_predictor_treble.state_dict(),
               f"{lp_treble_save_path}/lp{lp_save_number}_epoch{epoch}_loss{np.round(valid_loss, decimals=2)}")
    torch.save(link_predictor_bass.state_dict(),
               f"{lp_bass_save_path}/lp{lp_save_number}_epoch{epoch}_loss{np.round(valid_loss, decimals=2)}")
    torch.save(voice_predictor.state_dict(),
               f"{vp_save_path}/vp{vp_save_number}_epoch{epoch}_loss{np.round(valid_loss, decimals=2)}")


def save_reward_model(model, epoch, train_loss):
    gnn_save_path = f"saved_models/reward_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
    os.makedirs(gnn_save_path, exist_ok=True)
    gnn_save_number = len(os.listdir(gnn_save_path))

    torch.save(model.state_dict(),
               f"{gnn_save_path}/reward{gnn_save_number}_epoch{epoch}_loss{np.round(train_loss, decimals=2)}")


# def save_reward_model(model, epoch, valid_loss):
#     gnn_save_path = f"saved_models/reward_emb{GNN_EMB_DIM}_hidden{GNN_HIDDEN_DIM}_out{GNN_OUTPUT_DIM}_layers{GNN_NUM_LAYERS}"
#     os.makedirs(gnn_save_path, exist_ok=True)
#     gnn_save_number = len(os.listdir(gnn_save_path))

#     torch.save(model.state_dict(),
#                f"{gnn_save_path}/reward{gnn_save_number}_epoch{epoch}_loss{np.round(valid_loss, decimals=2)}")


def predict_layer(
        final_embedding,
        link_predictor_treble, link_predictor_bass,
        pos_edge_index_treble, neg_edge_index_treble,
        pos_edge_index_bass, neg_edge_index_bass
):
    pos_pred_treble = link_predictor_treble(final_embedding[pos_edge_index_treble[0]],
                                            final_embedding[pos_edge_index_treble[1]])
    neg_pred_treble = link_predictor_treble(final_embedding[neg_edge_index_treble[0]],
                                            final_embedding[neg_edge_index_treble[1]])
    pos_pred_bass = link_predictor_bass(final_embedding[pos_edge_index_bass[0]],
                                        final_embedding[pos_edge_index_bass[1]])
    neg_pred_bass = link_predictor_bass(final_embedding[neg_edge_index_bass[0]],
                                        final_embedding[neg_edge_index_bass[1]])
    return pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass


def calculate_loss_lp(pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass):
    loss_treble = -torch.log(pos_pred_treble + 1e-15).mean() \
                  - torch.log(1 - neg_pred_treble + 1e-15).mean()
    loss_bass = -torch.log(pos_pred_bass + 1e-15).mean() \
                - torch.log(1 - neg_pred_bass + 1e-15).mean()
    pred_sum = torch.sum(pos_pred_treble > 0.5) + torch.sum(pos_pred_bass > 0.5) \
               + torch.sum(neg_pred_treble < 0.5) + torch.sum(neg_pred_bass < 0.5)
    pred_num = pos_pred_treble.size(0) + pos_pred_bass.size(0) + neg_pred_treble.size(0) + neg_pred_bass.size(0)
    accuracy = pred_sum.item() / pred_num
    return loss_bass + loss_treble, accuracy


def plot_metrics(
        train_loss_curve,
        valid_loss_curve,
        train_acc_curve,
        valid_acc_curve,
        save_name=None
):
    epochs = np.arange(NUM_EPOCHS)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Loss Curve and Acc Curve for batch size = 1 with SAGE', fontsize=16, y=1.02)

    # Loss curves
    axs[0].plot(epochs, train_loss_curve, label='Training Loss')
    axs[0].plot(epochs, valid_loss_curve, label='Validation Loss')
    axs[0].set_xlabel('Epoch Number')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    # Accuracy curves
    axs[1].plot(epochs, train_acc_curve, label='Training Accuracy')
    axs[1].plot(epochs, valid_acc_curve, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch Number')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy Curves')
    axs[1].legend()

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)


def get_scale_degree_with_chromatic(key, pitch):
    diatonic_degree = key.getScaleDegreeFromPitch(pitch)


def transpose_to_C(key_sig: music21.key.Key):
    c = music21.pitch.Pitch('C4')
    tonic = key_sig.tonic
    interval = music21.interval.Interval(tonic, c)
    print(interval)
    key_transposed = key_sig.transpose(interval)
    print(key_transposed)


def calculate_loss_reward(sorted_rewards):
    # return -(1 / (1 + torch.exp(sorted_rewards[1] - sorted_rewards[0])))
    return -torch.log(torch.sigmoid(sorted_rewards[0] - sorted_rewards[1]))


def ppo_loss_fn(new_probs, initial_probs, kl_divergence, rewards, clip_param=0.2, kl_weight = 0.2, reward_weight=0.8):
    # Compute ratio between new and initial text probabilities
    ratio = new_probs / initial_probs

    # Compute R
    R = reward_weight * rewards + kl_divergence * kl_weight

    # Clip the ratio
    clipped_ratio = torch.clip(ratio, 1 - clip_param, 1 + clip_param)

    loss = -torch.min(ratio * R, clipped_ratio * R)

    return loss.mean()

# PPO training loop
def ppo_training_loop(model, input, optimizer, kl_divergence, rewards, dataloader, N = 4, clip_param=0.2, kl_weight = 0.2, reward_weight=0.8):
    model.train()

    initial_probs = model(input)  # Given for PPO

    # We have initial probs, and final reward already
    input, initial_probs, rewards = batch

    for i in range(N):
        new_probs = model(input)

        # PPO Loss
        loss = ppo_loss_fn(new_probs, initial_probs,kl_divergence,  rewards, clip_param, reward_weight, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'PPO Loss: {loss.item()}')

    print("PPO Training completed!")

if __name__ == "__main__":
    # from data_processing import HeteroGraphData
    # prepare_train_valid_data_loaders(
    #     TRAIN_NAMES, SAVE_FOLDER, 1, HeteroGraphData
    # )
    from music21.key import Key

    key_sig = Key("E")
    transpose_to_C(key_sig)
    # sd = get_scale_degree_with_chromatic(key_sig, music21.pitch.Pitch("D#"))
    # print(sd)
