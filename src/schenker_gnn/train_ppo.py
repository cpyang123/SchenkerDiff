import copy

import numpy as np
import torch.nn.functional as F
from torch import logsumexp

from config import *
from inference.inference_utils import get_node_list, sample_analysis
from model.RewardModel import RewardGNN
from ppo_utils import get_all_edge_index
from utils import prepare_train_valid_data_loaders
from visualization.visualization_utils import predict_link_probabilities, load_all_models


def rlhf_train_loop(
        gnn_model,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        optimizers,
        samples,
        reward_model: RewardGNN,
        N=4, clip_param=0.2, kl_weight=0.2, reward_weight=0.8
):
    # Set pretrained model to learn
    gnn_model.train()
    link_predictor_treble.train()
    link_predictor_bass.train()
    voice_predictor.train()
    # Set reward model to be frozen
    reward_model.eval()

    train_losses = []
    for name, data, edge_index_treble, edge_index_bass, voice_gt in zip(
            samples['name'],
            samples['data'],
            samples['all_edges_indices_treble'],
            samples['all_edges_indices_bass'],
            samples['voice']
    ):
        edge_index = get_all_edge_index(voice_gt)

        # Create frozen version for all models
        frozen_model_gnn = copy.deepcopy(gnn_model).eval()
        frozen_model_link_treble = copy.deepcopy(link_predictor_treble).eval()
        frozen_model_link_bass = copy.deepcopy(link_predictor_bass).eval()
        frozen_model_voice = copy.deepcopy(voice_predictor).eval()

        # GENERATE ANALYSIS
        _, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities(
            name,
            data,
            frozen_model_gnn,
            frozen_model_link_treble,
            frozen_model_link_bass,
            frozen_model_voice,
            edge_index
        )

        node_list = get_node_list(human_readable_voice)

        frozen_analysis_treble, frozen_analysis_bass = sample_analysis(
            node_list, human_readable_treble, human_readable_bass, k=3
        )

        frozen_probs = [human_readable_treble[edge] for edge in frozen_analysis_treble] + \
                       [human_readable_bass[edge] for edge in frozen_analysis_bass]
        frozen_probs = torch.tensor(frozen_probs)
        frozen_edge_index_treble = torch.tensor(frozen_analysis_treble).T
        frozen_edge_index_bass = torch.tensor(frozen_analysis_bass).T

        # CALCULATE REWARD
        data_reward = copy.deepcopy(data)
        for edge_type, schenker_edge_index in zip(
                ['schenker_treble', 'schenker_bass'],
                [frozen_edge_index_treble, frozen_edge_index_bass]
        ):
            data_reward[('note', edge_type, 'note')].edge_index = schenker_edge_index

            num_edges = data_reward[('note', edge_type, 'note')].edge_index.shape[1]
            edge_weights = torch.ones(num_edges)
            data_reward[('note', edge_type, 'note')].edge_attr = edge_weights

        reward = reward_model(data_reward)

        initial_probs = logsumexp(frozen_probs, 0)

        # PPO Loop
        for i in range(N):
            try:
                # Create Initial Policy Probabilities
                _, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities(
                    name,
                    data,
                    gnn_model,
                    link_predictor_treble,
                    link_predictor_bass,
                    voice_predictor,
                    edge_index
                )

                # add in the post processing
                # human_readable stuff is a dict of edge probabiflities, we need to process them into a list of probabilities
                training_probs = [human_readable_treble[edge] for edge in frozen_analysis_treble] + \
                                 [human_readable_bass[edge] for edge in frozen_analysis_bass]
                training_probs = torch.tensor(training_probs)
            except IndexError as e:
                print(name)
                print(e)
                continue

            # Calculate KL divergance
            training_dist = torch.log(torch.softmax(training_probs, 0))
            frozen_dist = torch.softmax(frozen_probs, 0)
            kl_divergence = F.kl_div(training_dist, frozen_dist, reduction="batchmean")

            # Calculate log sum of probs
            new_probs = logsumexp(training_probs, 0)

            # Compute ratio between new and initial text probabilities
            ratio = new_probs / initial_probs

            # Compute R
            R = reward_weight * reward + kl_divergence * kl_weight

            # Clip the ratio
            clipped_ratio = torch.clip(ratio, 1 - clip_param, 1 + clip_param)

            loss = -torch.min(ratio * R, clipped_ratio * R)

            loss.backward(retain_graph=True)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            print(f'PPO Loss: {loss.item()}')
        train_losses.append(loss.item())

    return np.mean(train_losses)


if __name__ == "__main__":
    from data_processing import HeteroGraphData

    training_samples, validation_samples = prepare_train_valid_data_loaders(
        TRAIN_NAMES, SAVE_FOLDER, NUM_DEPTHS, HeteroGraphData
    )

    # Load all pretrained models
    gnn, lp_treble, lp_bass, voice_model, reward_model, \
        optimizer_gnn, optimizer_lp_treble, optimizer_lp_bass, optimizer_voice, _, \
        scheduler_gnn, scheduler_lp_treble, scheduler_lp_bass, scheduler_voice, _ \
        = load_all_models(include_reward=True)

    optimizers = [optimizer_gnn, optimizer_lp_treble, optimizer_lp_bass, optimizer_voice]
    schedulers = [scheduler_gnn, scheduler_lp_treble, scheduler_lp_bass, scheduler_voice]

    for epoch in range(RLHF_EPOCHS):
        train_loss = rlhf_train_loop(
            gnn,
            lp_treble,
            lp_bass,
            voice_model,
            optimizers,
            training_samples,
            reward_model
        )
        print(
            f'Epoch: {epoch + 1}, '
            f'RLHF Training Loss: {train_loss:.4f}, '
        )
        for scheduler in schedulers:
            scheduler.step()
    #     if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 and SAVE_MODEL:
    #         save_model(model, link_predictor_treble, link_predictor_bass, voice_predictor, epoch, valid_loss)
    #
    # if SAVE_MODEL and (epoch + 1) % 5 != 0:
    #     save_model(model, link_predictor_treble, link_predictor_bass, voice_predictor, NUM_EPOCHS, valid_loss)

    # PPO Process Outline
    # Freeze base model
    # Calculate KL Loss
    # Find reward model somwhere
    # Need clarification on:
    """
    - Whether or not to recalculate KL divergance during PPO
    - How the reward is going to feed back into the model
    - How the dataflow is going to look like:
    - - Where's the feedback data?
        Use training data
    - - What format will it be, what kind of processing is needed
    """
    # Input into reward model to generate final reward
    # ppo_training_loop
