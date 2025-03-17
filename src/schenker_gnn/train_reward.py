import numpy as np

from config import *
from model.RewardModel import RewardGNN
from utils import prepare_model, save_reward_model, calculate_loss_reward, prepare_data_loaders_reward

def train_loop(
        reward_model,
        optimizer,
        samples
):
    reward_model.train()
    train_losses = []
    rewards_preferred = []
    rewards_not_preffered = []

    for idx, (name, graph_pair) in enumerate(zip(samples['name'], samples['graph_pair'])):
        graph1, graph2 = graph_pair

        graph1 = graph1.to(DEVICE)
        graph2 = graph2.to(DEVICE)

        try:
            reward1 = reward_model(graph1)
            reward2 = reward_model(graph2)
        except RuntimeError:
            # print(f"out of bounds error for {name}")
            continue
        # print(reward1.item(), reward2.item(), reward1.item() - reward2.item())
        
        sorted_rewards = (reward1, reward2)
        loss = calculate_loss_reward(sorted_rewards)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_losses.append(loss.item())
        rewards_preferred.append(reward1.item())
        rewards_not_preffered.append(reward2.item())

    return np.mean(train_losses), np.array(rewards_preferred), np.array(rewards_not_preffered)



def validation_loop(
    reward_model,
    samples
):
    reward_model.eval()
    valid_losses = []
    rewards_preferred = []
    rewards_not_preffered = []

    for idx, (name, graph_pair) in enumerate(zip(samples['name'], samples['graph_pair'])):
        graph1, graph2 = graph_pair

        graph1 = graph1.to(DEVICE)
        graph2 = graph2.to(DEVICE)

        try:
            reward1 = reward_model(graph1)
            reward2 = reward_model(graph2)
        except RuntimeError:
            # print(f"out of bounds error for {name}")
            continue
        # print(reward1.item(), reward2.item(), reward1.item() - reward2.item())

        sorted_rewards = (reward1, reward2)
        loss = calculate_loss_reward(sorted_rewards)

        valid_losses.append(loss.item())
        rewards_preferred.append(reward1.item())
        rewards_not_preffered.append(reward2.item())

    return np.mean(valid_losses), np.array(rewards_preferred), np.array(rewards_not_preffered)

def run_main():
    training_samples, validation_samples = prepare_data_loaders_reward(
        comparison_directory=COMPARISON_DIRECTORY
    )

    model, optimizer, scheduler = prepare_model(
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
    model.to(DEVICE)

    for epoch in range(10):
        train_loss, rewards_preferred, rewards_not_preferred = train_loop(model, optimizer, training_samples)
        valid_loss, valid_rewards_preferred, valid_rewards_not_preferred = validation_loop(model, validation_samples)
        print(
            f'Epoch: {epoch + 1}, '
            f'Training Loss: {train_loss:.4f}, '
            f'Validation Loss: {valid_loss:.4f}, '
            f'Average preferred reward: {np.mean(rewards_preferred)}, '
            f'Average not preferred re: {np.mean(rewards_not_preferred)}, '
            f'Average difference p v n: {np.mean(rewards_preferred - rewards_not_preferred)}'
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": valid_loss,
                "average_diff_train": np.mean(rewards_preferred - rewards_not_preferred),
                "average_diff_valid": np.mean(valid_rewards_preferred - valid_rewards_not_preferred)
            })
        scheduler.step()
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS_REWARD == 0 and SAVE_MODEL:
            save_reward_model(model, epoch, train_loss)

    if SAVE_MODEL and (epoch + 1) % SAVE_EVERY_N_EPOCHS_REWARD != 0:
        save_reward_model(model, NUM_EPOCHS, train_loss)

if __name__ == "__main__":
    import wandb

    use_wandb = True

    if use_wandb:
        wandb.init(
            project="schenkGNN-LP-experiments-reward",
            name=f"Reward | Metric Strength",
        )

    for _ in range(6):
        run_main()
