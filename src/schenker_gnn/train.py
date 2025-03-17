import numpy as np

from config import *
from model.LinkPredictor import LinkPredictor
from model.VoicePredictor import VoicePredictor
from utils import prepare_train_valid_data_loaders, prepare_test_data_loaders, prepare_model, save_pretrained_model, \
    calculate_loss_lp, predict_layer

import wandb


def train_loop(
        gnn_model,
        link_predictor_treble,
        link_predictor_bass,
        voice_predictor,
        optimizers,
        samples,
        ablations=None
):
    if ablations is None:
        ablations = {
            'voice_given': False,
            'loss_weight': LOSS_WEIGHT,
            'diffpool': True,
            'voice_concat': True
        }
    gnn_model.train()
    link_predictor_treble.train()
    link_predictor_bass.train()
    voice_predictor.train()

    train_losses = []
    train_voice_loss = []
    train_lp_accs = []
    train_vp_accs = []
    for name, data, pos_edge_index_treble, neg_edge_index_treble, pos_edge_index_bass, neg_edge_index_bass, voice_gt in zip(
            samples['name'],
            samples['data'],
            samples['positive_edge_indices_treble'],
            samples['negative_edge_indices_treble'],
            samples['positive_edge_indices_bass'],
            samples['negative_edge_indices_bass'],
            samples['voice']
    ):
        for optimizer in optimizers:
            optimizer.zero_grad()

        try:
            voice_pred, pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass = all_models_forward(
                data, gnn_model, voice_predictor, link_predictor_treble, link_predictor_bass,
                pos_edge_index_treble, neg_edge_index_treble,
                pos_edge_index_bass, neg_edge_index_bass,
                ablations, voice_gt
            )
        except IndexError as e:
            continue
        lp_loss, lp_accuracy = calculate_loss_lp(pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass)

        voice_loss = VOICE_LOSS(voice_pred, voice_gt)
        voice_accuracy = torch.sum(torch.round(voice_pred) == voice_gt).item() / voice_gt.flatten().size(0)

        if ablations['voice_concat']:
            loss = (1 - LOSS_WEIGHT) * lp_loss + LOSS_WEIGHT * voice_loss
        else:
            loss = lp_loss + voice_loss

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        train_losses.append(lp_loss.item())
        train_voice_loss.append(voice_loss.item())
        train_lp_accs.append(lp_accuracy)
        train_vp_accs.append(voice_accuracy)

    return np.mean(train_losses), np.mean(train_voice_loss), np.mean(train_lp_accs), np.mean(train_vp_accs)


def validation_loop(model, link_predictor_treble, link_predictor_bass, voice_predictor, samples, ablations=None):
    model.eval()
    link_predictor_bass.eval()
    link_predictor_treble.eval()
    voice_predictor.eval()

    if ablations is None:
        ablations = {
            'voice_given': False,
            'loss_weight': LOSS_WEIGHT,
            'diffpool': True,
            'voice_concat': True
        }

    with torch.no_grad():
        val_loss_link = []
        val_loss_voice = []
        val_lp_accs = []
        val_vp_accs = []
        for name, data, pos_edge_index_treble, neg_edge_index_treble, pos_edge_index_bass, neg_edge_index_bass, voice_gt in zip(
                samples['name'],
                samples['data'],
                samples['positive_edge_indices_treble'],
                samples['negative_edge_indices_treble'],
                samples['positive_edge_indices_bass'],
                samples['negative_edge_indices_bass'],
                samples['voice']
        ):
            try:
                voice_pred, pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass = all_models_forward(
                    data, model, voice_predictor, link_predictor_treble, link_predictor_bass,
                    pos_edge_index_treble, neg_edge_index_treble,
                    pos_edge_index_bass, neg_edge_index_bass,
                    ablations, voice_gt
                )
            except IndexError as e:
                continue

            lp_loss, lp_accuracy = calculate_loss_lp(pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass)
            voice_loss = VOICE_LOSS(voice_pred, voice_gt)
            voice_accuracy = torch.sum(torch.round(voice_pred) == voice_gt).item() / voice_gt.flatten().size(0)

            loss = (1 - LOSS_WEIGHT) * lp_loss + LOSS_WEIGHT * voice_loss

            val_loss_link.append(lp_loss.item())
            val_loss_voice.append(voice_loss.item())
            val_lp_accs.append(lp_accuracy)
            val_vp_accs.append(voice_accuracy)

    return np.mean(val_loss_link), np.mean(val_loss_voice), np.mean(val_lp_accs), np.mean(val_vp_accs)


def all_models_forward(
        data,
        model, voice_predictor, link_predictor_treble, link_predictor_bass,
        pos_edge_index_treble, neg_edge_index_treble,
        pos_edge_index_bass, neg_edge_index_bass,
        ablations, voice_gt
):
    final_embedding = model(data)
    if ablations['voice_given']:
        voice_pred = voice_gt
    else:
        voice_pred = voice_predictor(final_embedding)

    if ablations['voice_concat']:
        final_embedding = torch.cat((final_embedding, voice_pred), dim=1)

    pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass = predict_layer(
        final_embedding,
        link_predictor_treble, link_predictor_bass,
        pos_edge_index_treble, neg_edge_index_treble,
        pos_edge_index_bass, neg_edge_index_bass
    )
    return voice_pred, pos_pred_treble, neg_pred_treble, pos_pred_bass, neg_pred_bass


def run_main():
    training_samples, validation_samples = prepare_train_valid_data_loaders(
        TRAIN_NAMES, SAVE_FOLDER, NUM_DEPTHS, HeteroGraphData, include_depth_edges=True
    )

    test_samples = prepare_test_data_loaders(
        TEST_NAMES, TEST_SAVE_FOLDER, NUM_DEPTHS, HeteroGraphData, include_depth_edges=True
    )

    model, optimizer_gnn, scheduler_gnn = prepare_model(
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
    model.to(DEVICE)

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

    optimizers = [optimizer_gnn, optimizer_lp_treble, optimizer_lp_bass, optimizer_voice]
    schedulers = [scheduler_gnn, scheduler_lp_treble, scheduler_lp_bass, scheduler_voice]

    for epoch in range(NUM_EPOCHS):
        train_loss, train_voice_loss, train_lp_acc, train_vp_acc = train_loop(
            model, link_predictor_treble, link_predictor_bass, voice_predictor, optimizers, training_samples,
            ablations=ABLATIONS
        )
        valid_loss, valid_voice_loss, valid_lp_acc, valid_vp_acc = validation_loop(
            model, link_predictor_treble, link_predictor_bass, voice_predictor, validation_samples, ablations=ABLATIONS
        )
        test_loss, test_voice_loss, test_lp_acc, test_vp_acc = validation_loop(
            model, link_predictor_treble, link_predictor_bass, voice_predictor, test_samples, ablations=ABLATIONS
        )
        print(
            f'Epoch: {epoch + 1}, '
            f'Training Loss: {train_loss:.4f}, '
            f'Training Link Acc: {train_lp_acc:.4f}, '
            f'Training Voice Acc: {train_vp_acc:.4f}, '
            # f'Validation Loss: {valid_loss:.4f}, '
            f'Test Loss comb: {test_loss + test_voice_loss:.4f}, '
            f'Test Link Loss: {test_loss:.4f}, '
            f'Test Voice Loss: {test_voice_loss:.4f}, '
            f'Test Link Acc: {test_lp_acc:.4f}, '
            f'Test Voice Acc: {test_vp_acc:4f}'
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_voice_loss": train_voice_loss,
                "train_link_acc": train_lp_acc,
                "train_voice_acc": train_vp_acc,
                # "val_loss": valid_loss,
                "test_loss": test_loss,
                "test_voice_loss": test_voice_loss,
                "test_link_acc": test_lp_acc,
                "test_voice_acc": test_vp_acc
            })
        for scheduler in schedulers:
            scheduler.step()
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 and SAVE_MODEL:
            save_pretrained_model(model, link_predictor_treble, link_predictor_bass, voice_predictor, epoch, test_loss)

    if SAVE_MODEL and (epoch + 1) % 5 != 0:
        save_pretrained_model(
            model, link_predictor_treble, link_predictor_bass, voice_predictor, NUM_EPOCHS, test_loss
        )


if __name__ == "__main__":
    from model.schenker_GNN_model import SchenkerGNN
    from data_processing import HeteroGraphData

    use_wandb = False

    if use_wandb:
        wandb.init(
            project="schenkGNN-LP-experiments",
            name=f"LP-model | {list(ABLATIONS.items())} | {GNN_HIDDEN_DIM}",
        )

    for _ in range(10):
        run_main()


