import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NOTE: Musescore 4 is not supported.
MUSESCORE_PATH = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"

using_diff_pool = True
using_voice_concat = True
LOSS_WEIGHT = 0.1
ABLATIONS = {
    'voice_given': False,
    'loss_weight': LOSS_WEIGHT,
    'diffpool': using_diff_pool,
    'voice_concat': using_voice_concat
}

SAVE_MODEL = True
SAVE_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS_REWARD = 1

TRAIN_PERCENT = 1.0
NUM_DEPTHS = 7
ALL_DEPTHS_AT_ONCE = True

BATCH_SIZE = 1
LEARNING_RATE = 0.001
SCHEDULER_GAMMA = 0.5
MAX_GRAD_NORM = 0.3

TRAIN_NAMES = "train-names.txt"
SAVE_FOLDER = "processed_data"

TEST_NAMES = "test-names.txt"
TEST_SAVE_FOLDER = "processed_data_test"

INTERVAL_EDGES = [1, 2, 3, 4, 5, 8]
LAMBDA_CLAMP_MIN = 0

# -4=Treble start, -3=Treble end, -2=Bass start, -1=Bass end
INCLUDE_GLOBAL_NODES = True

SHARE_BACKBONE_WEIGHTS = False
EMBEDDING_METHOD = 'linear'
NUM_FEATURES = 6 + 2#44 + 42 - 1 - 35 - 26 - 6 - 2
# NUM_FEATURES = 50

GNN_EMB_DIM = 8
GNN_HIDDEN_DIM = 32
GNN_OUTPUT_DIM = 8
GNN_NUM_LAYERS = 1

DIFF_POOL_EMB_DIM = 8
DIFF_NODE_NUM = 2

LINK_PRED_HIDDEN_DIM = 16
LINK_PRED_NUM_LAYERS = 1

VOICE_PRED_HIDDEN_DIM = 16
VOICE_PRED_NUM_LAYERS = 1
VOICE_PRED_NUM_CLASSES = 3
VOICE_LOSS = nn.BCELoss()

REWARD_EMB_DIM = 32
REWARD_HIDDEN_DIM = 256
REWARD_OUT_DIM = 128
REWARD_NUM_LAYERS = 2

DROPOUT_PERCENT = 0.15

NUM_EPOCHS = 10
RLHF_EPOCHS = 10

CLASSIFICATION_CRITERION = torch.nn.NLLLoss()
SIM_CRITERION = nn.CosineEmbeddingLoss()
GROUPING_CRITERION = nn.BCELoss()

#Voice classification
# VOICE_SAVE_FOLDER = "voice_processed_data"
# VOICE_TEST_FOLDER = "voice_processed_data_test"
# VOICE_HIDDEN_DIM = 128

COMPARISON_DIRECTORY = "visualization/reward_model_preference_data"
NUM_COMPARED = 2
TRAIN_NAMES_REWARD = "reward-train-names.txt"