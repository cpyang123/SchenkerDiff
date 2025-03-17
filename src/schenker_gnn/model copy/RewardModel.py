import torch
import torch.nn as nn

from config import DEVICE, EMBEDDING_METHOD, INCLUDE_GLOBAL_NODES, DIFF_POOL_EMB_DIM, DIFF_NODE_NUM

from model.schenker_GNN_model import SchenkerGNN


class RewardGNN(torch.nn.Module):
    def __init__(self,
                 num_feature=111,
                 embedding_dim=64,
                 hidden_dim=64,
                 output_dim=64,
                 num_layers=2,
                 dropout=0.15,
                 diff_dim=DIFF_POOL_EMB_DIM,
                 diff_node_num=DIFF_NODE_NUM,
                 device=DEVICE,
                 include_global_nodes=INCLUDE_GLOBAL_NODES
                 ):
        super(RewardGNN, self).__init__()
        self.device = device

        self.schenkerGNN = SchenkerGNN(
            num_feature,
            embedding_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
            diff_dim,
            diff_node_num,
            device,
            include_global_nodes,
            include_schenkerian_edges=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.global_nodes = ['ST', 'SB', 'ET', 'EB']

    def forward(self, data, embedding_method=EMBEDDING_METHOD, mixed=True):
        node_embeddings = self.schenkerGNN(data, embedding_method, mixed)
        graph_embedding = node_embeddings.mean(dim=0)
        reward = self.mlp(graph_embedding)
        return reward