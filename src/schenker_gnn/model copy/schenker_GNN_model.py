import torch
import torch.nn as nn

from config import DEVICE, EMBEDDING_METHOD, INCLUDE_GLOBAL_NODES, DIFF_POOL_EMB_DIM, DIFF_NODE_NUM, LOSS_WEIGHT
from model.layers.CatGCN import CatEmbedder, one_hot_to_indices
from model.layers.GNN_backbone import HeteroGNN
from model.layers.diff_pool import DiffPool


class SchenkerGNN(torch.nn.Module):
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
                 include_global_nodes=INCLUDE_GLOBAL_NODES,
                 include_schenkerian_edges=False,
                 ablations=None
                 ):
        super(SchenkerGNN, self).__init__()

        if ablations is None:
            self.ablations = ablations = {
                'unknown': 0,
                'loss_weight': LOSS_WEIGHT,
                'diffpool': True,
                'voice_concat': True
            }
        else:
            self.ablations = ablations

        self.device = device

        self.include_global_nodes = include_global_nodes
        self.global_embedding_table = nn.Embedding(4, embedding_dim)

        self.cat_embed = CatEmbedder(num_feature, 3, embedding_dim, 1, 1, 0.5, 0.5)
        self.linear_embed = nn.Linear(num_feature, embedding_dim)

        if ablations['diffpool']:
            self.pool = DiffPool(device, hidden_dim, diff_dim, diff_node_num, num_layers)
            # self.pool = DiffPool(device, embedding_dim, diff_dim, diff_node_num, num_layers)

        self.message_passing = HeteroGNN(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            include_schenkerian_edges=include_schenkerian_edges
        )

        if ablations['diffpool']:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + diff_node_num + diff_dim, hidden_dim),
                # nn.Linear(embedding_dim + diff_node_num + diff_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )

        self.global_nodes = ['ST', 'SB', 'ET', 'EB']

    def forward_cat_embedder(self, x, mixed):
        x['note'] = x['note'].float()
        # We assume the node embedding is given as (cat, ..., cat, num, ..., num)
        num_numeric_features = 3
        num_features = x['note'][:, -num_numeric_features:]
        cat_features = x['note'][:, :-num_numeric_features]
        cat_indices = [one_hot_to_indices(cat_features[i]) for i in range(cat_features.shape[0])]

        if mixed:
            cat_embedding = self.cat_embed(torch.stack(cat_indices), num_features=num_features)
            x['note'] = cat_embedding
        else:
            cat_embedding = self.cat_embed(torch.stack(cat_indices))
            x['note'] = torch.cat((cat_embedding, torch.unsqueeze(num_features, dim=1)), 1)

        return x

    def forward(self, data, embedding_method=EMBEDDING_METHOD, mixed=True):
        x = data.x_dict
        edge_index_dict = data.edge_index_dict
        attribute_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}

        if self.include_global_nodes:
            x['note'] = x['note'][0:-4, :]

        if embedding_method == "cat":
            x = self.forward_cat_embedder(x, mixed)
        elif embedding_method == "linear":
            x['note'] = self.linear_embed(x['note'])

        if self.include_global_nodes:
            global_embeddings = self.global_embedding_table(torch.arange(len(self.global_nodes), device=self.device))
            x['note'] = torch.cat((x['note'], global_embeddings), 0)

        x = self.message_passing(x, edge_index_dict, attribute_dict)
        
        if self.ablations['diffpool']:
            pooled_notes, s1, x1 = self.pool(x, edge_index_dict, attribute_dict)
            pooled_notes = pooled_notes.expand(x['note'].shape[0], pooled_notes.shape[1])
            x['note'] = torch.cat((x['note'], pooled_notes, torch.matmul(s1, x1)), dim=1)

        x['note'] = self.classifier(x['note'])
        return x['note']
