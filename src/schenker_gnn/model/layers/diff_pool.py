import torch
import torch.nn.functional as F
from torch.nn import LogSoftmax

from src.schenker_gnn.config import DEVICE
from src.schenker_gnn.model.layers.GNN_backbone import HeteroGNN


class DiffPool(torch.nn.Module):
    def __init__(self, device=DEVICE, num_feature=111, output_dim=32, num_nodes=16, num_layers=2):
        super(DiffPool, self).__init__()

        self.device = device

        # Vanila GNN
        # self.gnn1_pool = GNN(num_feature, 32, num_nodes)
        # self.gnn1_embed = GNN(num_feature, 32, 32)

        # Hetero Version
        self.gnn1_pool = HeteroGNN(num_feature, num_nodes, num_layers)
        self.gnn1_embed = HeteroGNN(num_feature, num_nodes, num_layers)

        # num_nodes = ceil(0.25 * num_nodes)
        # self.gnn2_pool = GNN(64, 64, num_nodes)
        # self.gnn2_embed = GNN(64, 64, 64, lin=False)

        # self.gnn3_embed = GNN(32, 32, 32, lin=False)

        # Hetero Version
        self.gnn2_embed = HeteroGNN(num_nodes, num_nodes, num_layers)

        self.softmax = LogSoftmax(dim=1)

        self.lin1 = torch.nn.Linear(num_nodes, 32)
        self.lin2 = torch.nn.Linear(32, output_dim)

    def forward(self, data, edge_index_dict, attribute_dict, mask=None):
        s = torch.squeeze(self.gnn1_pool(data, edge_index_dict, attribute_dict)['note'])
        x = torch.squeeze(self.gnn1_embed(data, edge_index_dict, attribute_dict)['note'])
        s1 = s.clone()  # .detach()

        # print("x1-shape: {}".format(x.shape))
        # print("s1-shape: {}".format(s.shape))

        # Just do softmax of s*x instead of the diffpool - row-wise,
        adj_shape = [x.shape[0], x.shape[0]]

        # Initialize dictionaries for index and edge attributes
        edge_index_dict2 = {}
        attribute_dict2 = {}

        # Check the shapes for the adjacency matriies
        for key, val in edge_index_dict.items():
            sparse_matrix = torch.sparse_coo_tensor(val, attribute_dict[key], adj_shape)
            transformed_matrix = torch.sparse.mm(torch.transpose(self.softmax(s), 0, 1).to_sparse(),
                                                 torch.mm(sparse_matrix, self.softmax(s).to_sparse()))
            edge_index_dict2[key] = transformed_matrix.indices()
            attribute_dict2[key] = transformed_matrix.values()

        x = torch.unsqueeze(torch.matmul(torch.transpose(self.softmax(s), 0, 1), x), 0)
        x1 = x.clone()  # .detach()

        # print("x2-shape: {}".format(x.shape))
        # print("s2-shape: {}".format(s.shape))

        x = self.gnn2_embed({'note': x}, edge_index_dict2, attribute_dict2)['note']

        # print("x3-shape: {}".format(x.shape))

        x = x.mean(dim=1)
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)

        # print("x-shape after activation: {}".format(x.shape))

        return F.log_softmax(x, dim=-1), torch.squeeze(s1), torch.squeeze(x1)