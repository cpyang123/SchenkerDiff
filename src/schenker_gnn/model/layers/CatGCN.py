import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class BatchAcc(nn.Module):

    '''
        Batched global accumulation layer
        Code by: https://github.com/TachiChan/CatGCN/blob/master/gnn_layers.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(BatchAcc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class CatEmbedder(nn.Module):

    def __init__(self, n_cats, n_nums, d_embed, n_global, n_local, probe, alpha):

        super(CatEmbedder, self).__init__()

        self.n_cats = n_cats
        self.n_nums = n_nums
        self.d_embed = d_embed
        self.n_global = n_global
        self.n_local = n_local

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedder = nn.Embedding(n_cats, d_embed)
        self.embedder.weight.requires_grad = True

        # For mixed numerical-categorical feature embedding
        self.num_embedder = [nn.Linear(1, d_embed).to(self.device) for _ in range(n_nums)]

        self.global_acc = BatchAcc(d_embed, d_embed)
        self.global_stack = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=True) for _ in range(n_global)])
        self.local_stack = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=True) for _ in range(n_local)])

        # Diagonal probing hyperparameter rho
        self.probe = probe
        # Hyperparameter for convex combination of global and local embedding
        self.alpha = alpha

    def forward(self, cat_indices, num_features=None):

        n_num = self.n_nums if num_features is not None else 0

        if num_features is not None:
            num_embedding_list = []
            for i in range(self.n_nums):
                curr_num_embedding = self.num_embedder[i](num_features[:, i].unsqueeze(1))
                num_embedding_list.append(curr_num_embedding.unsqueeze(1))
            num_embedding = torch.cat(num_embedding_list, 1).squeeze(2).to(self.device)

        cat_adjs = self.generate_cat_adjs(cat_indices.shape[0], cat_indices.shape[1] + n_num)
        shallow_embedding = self.embedder(cat_indices)
        embedding = torch.cat((shallow_embedding, num_embedding), 1)

        # Global feature pooling
        global_features = self.global_acc(embedding, cat_adjs.float())
        global_features = F.relu(global_features)
        # Pool global features together
        global_features = torch.mean(global_features, dim=-2)

        for i, layer in enumerate(self.global_stack):
            global_features = layer(global_features)
            if i+1 < self.n_global:
                global_features = F.relu(global_features)

        # Local feature pooling
        summed_local_features = torch.sum(embedding, 1) 
        square_summed_field_features = summed_local_features ** 2 
        # squre-sum-part
        squared_local_features = shallow_embedding ** 2 
        sum_squared_local_features = torch.sum(squared_local_features, 1)
        # second order
        local_features = 0.5 * (square_summed_field_features - sum_squared_local_features)

        for i, layer in enumerate(self.local_stack):
            local_features = layer(local_features)
            if i + 1 < self.n_local:
                local_features = F.relu(local_features)

        return self.alpha * global_features + (1 - self.alpha) * local_features

    def generate_cat_adjs(self, node_count, num_cats):
        # Normalization by  P'' = Q^{-1/2}*P'*Q^{-1/2}, P' = P+probe*O
        # Note that num_cats is the number of "active" or non-zero categories
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cat_adjs = torch.ones((node_count, num_cats, num_cats))
        cat_adjs += self.probe * torch.eye(num_cats)
        row_sum = num_cats + self.probe
        cat_adjs = (1. / row_sum) * cat_adjs
        return cat_adjs.to(device)

def one_hot_to_indices(one_hot):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.nonzero(one_hot == 1).squeeze().to(device)
