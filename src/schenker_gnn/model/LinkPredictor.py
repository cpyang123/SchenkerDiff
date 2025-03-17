import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on: https://colab.research.google.com/drive/1mzvskulGoM9uXCkc4Cke3_Ch0-lz_HnN#scrollTo=kLDpWpf7TeTc
class LinkPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout
                 ):
        super(LinkPredictor, self).__init__()

        # Create linear layers
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.linear_layers.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.linear_layers:
            layer.reset_parameters()

    def forward(self, x_i, x_j):
        # x_i and x_j are both of shape (E, D)
        x = x_i * x_j
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_layers[-1](x)
        return torch.sigmoid(x)
