import torch
import torch.nn as nn
import torch.nn.functional as F


class VoicePredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout
                 ):
        super(VoicePredictor, self).__init__()

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

    def forward(self, x):
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_layers[-1](x)
        return torch.sigmoid(x)
