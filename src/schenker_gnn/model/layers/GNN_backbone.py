import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import HeteroConv, GCNConv, LayerNorm

from src.schenker_gnn.config import INTERVAL_EDGES, NUM_DEPTHS
from src.schenker_gnn.model.layers.dir_gcn import DirGNNConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, dropout=0.15, include_schenkerian_edges=False):
        super(HeteroGNN, self).__init__()

        self.num_layers = num_layers
        edge_types = [
            ('note', 'forward', 'note'),
            ('note', 'onset', 'note'),
            ('note', 'sustain', 'note'),
            ('note', 'rest', 'note')
        ] + [
            ('note', f'{contour}{interval}', 'note')
            for contour in ["up", "down"]
            for interval in INTERVAL_EDGES
        # ] + [
            # ('note', f'{voice}_depth{d}', 'note')
            # for voice in ['treble', 'bass']
            # for d in range(NUM_DEPTHS)
        ] + [
            ('note', f"global_{end}_{voice}", 'note')
            for voice in ['treble', 'bass']
            for end in ['start', 'end']
        ]

        if include_schenkerian_edges:
            edge_types += [
                ('note', 'schenker_treble', 'note'),
                ('note', 'schenker_bass', 'note')
            ]

        self.dropout = dropout
        self.directional_alpha = 0.75
        self.conv_layers = ModuleList()
        self.layer_norms = ModuleList()
        mpnn_layer = GCNConv
        self.conv_layers.append(
            HeteroConv({
                edge_type: DirGNNConv(mpnn_layer(input_channels, hidden_channels), self.directional_alpha)
                for edge_type in edge_types
            })
        )
        self.layer_norms.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                HeteroConv({
                    edge_type: DirGNNConv(mpnn_layer(hidden_channels, hidden_channels), self.directional_alpha)
                    for edge_type in edge_types
                })
            )
            self.layer_norms.append(LayerNorm(hidden_channels))

    def forward(self, x, edge_index_dict, attribute_dict):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index_dict, attribute_dict)
            x['note'] = self.layer_norms[i](x['note'])
            x['note'] = F.relu(x['note'])
            x['note'] = F.dropout(x['note'], p=self.dropout, training=self.training)

        return x
        # x = self.conv_1(x, edge_index_dict, attribute_dict)
        # x['note'] = self.norm_1(x['note'])
        # x['note'] = self.dropout(x['note'])
        # x = self.conv_2(x, edge_index_dict, attribute_dict)
        # x['note'] = self.norm_2(x['note'])

        # return x['note']
