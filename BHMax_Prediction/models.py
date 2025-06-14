# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


class CustomConv(MessagePassing):
    def __init__(self, node_fea_len, edge_fea_len):
        super(CustomConv, self).__init__(aggr='mean')  # Aggregation method: mean
        self.fc = nn.Linear(2 * node_fea_len + edge_fea_len, node_fea_len)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.relu(self.fc(z))


class GNNModel(nn.Module):
    def __init__(self, orig_node_fea_len, edge_fea_len, node_fea_len=18,
                 n_conv=1, h_fea_len=16, n_h=0):
        super(GNNModel, self).__init__()
        self.embedding = nn.Linear(orig_node_fea_len, node_fea_len)
        self.convs = nn.ModuleList([
            CustomConv(node_fea_len, edge_fea_len) for _ in range(n_conv)
        ])
        self.fc_post_conv = nn.Sequential(
            nn.Linear(node_fea_len, h_fea_len),
            nn.Dropout(p=0.6),
            nn.Softplus()
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_fea_len, h_fea_len),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            ) for _ in range(n_h - 1)
        ])
        self.final_fc = nn.Sequential(
            nn.Linear(h_fea_len, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = self.fc_post_conv(x)
        x = global_mean_pool(x, batch)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.final_fc(x)
