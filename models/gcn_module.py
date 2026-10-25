import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SparseGCNModule(nn.Module):
    """
    Two-layer Graph Convolutional Module with edge weights and dropout.
    Corresponds to Equation (6) in the JBHI paper.
    """
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=64, dropout_rate=0.2):
        super(SparseGCNModule, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input latent features, shape [N, in_channels]
            edge_index: Graph edge indices, shape [2, E]
            edge_weight: Graph edge weights, shape [E]
        Returns:
            High-order topological representation, shape [N, out_channels]
        """
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.gcn2(h, edge_index, edge_weight)
        return h