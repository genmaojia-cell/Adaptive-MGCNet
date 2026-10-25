import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearner(nn.Module):
    """
    Adaptive Population Graph Learner with Dynamic Top-K Sparsification.
    Corresponds to Equations (3), (4), and (5) in the JBHI paper.
    """

    def __init__(self, feature_dim=64, embedding_dim=64):
        super(AdaptiveGraphLearner, self).__init__()
        self.lin1 = nn.Linear(feature_dim, embedding_dim)
        self.lin2 = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x, k_neighbors=5):
        """
        Args:
            x: Refined latent fMRI feature matrix, shape [N, feature_dim]
            k_neighbors: Sparsity degree constraint K
        Returns:
            edge_index: PyG graph edge index tensor, shape [2, N * K]
            edge_weight: Inferred affinity edge weights, shape [N * K]
        """
        num_nodes = x.size(0)
        x1 = F.relu(self.lin1(x))
        x2 = F.relu(self.lin2(x))

        # Scaled dot-product attention
        scores = torch.matmul(x1, x2.t()) / np.sqrt(x1.size(-1))
        S = F.softmax(scores, dim=-1)

        # Dynamic Top-K sparsification
        topk_values, topk_indices = torch.topk(S, k=k_neighbors, dim=-1)

        row = torch.arange(num_nodes).view(-1, 1).repeat(1, k_neighbors).view(-1).to(x.device)
        col = topk_indices.view(-1).to(x.device)

        edge_index = torch.stack([row, col], dim=0)
        edge_weight = topk_values.view(-1)
        return edge_index, edge_weight