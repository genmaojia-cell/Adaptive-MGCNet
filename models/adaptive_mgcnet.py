import torch
import torch.nn as nn
from .adaptive_graph import AdaptiveGraphLearner
from .gcn_module import SparseGCNModule


class AdaptiveMGCNet(nn.Module):
    """
    Adaptive-MGCNet: Adaptive Multimodal Population Graph Convolutional Network.
    Integrates fMRI pre-projection, dynamic graph learning, sparse GCN convolutions,
    and decoupled late phenotypic fusion for ASD diagnosis.
    """

    def __init__(self, fmri_dim=500, pheno_dim=31, hidden_dim=64, k_neighbors=5):
        super(AdaptiveMGCNet, self).__init__()
        self.k_neighbors = k_neighbors

        # 1. fMRI pre-projection and LayerNorm denoising (Eq. 2)
        self.fmri_projection = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 2. Dynamic graph learner (Eq. 3, 4, 5)
        self.graph_learner = AdaptiveGraphLearner(feature_dim=hidden_dim, embedding_dim=hidden_dim)

        # 3. Two-layer GCN feature aggregation (Eq. 6)
        self.gcn_module = SparseGCNModule(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            dropout_rate=0.2
        )

        # 4. Clinical phenotypic projection layer (Eq. 7)
        self.pheno_layer = nn.Sequential(
            nn.Linear(pheno_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 5. Multimodal late fusion classifier (Eq. 8, 9)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_f, x_p):
        """
        Args:
            x_f: Fold-specific standardized fMRI FC features [N, 500]
            x_p: Standardized clinical phenotypic covariates [N, 31]
        Returns:
            out: Predicted posterior probabilities for ASD, clamped to [1e-6, 1 - 1e-6]
        """
        x_f_clean = self.fmri_projection(x_f)
        edge_index, edge_weight = self.graph_learner(x_f_clean, k_neighbors=self.k_neighbors)

        h_f = self.gcn_module(x_f_clean, edge_index, edge_weight)
        h_p = self.pheno_layer(x_p)

        out = self.classifier(torch.cat([h_f, h_p], dim=1))
        return torch.clamp(out, 1e-6, 1.0 - 1e-6)