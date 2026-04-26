import torch
import torch.nn as nn
import torch.nn.functional as F
 

# ─────────────────────────────────────────────
# 1. GRAPH-CONDITIONED NORMALIZATION
# ─────────────────────────────────────────────
 
class GraphConditionedNorm(nn.Module):
    """
    Learned, metadata-free normalization.
 
    For each sample/graph:
      1. Compute summary statistics over all node features
         (mean, std, per-channel quantile sketch).
      2. Pass through a small MLP → per-node scale + shift parameters.
      3. Apply affine transform to raw features.
 
    Why this works on log2(TPM+1):
      The grossest multiplicative artifacts are already gone.
      What remains are structured additive offsets in log-space
      (platform/protocol effects on gene subsets).
      This layer learns to map those offsets toward a representation
      that produces stable GAT attention — driven purely by task loss,
      not by external batch labels.
 
    Args:
        in_channels:  number of node features (genes)
        hidden_dim:   MLP hidden size for the conditioning network
    """
 
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
 
        # Summary stat dimensionality:
        #   mean (C) + std (C) + [q25, q50, q75] (3*C) = 5*C
        stat_dim = 5 * in_channels
 
        self.conditioner = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * in_channels),  # → scale, shift
        )
 
        # Learnable fallback for stability
        self.default_scale = nn.Parameter(torch.ones(in_channels))
        self.default_shift = nn.Parameter(torch.zeros(in_channels))
 
    def _graph_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N_nodes, C]  — all nodes in ONE graph
        returns: [5*C]
        """
        mean = x.mean(dim=0)                           # [C]
        std  = x.std(dim=0).clamp(min=1e-6)            # [C]
        q25, q50, q75 = x.quantile(
            torch.tensor([0.25, 0.50, 0.75], device=x.device), dim=0
        )                                               # each [C]
        return torch.cat([mean, std, q25, q50, q75])   # [5*C]
 
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x:     [total_nodes, C]
        batch: [total_nodes]  — PyG batch index
        """
        out = torch.zeros_like(x)
 
        for graph_idx in batch.unique():
            mask = (batch == graph_idx)
            x_g  = x[mask]                              # [N_g, C]
 
            stats  = self._graph_stats(x_g)             # [5*C]
            params = self.conditioner(stats.unsqueeze(0)).squeeze(0)  # [2*C]
            scale, shift = params.chunk(2, dim=-1)      # each [C]
 
            # Residual: start from identity, learn corrections
            scale = self.default_scale + scale
            shift = self.default_shift + shift
 
            # Instance-normalize first, then apply learned affine
            x_norm = (x_g - x_g.mean(0)) / (x_g.std(0).clamp(min=1e-6))
            out[mask] = x_norm * scale + shift
 
        return out