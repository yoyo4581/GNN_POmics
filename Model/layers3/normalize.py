"""GraphConditionedNorm: learned, per-graph feature normalization.

Shape contract:
    input  x:     [N_total, C]  (all graphs in the batch, concatenated)
    input  batch: [N_total]     (PyG batch index, graph id per node)
    output:       [N_total, C]  (same shape as input, values rescaled)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConditionedNorm(nn.Module):
    """Learned, metadata-free normalization conditioned on each graph's own statistics.

    For every graph in the batch:
      1. Compute summary statistics over all of its node features
         (mean, std, and the 25th/50th/75th percentiles, per channel).
      2. Pass those statistics through a small MLP to get per-channel
         scale and shift corrections.
      3. Instance-normalize the graph's features, then apply the learned
         affine transform (scale, shift) as a residual on top of an
         identity fallback.

    Why this works on log2(TPM+1) expression data:
        The grossest multiplicative artifacts are already gone after the
        log transform. What remains are structured additive offsets in
        log-space (platform/protocol effects on gene subsets). This layer
        learns to correct those offsets in a way that stabilizes GAT
        attention downstream — driven purely by task loss, with no
        external batch labels required.

    Shapes:
        x:      [N_total, C]  — node features, all graphs concatenated
        batch:  [N_total]     — PyG batch index (graph id per node)
        output: [N_total, C]  — normalized features, same shape as x

    Args:
        in_channels: Number of node features (genes), C.
        hidden_dim: Hidden size of the MLP that predicts scale/shift.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()

        # Summary stat dimensionality per graph:
        #   mean (C) + std (C) + [q25, q50, q75] (3*C) = 5*C
        stat_dim = 5 * in_channels

        self.conditioner = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * in_channels),  # -> scale, shift
        )

        # Learnable identity fallback: the conditioner learns a correction
        # on top of this rather than the affine params outright.
        self.default_scale = nn.Parameter(torch.ones(in_channels))
        self.default_shift = nn.Parameter(torch.zeros(in_channels))

    def _graph_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-channel summary statistics for one graph's nodes.

        Args:
            x: Tensor[N_g, C] — node features for a single graph (N_g nodes).

        Returns:
            Tensor[5*C]: concatenation of [mean, std, q25, q50, q75], each [C].
        """
        mean = x.mean(dim=0)                            # [C]
        std = x.std(dim=0).clamp(min=1e-6)               # [C]
        q25, q50, q75 = x.quantile(
            torch.tensor([0.25, 0.50, 0.75], device=x.device), dim=0
        )                                                 # each [C]
        return torch.cat([mean, std, q25, q50, q75])     # [5*C]

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Normalize each graph's node features using its own statistics.

        Args:
            x: Tensor[N_total, C] — node features, all graphs concatenated.
            batch: Tensor[N_total] — PyG batch index (graph id per node).

        Returns:
            Tensor[N_total, C]: normalized node features, same shape as x.
        """
        out = torch.zeros_like(x)  # [N_total, C]

        # Every graph is normalized independently, using only its own nodes.
        for graph_idx in batch.unique():
            mask = (batch == graph_idx)
            x_g = x[mask]  # [N_g, C]

            stats = self._graph_stats(x_g)                            # [5*C]
            params = self.conditioner(stats.unsqueeze(0)).squeeze(0)  # [2*C]
            scale, shift = params.chunk(2, dim=-1)                    # each [C]

            # Residual: start from identity, learn corrections on top.
            scale = self.default_scale + scale
            shift = self.default_shift + shift

            # Instance-normalize first, then apply the learned affine transform.
            x_norm = (x_g - x_g.mean(0)) / (x_g.std(0).clamp(min=1e-6))
            out[mask] = x_norm * scale + shift

        return out
