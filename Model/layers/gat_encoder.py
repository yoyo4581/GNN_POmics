
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATEncoder(nn.Module):
    """
    Two-layer GATv2 encoder.
 
    GATv2 fixes the static attention problem of GATv1 — attention scores
    depend on both source and target features dynamically, which matters
    when node features carry expression profiles that differ across batches.
 
    Args:
        in_channels:   node feature dim after normalization
        hidden_dim:    internal GAT dim
        out_dim:       graph embedding dim (should match label embedding dim)
        heads:         number of attention heads
        dropout:       attention dropout
    """
 
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_dim: int = 768,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = 2
 
        self.conv1 = GATv2Conv(
            in_channels, hidden_dim,
            heads=heads, dropout=dropout, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, out_dim,
            heads=1, dropout=dropout, concat=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, edge_index, batch):
        # Layer 1
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
 
        # Layer 2
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.norm2(x)
 
        # Graph-level embedding via mean pooling
        graph_emb = global_mean_pool(x, batch)          # [B, out_dim]
 
        return graph_emb, x, (attn1, attn2)
 
 