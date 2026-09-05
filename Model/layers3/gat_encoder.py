"""GATEncoder: two-layer GATv2 graph encoder with attention-weighted pooling.

Shape contract:
    input  x:               [N_total, in_channels]  (all graphs in the batch, concatenated)
    input  edge_index:      [2, E]                   (PyG edge list, all graphs concatenated)
    input  batch:            [N_total]               (PyG batch index, graph id per node)
    output graph_embedding:  [B, out_dim]             (B = number of graphs in the batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as scatter_softmax


class GATEncoder(nn.Module):
    """Two-layer GATv2 encoder that pools node features into one embedding per graph.

    GATv2 fixes the static attention problem of GATv1 — attention scores
    depend on both source and target features dynamically, which matters
    when node features carry expression profiles that differ across batches.

    Pooling is attention-weighted rather than a plain mean/sum: each node's
    contribution to its graph embedding is scaled by how much the second
    GAT layer's attention "valued" that node as a source, softmax-normalized
    within its own graph.

    Shapes:
        x:               [N_total, in_channels]
        edge_index:      [2, E]
        batch:           [N_total]
        graph_embedding: [B, out_dim]   (B = number of graphs in the batch)

    Args:
        in_channels: Node feature dimension after normalization.
        hidden_dim: Hidden dimension of the first GAT layer (per head).
        out_dim: Output graph embedding dimension (should match the label
            embedding dimension used downstream).
        heads: Number of attention heads in the first GAT layer.
        dropout: Attention dropout used inside both GAT layers.
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> dict:
        """Encode a batch of graphs into one embedding per graph.

        Args:
            x: Tensor[N_total, in_channels] — node features, all graphs concatenated.
            edge_index: Tensor[2, E] — edge list, all graphs concatenated.
            batch: Tensor[N_total] — PyG batch index (graph id per node).

        Returns:
            dict:
                graph_embedding (Tensor[B, out_dim]): pooled per-graph embedding,
                    B = number of graphs in the batch.
                attention (list[B] of (Tensor[2, E_g], Tensor[E_g])): per-graph
                    edge index and second-layer attention weight, for explainability.
                node_scores (list[B] of Tensor[N_g]): per-graph, per-node pooling
                    weight (attention-derived, softmax-normalized within the graph).
        """
        # Layer 1 — GATv2Conv adds self-loops internally.
        x, (edge_index, attn1) = self.conv1(x, edge_index, return_attention_weights=True)  # x: [N_total, hidden_dim*heads]
        x = self.norm1(x)
        x = F.gelu(x)

        # Layer 2
        x, (edge_index, attn2) = self.conv2(x, edge_index, return_attention_weights=True)  # x: [N_total, out_dim], attn2: [E, 1]
        x = self.norm2(x)

        # attn2: how much each target values its source, per edge.
        # Bin by source node (edge_index[0]) to get one raw score per node.
        node_scores = scatter(attn2.squeeze(), edge_index[0], reduce='sum', dim_size=x.size(0))  # [N_total]

        # Normalize competitively within each graph (softmax over nodes of the same graph).
        node_scores = scatter_softmax(node_scores, batch)  # [N_total]

        # Attention-weighted sum pooling: [N_total, 1] * [N_total, out_dim] -> pooled per graph.
        graph_emb = global_add_pool(node_scores.unsqueeze(-1) * x, batch)  # [B, out_dim]
        node_scores = node_scores.squeeze(-1)  # [N_total] (no-op unless N_total == 1)

        # Split per-graph attention / node-score slices for explainability/visualization.
        edge_graph = batch[edge_index[0]]  # [E] — graph id of each edge's source node
        edge_masks = []
        graph_node_scores = []
        for g in range(batch.max().item() + 1):
            edge_mask = (edge_graph == g)
            node_mask = (batch == g)

            edge_masks.append((edge_index[:, edge_mask].cpu(), attn2[edge_mask].squeeze(-1).cpu()))
            graph_node_scores.append(node_scores[node_mask].cpu())

        return {
            "graph_embedding": graph_emb,       # [B, out_dim]
            "attention": edge_masks,            # list[B] of (Tensor[2, E_g], Tensor[E_g])
            "node_scores": graph_node_scores,   # list[B] of Tensor[N_g]
        }
