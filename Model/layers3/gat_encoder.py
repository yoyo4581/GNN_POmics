
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import degree, scatter
from torch_geometric.utils import softmax as scatter_softmax



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
        topk_edges: int = 1000
    ):
        super().__init__()

        self.num_layers = 2
        self.topk_edges = topk_edges
 
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

    def prune(self, x, edge_index, attention, batch, topK):
        attn_scalar = attention.mean(dim=-1)

        deg_src = degree(edge_index[0], num_nodes=x.size(0)).clamp(min=1)
        deg_dst = degree(edge_index[1], num_nodes=x.size(0)).clamp(min=1)
        # symmetric normalization: 1 / sqrt(d_i * d_j)
        norm = 1.0 / torch.sqrt(deg_src[edge_index[0]] * deg_dst[edge_index[1]])
        attn_scalar = attn_scalar * norm
        
        num_graphs = batch.max().item() + 1
        edges_per_graph = attn_scalar.size(0) // num_graphs
        k = min(topK, attn_scalar.size(0))

        # Changes attn_scalar from [Total E] to [G, E]
        scores = attn_scalar.view(num_graphs, edges_per_graph)
        _, local_indices = torch.topk(scores, k, dim=1) # (num_graphs, k)

        mask = torch.zeros(num_graphs, edges_per_graph, dtype=torch.bool, device=x.device)
        mask.scatter_(1, local_indices, True)
        mask = mask.view(-1)
        
        # Zero out pruned entries rather than removing them
        attn_scalar = attn_scalar * mask.float()
        
        return attn_scalar, edge_index


    def forward(self, x, edge_index, batch):
        # Layer 1, adds self loops, attn1 refers to edges.
        x, (edge_index, attn1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.norm1(x)
        x = F.gelu(x)
 
        # Layer 2
        x, (edge_index, attn2) = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.norm2(x)

        # attn2: how much each target values its source, per edge
        # bin by source (edge_index[0]) — how much is each node valued across all graphs
        node_scores = scatter(attn2.squeeze(), edge_index[0], reduce='sum', dim_size=x.size(0))

        # normalize competitively within each graph
        node_scores = scatter_softmax(node_scores, batch)

        # pool, (B, F) = (32, 768)
        graph_emb = global_add_pool(node_scores.unsqueeze(-1) * x, batch)
        node_scores = node_scores.squeeze(-1)

        edge_graph = batch[edge_index[0]]
        edge_masks = []
        graph_node_scores = []
        for g in range(batch.max().item() + 1):
          edge_mask = (edge_graph == g)
          node_mask = (batch == g)
          
          edge_masks.append((edge_index[:, edge_mask].cpu(), attn2[edge_mask].squeeze(-1).cpu()))
          graph_node_scores.append(node_scores[node_mask].cpu())
 
        return {"graph_embedding": graph_emb, 
        "attention":   edge_masks,
        "node_scores": graph_node_scores
        }
 
 