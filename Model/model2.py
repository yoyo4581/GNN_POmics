import torch
import torch.nn as nn
import torch.nn.functional as F
 
from Model.layers.normalize import GraphConditionedNorm
import importlib
import Model.layers2.gat_encoder
importlib.reload(Model.layers2.gat_encoder)
from Model.layers2.gat_encoder import GATEncoder
from Model.layers.label_encoder import LabelEmbeddingHead

from torch_geometric.data import Data
from Model.embeddings import BioBERTEmbeddings
from Model.data_model import tissue_descriptions
from Model.model import TissueClassificationPipeline


class TissueClassificationPipeline_Model2(TissueClassificationPipeline):
  """
  Similar to the base class model except we'll have to define a new GAT Encoder.
  The GATEncoder in this rendition will employ graph pruning between its layers.
  The hope is that graph embeddings will become 
  """
  def __init__(
        self,
        in_channels: int,
        gat_hidden: int,
        gat_heads: int,
        norm_hidden: int,
        gat_embed_dim: int = 768,
        dropout: float = 0.1,
        topk_edges: int = 1000,
        margin: float = 0.35,
        temperature: float = 0.07,
    ):
        super().__init__(
            in_channels=in_channels,
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            norm_hidden=norm_hidden,
            gat_embed_dim=gat_embed_dim,
            dropout=dropout,
            margin=margin,
            temperature=temperature,
        )

        # Now override just the GAT with the pruning version
        self.gat = GATEncoder(in_channels, gat_hidden, gat_embed_dim, gat_heads, dropout, topk_edges)
  
  def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> dict:
 
        # ── Stage 1: graph-conditioned normalization ──────────────────
        x = self.norm(x, batch)

        # ── Stage 2: GAT encoding ─────────────────────────────────────
        graph_emb, batch_node_scores, (gat_edge_index, edge_attention) = self.gat(x, edge_index, batch)

        # ── Stage 3: label embedding similarity ───────────────────────
        logits, confidence, pred_idx, proj = self.head(graph_emb)

        edge_graph = batch[gat_edge_index[0]]
        edge_masks = []
        graph_node_scores = []
        for g in range(batch.max().item() + 1):
          edge_mask = (edge_graph == g)
          node_mask = (batch == g)
          
          edge_masks.append((gat_edge_index[:, edge_mask].cpu(), edge_attention[edge_mask].squeeze(-1).cpu()))
          graph_node_scores.append(batch_node_scores[node_mask].cpu())
 
        return {
            "logits":       logits,
            "confidence":   confidence,
            "pred_idx":     pred_idx,
            "attention":   edge_masks,
            "node_scores": graph_node_scores
        }, proj.detach().cpu().numpy()



def initialize_model(norm_hidden_channels, gat_hidden_channels, heads, dropout, topk_edges, margin, temperature) -> TissueClassificationPipeline:
  biobert = BioBERTEmbeddings()
  label_embeddings = biobert.get_embeddings(list(tissue_descriptions.values()))
  label_embeddings = torch.tensor(label_embeddings)

  model = TissueClassificationPipeline_Model2(
      in_channels=1,
      norm_hidden = norm_hidden_channels,
      gat_hidden=gat_hidden_channels,
      gat_heads=heads,
      dropout=dropout,
      topk_edges = topk_edges,
      margin= margin,
      temperature=temperature,
  )
  model.set_label_embeddings(label_embeddings, list(tissue_descriptions.values()))

  return model