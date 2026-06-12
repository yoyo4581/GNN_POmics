"""
Tissue Classification Pipeline
================================
Architecture:
    Raw node features [log2(TPM+1)]
        ↓
    GraphConditionedNorm       — learned, metadata-free batch absorption
        ↓
    GATEncoder + GraphMASK     — relational pattern learning + subgraph explanation
        ↓
    LabelEmbeddingSimilarity   — OOD-capable tissue classification
        ↓
    Prediction + confidence
 
Dependencies:
    torch, torch_geometric, sentence-transformers (or any text encoder)
 
    pip install torch torch_geometric sentence-transformers
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from Model.layers.normalize import GraphConditionedNorm
from Model.layers.gat_encoder import GATEncoder
from Model.layers.label_encoder import LabelEmbeddingHead

from torch_geometric.data import Data
from Model.embeddings import BioBERTEmbeddings
from Model.data_model import tissue_descriptions



 
class TissueClassificationPipeline(nn.Module):
    """
    End-to-end pipeline:
 
        log2(TPM+1) node features
            → GraphConditionedNorm
            → GATEncoder (GATv2)
            → GraphMASKLayer (subgraph explanation)
            → LabelEmbeddingHead (cosine similarity classification)
 
    Training loss:
        total = cross_entropy(logits, labels) + graphmask_penalty
 
    OOD inference:
        Add new label embeddings from text descriptions.
        Confidence score flags low-similarity (unreliable) predictions.
 
    Args:
        in_channels:       number of gene features
        gat_hidden:      GAT hidden dimension
        emb_dim:         shared graph + label embedding dimension
        gat_heads:       number of GAT attention heads
        dropout:         dropout rate
        temperature:     label similarity temperature
        mask_penalty:    GraphMASK sparsity penalty coefficient
    """
 
    def __init__(
        self,
        in_channels: int,
        gat_hidden: int,
        gat_heads: int,
        norm_hidden: int,
        gat_embed_dim: int = 768,
        dropout: float = 0.1,
        margin: float = 0.35,
        temperature: float = 0.07,
    ):
        super().__init__()

        emb_dim = 768
        
        #Normalizer takes in the feature dimension (1 - gene expression), and the number of layers
        self.norm  = GraphConditionedNorm(in_channels=in_channels, hidden_dim=norm_hidden)
        self.gat   = GATEncoder(in_channels, gat_hidden, gat_embed_dim, gat_heads, dropout)
        self.head  = LabelEmbeddingHead(gat_embed_dim, emb_dim, temperature)
        self.num_gat_layers = self.gat.num_layers
        self.margin = 0.35  # CosFace margin
        self.scale = 1 / temperature  # or a separate fixed scale
 
 
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> dict:
 
        # ── Stage 1: graph-conditioned normalization ──────────────────
        x = self.norm(x, batch)
 
        # ── Stage 2: GAT encoding ─────────────────────────────────────
        graph_emb, edge_mask = self.gat(x, edge_index, batch)
 
        # ── Stage 3: label embedding similarity ───────────────────────
        logits, confidence, pred_idx, proj = self.head(graph_emb)
 
        return {
            "logits":       logits,
            "confidence":   confidence,
            "pred_idx":     pred_idx,
            "edge_mask": edge_mask,
        }, proj.detach().cpu().numpy()
 
    def loss(self, out: dict, labels: torch.Tensor):
      cosine_logits = out["logits"]

      # Apply CosFace margin ONLY to target class
      target_logits = cosine_logits[
          torch.arange(labels.size(0)),
          labels
      ]

      cosine_logits[
          torch.arange(labels.size(0)),
          labels
      ] = target_logits - self.margin

      scaled_logits = cosine_logits * self.scale
      
      # Cross entropy
      loss = F.cross_entropy(scaled_logits, labels)

      # Predictions
      pred_idx = out["pred_idx"]

      correct_mask = pred_idx == labels

      correct_conf = (
          out["confidence"][correct_mask].mean().item()
          if correct_mask.any()
          else float("nan")
      )

      incorrect_conf = (
          out["confidence"][~correct_mask].mean().item()
          if (~correct_mask).any()
          else float("nan")
      )

      conf = {
          "correct": correct_conf,
          "incorrect": incorrect_conf,
          "all": out["confidence"].mean().item(),
      }

      acc = correct_mask.float().mean().item()

      return (
          loss,
          acc,
          conf,
          pred_idx,
          out["confidence"],
      )


    # ── Convenience methods ───────────────────────────────────────────
    def set_label_embeddings(self, embeddings: torch.Tensor, names: list[str]):
        self.head.set_label_embeddings(embeddings, names)
 

    @torch.no_grad()
    def predict(self, x: torch.tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Inference with explicit OOD flagging.
 
        Returns dict of lists including logits, confidence, pred_idx.
        """
        self.eval()
        out = self.forward(x, edge_index, batch)
 
        return out

def initialize_model(norm_hidden_channels, gat_hidden_channels, heads, dropout, margin, temperature) -> TissueClassificationPipeline:
  biobert = BioBERTEmbeddings()
  label_embeddings = biobert.get_embeddings(list(tissue_descriptions.values()))
  label_embeddings = torch.tensor(label_embeddings)

  model = TissueClassificationPipeline(
      in_channels=1,
      norm_hidden = norm_hidden_channels,
      gat_hidden=gat_hidden_channels,
      gat_heads=heads,
      dropout=dropout,
      margin= margin,
      temperature=temperature,
  )
  model.set_label_embeddings(label_embeddings, list(tissue_descriptions.values()))

  return model