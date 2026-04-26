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
 
from .layers.normalize import GraphConditionedNorm
from .layers.gat_encoder import GATEncoder
from .layers.label_encoder import LabelEmbeddingHead

from torch_geometric.data import Data
 
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
        gat_hidden: int = 256,
        emb_dim: int = 512,
        gat_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
        norm_hidden: int = 128,
    ):
        super().__init__()
 
        self.norm  = GraphConditionedNorm(in_channels=in_channels, hidden_dim=norm_hidden)
        self.gat   = GATEncoder(in_channels, gat_hidden, emb_dim, gat_heads, dropout)
        self.head  = LabelEmbeddingHead(emb_dim, emb_dim, temperature)
        self.num_gat_layers = self.gat.num_layers
 
 
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> dict:
 
        # ── Stage 1: graph-conditioned normalization ──────────────────
        x = self.norm(x, batch)
 
        # ── Stage 2: GAT encoding ─────────────────────────────────────
        graph_emb, node_emb, attn_weights = self.gat(x, edge_index, batch)
 
        # ── Stage 3: label embedding similarity ───────────────────────
        logits, confidence, pred_idx = self.head(graph_emb)
 
        return {
            "logits":       logits,
            "confidence":   confidence,
            "pred_idx":     pred_idx,
        }
 
    def loss(self, out: dict, labels: torch.Tensor) -> tuple[torch.Tensor, float, dict]:
        ce_loss =  F.cross_entropy(out["logits"], labels)
        correct_mask   = out["pred_idx"] == labels
        correct_conf   = out["confidence"][correct_mask].mean().item()
        incorrect_conf = out["confidence"][~correct_mask].mean().item()

        conf = {
            "correct":   correct_conf,
            "incorrect": incorrect_conf,
            "all":       out["confidence"].mean().item(),
        }

        acc = (correct_mask).float().mean().item()
        
        return ce_loss, acc, conf


    # ── Convenience methods ───────────────────────────────────────────
    def set_label_embeddings(self, embeddings: torch.Tensor, names: list[str]):
        self.head.set_label_embeddings(embeddings, names)
 

    @torch.no_grad()
    def predict(self, x: torch.tensor, edge_index: torch.Tensor, batch: torch.Tensor, confidence_threshold: float = 0.5):
        """
        Inference with explicit OOD flagging.
 
        Returns list of dicts with:
            tissue:     predicted tissue name
            confidence: similarity score
            reliable:   False if confidence < threshold
            subgraph:   edge indices of top-k explanatory edges
        """
        self.eval()
        out = self.forward(x, edge_index, batch)
 
        results = []
        for i in range(out["confidence"].shape[0]):
            idx  = out["pred_idx"][i].item()
            conf = out["confidence"][i].item()
 
            # Top-k edges for this graph's explanation
            # (GraphMASK mask is over all edges; filter by graph if batched)
            results.append({
                "tissue":     self.head.label_names[idx] if self.head.label_names else str(idx),
                "confidence": conf,
                "reliable":   conf >= confidence_threshold,
            })

        return results