import torch
import torch.nn as nn
import torch.nn.functional as F
 
from Model.layers3.normalize import GraphConditionedNorm
import importlib
import Model.layers3.gat_encoder
importlib.reload(Model.layers3.gat_encoder)
from Model.layers3.gat_encoder import GATEncoder

import Model.layers3.label_encoder
importlib.reload(Model.layers3.label_encoder)
from Model.layers3.label_encoder import LabelEmbeddingHead

import Model.layers3.CosFaceLoss
importlib.reload(Model.layers3.CosFaceLoss)
from Model.layers3.CosFaceLoss import CosFaceLoss

import Model.layers3.proto_mem
importlib.reload(Model.layers3.proto_mem)
from Model.layers3.proto_mem import Prototype_Memory

from torch_geometric.data import Data
from Model.embeddings import BioBERTEmbeddings
from Model.data_model import tissue_descriptions


class TissueClassificationPipeline_Model3(nn.Module):
    """
    End-to-end pipeline:
 
        log2(TPM+1) node features
            → GraphConditionedNorm -> normalized feature vectors
            → GATEncoder (GATv2) -> graph embeddings
            → Memory Layer -> Agglomerative clustering -> prototype vectors (also behaves as Replay layer)
        -> Translation Layer (MLP Prototypes) -> BioBERT
 
    Training loss:
      Warmup Epochs
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
        label_map: dict,
        memory: Prototype_Memory,
        in_channels: int,
        norm_hidden: int,
        gat_hidden: int,
        gat_heads: int,
        gat_embed_dim: int = 768,
        dropout: float = 0.1,
        margin: float = 0.35,
        temperature: float = 0.07,
    ):
        super().__init__()

        emb_dim = 768
        self.label_map = label_map
        n_classes = len(label_map)
        
        #Normalizer takes in the feature dimension (1 - gene expression), and the number of layers
        self.norm  = GraphConditionedNorm(in_channels=in_channels, hidden_dim=norm_hidden)
        self.gat   = GATEncoder(in_channels, gat_hidden, gat_embed_dim, gat_heads, dropout)
        self.head  = LabelEmbeddingHead(gat_embed_dim, emb_dim, temperature)

        self.proto_mem = memory
        self.bio_bert = BioBERTEmbeddings()
        self.set_label_embeddings()

        self.cosface_loss = CosFaceLoss(in_features = gat_embed_dim, num_classes = n_classes)
        self.num_gat_layers = self.gat.num_layers
        self.margin = 0.35  # CosFace margin
        self.gamma = 0.2
        self.temperature = temperature

        self.warmup_complete = False

    def parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """
        Backbone and translation head have different learning rates.
        """
        backbone_params = [
          *self.norm.parameters(),
          *self.gat.parameters(),
          *self.cosface_loss.parameters(),
        ]
        return [
          {"params": backbone_params, "lr": backbone_lr},
          {"params": list(self.head.parameters()), "lr": head_lr}
        ]

 
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> dict:
 
        # ── Stage 1: graph-conditioned normalization ──────────────────
        x = self.norm(x, batch)
 
        # ── Stage 2: GAT encoding ─────────────────────────────────────
        graph_encoder_out = self.gat(x, edge_index, batch)
 
        # ── Stage 3: Translator Training ───────────────────────
        translator_out = self.head(graph_encoder_out["graph_embedding"])
 
        return graph_encoder_out, translator_out

    def loss(self, out: dict, labels: torch.Tensor):
        if self.warmup_complete:
          cosface_loss, metrics = self.cosface_loss(out["graph_embedding"], labels)
          infonce_loss = self.gamma * self.proto_mem.InfoNCELoss(out["graph_embedding"], labels)
          computed_loss = cosface_loss + infonce_loss
        else:
          cosface_loss, metrics = self.cosface_loss(out["graph_embedding"], labels)
          computed_loss = cosface_loss

        return computed_loss, {
            **metrics,
            "cosface_loss": cosface_loss.item(),
            "infonce_loss": infonce_loss.item() if self.warmup_complete else float("nan"),
            "total_loss": computed_loss.item(),
        }

    def add_class(self, key, description):
        if key not in self.label_map:
          self.label_map[key] = description
          embed = self.bio_bert.get_embeddings(list(description))
          embed = torch.tensor(embed)
          self.head.add_label_embeddings(embed)


    # ── Convenience methods ───────────────────────────────────────────
    def set_label_embeddings(self):
        names = self.label_map.values()
        embeddings = self.bio_bert.get_embeddings(list(names))
        embeddings = torch.tensor(embeddings)
        self.head.add_label_embeddings(embeddings, names)
 

    @torch.no_grad()
    def predict(self, x: torch.tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Inference with explicit OOD flagging.
 
        Returns dict of lists including logits, confidence, pred_idx.
        """
        self.eval()
        out = self.forward(x, edge_index, batch)
 
        return out



