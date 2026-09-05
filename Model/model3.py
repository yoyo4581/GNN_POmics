"""TissueClassificationPipeline_Model3: end-to-end tissue classification over gene-expression graphs.

Shape contract (B = graphs per batch, N = total nodes in batch, C = gene features):
    input  x:          [N, C]     — log2(TPM+1) node features, all graphs concatenated
    input  edge_index: [2, E]     — PyG edge list, all graphs concatenated
    input  batch:      [N]        — PyG batch index (graph id per node)
    graph_embedding:   [B, gat_embed_dim]  — GATEncoder output (default gat_embed_dim=768)
    label_embedding:   [B, gat_embed_dim]  — LabelEmbeddingHead's projection into BioBERT space
"""

import torch
import torch.nn as nn

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

from Model.embeddings import BioBERTEmbeddings


class TissueClassificationPipeline_Model3(nn.Module):
    """End-to-end pipeline: gene-expression graph -> tissue classification, with a BioBERT-anchored OOD path.

    Pipeline:

        log2(TPM+1) node features                          [N, C]
            -> GraphConditionedNorm                         [N, C]
            -> GATEncoder (GATv2)                            graph_embedding: [B, gat_embed_dim]
            -> two parallel heads, trained on separate losses:
                 (a) CosFaceLoss           — learned per-class weight matrix, trained with the backbone
                 (b) LabelEmbeddingHead    — projects (detached) graph_embedding into BioBERT label
                                             space; classifies by cosine similarity to text-description
                                             embeddings of each tissue. Adding a class = embedding a new
                                             description, no retraining of the backbone required.

        Memory Layer (Prototype_Memory): a per-class replay buffer of recent graph_embeddings,
        periodically agglomerative-clustered into prototype vectors, used for InfoNCE contrastive
        training once warmup is complete.

    Two independent loss paths (both must be stepped by the training loop):
        self.loss(graph_encoder_out, labels)  — trains norm + gat + cosface_loss
            warmup:          cosface_loss
            after warmup:    cosface_loss + gamma * InfoNCE(graph_embedding, prototypes)
        self.head.loss(translator_out, labels) — trains only self.head's projector
            margin cross-entropy over cosine similarity to BioBERT label embeddings

    OOD inference:
        Call add_class(key, description) to embed a new tissue's text description with BioBERT
        and append it to LabelEmbeddingHead's label bank — no retraining required. At inference,
        LabelEmbeddingHead's confidence (top1 - top2 cosine margin) flags low-confidence,
        possibly-unreliable predictions.

    Args:
        label_map: dict mapping class index -> tissue description string.
        memory: pre-constructed Prototype_Memory (owns its own temperature/capacity).
        in_channels: Number of gene features, C.
        norm_hidden: Hidden dim of GraphConditionedNorm's conditioning MLP.
        gat_hidden: GATEncoder's first-layer hidden dimension (per head).
        gat_heads: Number of GATEncoder first-layer attention heads.
        gat_embed_dim: Shared graph embedding / label embedding dimension.
        dropout: Attention dropout used inside GATEncoder.
        margin: CosFace margin (see CosFaceLoss).
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
    ):
        super().__init__()

        emb_dim = 768
        self.label_map = label_map
        n_classes = len(label_map)

        # Normalizer takes in the feature dimension (gene expression) and conditions per-graph.
        self.norm = GraphConditionedNorm(in_channels=in_channels, hidden_dim=norm_hidden)
        self.gat = GATEncoder(in_channels, gat_hidden, gat_embed_dim, gat_heads, dropout)
        self.head = LabelEmbeddingHead(gat_embed_dim, emb_dim)

        self.proto_mem = memory
        self.bio_bert = BioBERTEmbeddings()
        self.set_label_embeddings()

        self.cosface_loss = CosFaceLoss(in_features=gat_embed_dim, num_classes=n_classes)
        self.num_gat_layers = self.gat.num_layers
        self.margin = margin  # CosFace margin
        self.gamma = 0.2

        self.warmup_complete = False

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """Run the full pipeline: normalize -> encode -> translate to label space.

        Args:
            x: Tensor[N, in_channels] — log2(TPM+1) node features, all graphs concatenated.
            edge_index: Tensor[2, E] — edge list, all graphs concatenated.
            batch: Tensor[N] — PyG batch index (graph id per node).

        Returns:
            tuple:
                graph_encoder_out (dict): GATEncoder's output — see GATEncoder.forward().
                    Notably contains "graph_embedding": Tensor[B, gat_embed_dim].
                translator_out (dict): LabelEmbeddingHead's output — see LabelEmbeddingHead.forward().
                    Notably contains "label_embedding": Tensor[B, gat_embed_dim].
        """
        # -- Stage 1: graph-conditioned normalization --
        x = self.norm(x, batch)  # [N, in_channels]

        # -- Stage 2: GAT encoding --
        graph_encoder_out = self.gat(x, edge_index, batch)  # graph_embedding: [B, gat_embed_dim]

        # -- Stage 3: translation into BioBERT label space --
        translator_out = self.head(graph_encoder_out["graph_embedding"])  # label_embedding: [B, gat_embed_dim]

        return graph_encoder_out, translator_out

    def loss(self, out: dict, labels: torch.Tensor):
        """Backbone loss: CosFace, plus prototype-contrastive InfoNCE once warmup is complete.

        NOTE: `out` must be `graph_encoder_out` (the dict returned by GATEncoder / this class's
        forward() first element) — not `translator_out`. Both carry different embeddings but
        `translator_out` uses the key "label_embedding" specifically so passing it here raises
        a clear KeyError instead of silently training on the wrong (detached) embedding.

        Args:
            out: dict containing "graph_embedding": Tensor[B, gat_embed_dim] (from GATEncoder).
            labels: Tensor[B] — ground-truth class indices.

        Returns:
            tuple:
                computed_loss (Tensor[]): scalar total loss for the backbone.
                metrics (dict): CosFaceLoss's metrics plus "cosface_loss", "infonce_loss"
                    (nan until warmup completes), and "total_loss".
        """
        cosface_loss, metrics = self.cosface_loss(out["graph_embedding"], labels)
        computed_loss = cosface_loss

        if self.warmup_complete:
            infonce_loss = self.gamma * self.proto_mem.InfoNCELoss(out["graph_embedding"], labels)
            computed_loss = cosface_loss + infonce_loss

        return computed_loss, {
            **metrics,
            "cosface_loss": cosface_loss.item(),
            "infonce_loss": infonce_loss.item() if self.warmup_complete else float("nan"),
            "total_loss": computed_loss.item(),
        }

    def add_class(self, key, description):
        """Register a new out-of-distribution tissue class without retraining the backbone.

        Embeds `description` with BioBERT and appends it to LabelEmbeddingHead's label
        bank, so the translator head can immediately classify against it by cosine
        similarity. Does not add a row to CosFaceLoss's weight matrix (see
        CosFaceLoss.add_class's note) — this only extends the BioBERT-anchored path.

        Args:
            key: New class index; ignored if already present in `self.label_map`.
            description: Text description of the tissue, embedded via BioBERT.
        """
        if key not in self.label_map:
            self.label_map[key] = description
            embed = self.bio_bert.get_embeddings([description])
            embed = torch.tensor(embed)
            self.head.add_label_embeddings(embed, [description])

    # -- Convenience methods --
    def set_label_embeddings(self):
        """Embed every known tissue description with BioBERT and load them into `self.head`."""
        names = self.label_map.values()
        embeddings = self.bio_bert.get_embeddings(list(names))  # [n_classes, gat_embed_dim]
        embeddings = torch.tensor(embeddings)
        self.head.add_label_embeddings(embeddings, names)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """Run inference (eval mode, no grad).

        Args:
            x: Tensor[N, in_channels] — log2(TPM+1) node features, all graphs concatenated.
            edge_index: Tensor[2, E] — edge list, all graphs concatenated.
            batch: Tensor[N] — PyG batch index (graph id per node).

        Returns:
            tuple: same as forward() — (graph_encoder_out, translator_out).
        """
        self.eval()
        out = self.forward(x, edge_index, batch)

        return out
