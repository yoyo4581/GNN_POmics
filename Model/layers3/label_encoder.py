"""LabelEmbeddingHead: maps graph embeddings into a text-anchored label space.

Shape contract:
    input  graph_emb:       [B, graph_emb_dim]
    output logits:          [B, num_classes]
    output confidence:      [B]
    output pred_idx:        [B]
    output label_embedding: [B, label_emb_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEmbeddingHead(nn.Module):
    """Projects graph embeddings into BioBERT label-description space and classifies by cosine similarity.

    At inference, known label embeddings (one per tissue, from BioBERT text
    descriptions) are pre-computed and cached in `label_embeddings`. Adding a
    new, out-of-distribution tissue just means embedding its text description
    and appending it via `add_label_embeddings` — no retraining required.

    The incoming graph embedding is detached before projection: this head is
    trained on its own loss (`loss()`), independently of the GAT backbone /
    CosFace loss, so gradients from this head never flow back into the encoder.

    Confidence is a margin score: the gap between the top-1 and top-2 cosine
    similarities, so a prediction close to a second label reads as uncertain.

    Shapes:
        graph_emb:        [B, graph_emb_dim]
        label_embeddings: [C, label_emb_dim]  (C = number of known classes)
        logits:           [B, C]
        confidence:       [B]
        pred_idx:         [B]
        label_embedding:  [B, label_emb_dim]

    Args:
        graph_emb_dim: Output dimension of the GAT encoder.
        label_emb_dim: Dimensionality of the BioBERT text embeddings.
    """

    def __init__(
        self,
        graph_emb_dim: int = 512,
        label_emb_dim: int = 512,
    ):
        super().__init__()

        # Project graph embedding into label embedding space
        # (allows graph_emb_dim != label_emb_dim if needed).
        self.projector = nn.Sequential(
            nn.Linear(graph_emb_dim, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
        )
        self.scale = 64
        self.margin = 0.35

        # Label embeddings: registered as a buffer so they move with .to(device)
        # but are NOT updated by the optimizer.
        self.register_buffer("label_embeddings", None)
        self.label_names: list[str] = []

    def add_label_embeddings(self, embeddings: torch.Tensor, names: list[str]):
        """Append new (L2-normalized) label embeddings, e.g. for OOD classes.

        Args:
            embeddings: Tensor[C_new, label_emb_dim] — new class text embeddings.
            names: list[C_new] — tissue names, row-aligned with `embeddings`.
        """
        new_embeddings = F.normalize(embeddings, dim=-1)  # [C_new, label_emb_dim]

        if self.label_embeddings is None:
            self.label_embeddings = new_embeddings
        else:
            self.label_embeddings = torch.cat(
                [self.label_embeddings, new_embeddings], dim=0
            )  # [C_total, label_emb_dim]

        self.label_names.extend(names)

    def forward(self, graph_emb: torch.Tensor) -> dict:
        """Classify graph embeddings by cosine similarity to known label embeddings.

        Args:
            graph_emb: Tensor[B, graph_emb_dim] — pooled graph embeddings from GATEncoder.

        Returns:
            dict:
                logits (Tensor[B, C]): cosine similarity to every known label.
                confidence (Tensor[B]): top1 - top2 similarity margin.
                pred_idx (Tensor[B]): argmax predicted class index.
                label_embedding (Tensor[B, label_emb_dim]): L2-normalized
                    projection of graph_emb into label space. Named distinctly
                    from GATEncoder's "graph_embedding" so the two can't be
                    confused when picking a dict to pass into model.loss().
        """
        assert self.label_embeddings is not None, \
            "Call set_label_embeddings() before forward()"

        detached_graph_emb = graph_emb.detach()  # [B, graph_emb_dim] — stop-gradient into backbone
        proj = self.projector(detached_graph_emb)  # [B, label_emb_dim]
        proj = F.normalize(proj, dim=-1)            # unit sphere

        # Cosine similarity against all label embeddings: [B, D] @ [D, C] -> [B, C]
        cosine_logits = proj @ self.label_embeddings.T

        # Predicted class + margin-based confidence.
        top2_vals, top2_idx = cosine_logits.topk(2, dim=-1)  # each [B, 2]
        pred_idx = top2_idx[:, 0]                             # [B]
        confidence = top2_vals[:, 0] - top2_vals[:, 1]        # [B]

        return {
            "logits": cosine_logits,      # [B, C]
            "confidence": confidence,     # [B]
            "pred_idx": pred_idx,         # [B]
            "label_embedding": proj,      # [B, label_emb_dim]
        }

    def loss(self, out: dict, labels: torch.Tensor):
        """CosFace-style margin cross-entropy over the cosine logits from forward().

        Args:
            out: dict returned by forward() — must contain "logits", "confidence", "pred_idx".
            labels: Tensor[B] — ground-truth class indices.

        Returns:
            tuple:
                loss (Tensor[]): scalar margin cross-entropy loss.
                metrics (dict): loss, accuracy, per-bucket confidence averages,
                    predictions, and per-sample confidences.
        """
        cosine_logits = out["logits"]  # [B, C]

        one_hot = torch.zeros_like(cosine_logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        scaled_logits = self.scale * (cosine_logits - self.margin * one_hot)  # [B, C]
        loss = F.cross_entropy(scaled_logits, labels)

        pred_idx = out["pred_idx"]                # [B]
        correct_mask = pred_idx == labels          # [B]

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

        return loss, {
            "loss": loss,
            "accuracy": acc,
            "conf_avgs": conf,
            "pred_idx": pred_idx,
            "all_confidences": out["confidence"],
        }
