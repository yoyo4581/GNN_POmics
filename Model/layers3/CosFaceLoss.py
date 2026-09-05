"""CosFaceLoss: large-margin cosine classifier (CosFace) over graph embeddings.

Shape contract:
    input  embeddings: [B, in_features]
    input  labels:     [B]
    internal cosine:   [B, num_classes]
    output loss:       []  (scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """Large-margin cosine loss (CosFace) with its own learned per-class weight matrix.

    Normalizes both embeddings and class weight vectors onto the unit
    sphere, subtracts a fixed margin `m` from the target class's cosine
    score, then applies scaled cross-entropy. This classifier is trained
    end-to-end with the GAT backbone — unlike LabelEmbeddingHead, which is
    trained separately against BioBERT text embeddings and never backprops
    into the backbone.

    Shapes:
        embeddings: [B, in_features]
        weight:     [num_classes, in_features]
        cosine:     [B, num_classes]
        loss:       []  (scalar)

    Args:
        in_features: Dimensionality of input embeddings.
        num_classes: Number of output classes.
        s: Feature scale applied to logits before cross-entropy.
        m: Cosine margin subtracted from the target class's score.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.35,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.empty(num_classes, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Compute the CosFace loss and prediction/confidence metrics for a batch.

        Args:
            embeddings: Tensor[B, in_features] — graph embeddings.
            labels: Tensor[B] — ground-truth class indices.

        Returns:
            tuple:
                loss (Tensor[]): scalar margin cross-entropy loss.
                metrics (dict): loss, per-bucket cosine-confidence averages,
                    predictions, and per-sample confidences.
        """
        # L2-normalize embeddings and weight vectors onto the unit sphere.
        x_norm = F.normalize(embeddings, p=2, dim=1)    # [B, D]
        w_norm = F.normalize(self.weight, p=2, dim=1)   # [C, D]

        cosine = F.linear(x_norm, w_norm)  # [B, C]

        # Subtract the margin from the ground-truth class column only.
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (cosine - self.m * one_hot)  # [B, C]

        loss = F.cross_entropy(logits, labels)

        pred_idx = cosine.argmax(dim=1)        # [B]
        correct_mask = (pred_idx == labels)     # [B]

        # Similarity of each sample to its predicted class only.
        max_cosine = cosine.gather(1, pred_idx.view(-1, 1)).squeeze(1)  # [B]

        correct_conf = max_cosine[correct_mask].mean().item() if correct_mask.any() else float('nan')
        incorrect_conf = max_cosine[~correct_mask].mean().item() if (~correct_mask).any() else float('nan')

        conf = {
            "correct": correct_conf,
            "incorrect": incorrect_conf,
            "all": max_cosine.mean().item(),
        }

        return loss, {
            "loss": loss,
            "conf_avgs": conf,
            "pred_idx": pred_idx,
            "all_confidences": max_cosine,
        }

    def add_class(self):
        """Grow the classifier by one class, preserving existing class weight vectors.

        The new class's weight row is randomly initialized (Xavier uniform);
        it starts untrained until this class's samples are seen.

        NOTE: not currently called anywhere. `TissueClassificationPipeline_Model3.add_class()`
        only extends `LabelEmbeddingHead`'s label embeddings — it does not call this method,
        so a class added at runtime gets a BioBERT label embedding but no CosFace weight row.
        Wire this in if new classes should also be trainable via the CosFace loss.
        """
        old_weight = self.weight.data
        C, D = old_weight.shape
        new_weight = nn.Parameter(torch.empty(C + 1, D))
        nn.init.xavier_uniform_(new_weight)
        new_weight.data[:C] = old_weight  # preserve existing class vectors
        self.weight = new_weight
