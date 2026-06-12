import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """Large Margin Cosine Loss (CosFace).

    Normalises both embeddings and class weight vectors, then
    subtracts a fixed margin m from the target-class cosine score
    before applying scaled cross-entropy.

    Args:
        in_features:  dimensionality of input embeddings
        num_classes:  number of output classes
        s:            feature scale (temperature), default 64
        m:            cosine margin, default 0.35
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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # L2-normalise embeddings and weight vectors
        x_norm = F.normalize(embeddings, p=2, dim=1)       # (B, D)
        w_norm = F.normalize(self.weight, p=2, dim=1)   # (C, D)

        # Cosine similarities: (B, C)
        cosine = F.linear(x_norm, w_norm)

        # Subtract margin from the ground-truth class column only
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.s * (cosine - self.m * one_hot)

        loss = F.cross_entropy(logits, labels)

        # predictions
        pred_idx = cosine.argmax(dim=1) # (B,)

        correct_mask = (pred_idx == labels)

        # similarity of each sample to its predicted class only
        max_cosine = cosine.gather(1, pred_idx.view(-1, 1)).squeeze(1)  # (B,)

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
        old_weight = self.weight.data
        C, D = old_weight.shape
        new_weight = nn.Parameter(torch.empty(C + 1, D))
        nn.init.xavier_uniform_(new_weight)
        new_weight.data[:C] = old_weight          # preserve existing class vectors
        self.weight = new_weight