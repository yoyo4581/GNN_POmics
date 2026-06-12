import torch
import torch.nn as nn
import torch.nn.functional as F
 

class LabelEmbeddingHead(nn.Module):
    """
    Maps tissue text descriptions → embedding vectors, then classifies
    by cosine similarity between graph embedding and label embeddings.
 
    At inference, known label embeddings are pre-computed and cached.
    For OOD tissues, you simply add a new label embedding from the text
    description — no retraining required.
 
    Confidence is the cosine similarity to the nearest label, optionally
    calibrated against a held-out set.
 
    Args:
        graph_emb_dim:   output dim of GATEncoder (must match label_emb_dim)
        label_emb_dim:   dimensionality of text embeddings
        temperature:     softmax temperature for similarity scores
    """
 
    def __init__(
        self,
        graph_emb_dim: int = 512,
        label_emb_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
 
        # Project graph embedding into label embedding space
        # (allows graph_emb_dim ≠ label_emb_dim if needed)
        self.projector = nn.Sequential(
            nn.Linear(graph_emb_dim, label_emb_dim),
            nn.LayerNorm(label_emb_dim),
        )
        self.scale = 64
        self.margin = 0.35
 
        # Label embeddings: registered as buffer so they move with .to(device)
        # but are NOT updated by the optimizer
        self.register_buffer("label_embeddings", None)
        self.label_names: list[str] = []
 
    def add_label_embeddings(self, embeddings: torch.Tensor, names: list[str]):
        """
        embeddings: [num_new_classes, label_emb_dim]
        names:      tissue names for the new classes
        """
        new_embeddings = F.normalize(embeddings, dim=-1)

        if self.label_embeddings is None:
            self.label_embeddings = new_embeddings
        else:
            self.label_embeddings = torch.cat(
                [self.label_embeddings, new_embeddings], dim=0
            )

        self.label_names.extend(names)
 
    def forward(self, graph_emb: torch.Tensor):
        """
        graph_emb: [B, graph_emb_dim]
 
        Returns:c
            logits:      [B, num_classes]  — cosine similarity * (1/T)
            confidence:  [B]               — max similarity in [0,1]
            pred_idx:    [B]               — argmax class index
        """
        assert self.label_embeddings is not None, \
            "Call set_label_embeddings() before forward()"

        detached_graph_emb = graph_emb.detach()
        proj = self.projector(detached_graph_emb)                     # [B, D]
        proj = F.normalize(proj, dim=-1)                     # unit sphere
 
        # Cosine similarity against all label embeddings
        # label_embeddings: [num_classes, D]
        cosine_logits = proj @ self.label_embeddings.T  # [B, C]

        # Predicted class
        top2_vals, top2_idx = cosine_logits.topk(2, dim=-1)
        
        pred_idx = top2_idx[:, 0]
        
        #Margin confidence
        confidence = top2_vals[:, 0] - top2_vals[:, 1]

        return {"logits": cosine_logits, 
        "confidence": confidence,
        "pred_idx": pred_idx,
        "graph_embedding": proj
        }


    def loss(self, out: dict, labels: torch.Tensor):
      cosine_logits = out["logits"]

      one_hot = torch.zeros_like(cosine_logits)
      one_hot.scatter_(1, labels.view(-1, 1), 1.0)
      scaled_logits = self.scale * (cosine_logits - self.margin * one_hot)
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

      return loss, {
          "loss": loss,
          "accuracy": acc,
          "conf_avgs": conf,
          "pred_idx": pred_idx,
          "all_confidences": out["confidence"],
      }