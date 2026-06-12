from __future__ import annotations
from dataclasses import dataclass, field, fields
from collections import defaultdict
import uuid
import torch
from typing import Literal, ClassVar

from wandb.sdk.data_types.graph import Edge


tissue_descriptions = {0: 'Adipose Subcutaneous', 1: 'Artery Tibial', 
2: 'Breast Mammary Tissue', 3: 'Cells Cultured Fibroblasts', 
4: 'Esophagus Mucosa', 5: 'Lung', 6: 'Muscle Skeletal', 
7: 'Nerve Tibial', 8: 'Thyroid', 9: 'Whole Blood'}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PrototypeEmbedding:
    class_label: int
    embedding:   torch.Tensor
    centroid: torch.Tensor

@dataclass
class EdgeMask:
  edge_attention: torch.Tensor
  edge_indices: torch.Tensor
  node_scores: torch.Tensor
  source: Literal['Attention', 'GNNExplainer']


@dataclass
class MLResults:
  predictions: list[int]
  pred_confidence: list[float]
  graph_embeddings: torch.Tensor
  labels: list[int]
  dataset_index: list[int]
  loss: float

  def subset(self, indices: list[int]):
      def _index(value):
          if isinstance(value, torch.Tensor):
              return value[indices]
          elif isinstance(value, list):
              return [value[i] for i in indices]
          else:
              return value

      return self.__class__(**{
          f.name: _index(getattr(self, f.name))
          for f in fields(self)
      })

  def where(self, condition: list[bool]) -> ModelResults:
    """Subset by a boolean mask of same length as labels."""
    assert len(condition) == len(self.labels)
    indices = [i for i, keep in enumerate(condition) if keep]    
    return self.subset(indices)

  def correct(self)-> ModelResults:
    """Only correctly predicted samples."""
    return self.where([p == l for p, l in zip(self.predictions, self.labels)])

  def incorrect(self)-> ModelResults:
    """ Only misclassified samples."""
    return self.where([p != l for p, l in zip(self.predictions, self.labels)])

  def for_class(self, label: int)-> ModelResults:
    """Only samples within a specific ground-truth label"""
    return self.where([l == label for l in self.labels])


@dataclass
class ModelResults(MLResults):
  edge_masks: list[EdgeMask]

@dataclass
class TranslatorResults(MLResults):
  pass



@dataclass
class ExplainerResult:
    accuracy: float
    avg_confidence: float
    avg_edge_mask: EdgeMask | None
    edge_masks: list[EdgeMask] | None
    n_samples: int
    n_correct: int

    def __str__(self) -> str:
        edges_retained = (
            (self.avg_edge_mask.edge_attention > 0.5).sum().item()
            if self.avg_edge_mask is not None
            else "N/A"
        )
        return (
            f"accuracy={self.accuracy:.3f} ({self.n_correct}/{self.n_samples})"
            f" | avg_conf={self.avg_confidence:.3f}"
            f" | edges retained (>0.5): {edges_retained}"
        )