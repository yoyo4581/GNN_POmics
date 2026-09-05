import networkx as nx
from wandb import Run
import numpy as np
from collections import defaultdict
from typing import Literal

from Dataset.utils import GraphTraverse
from Model.data_model import EdgeMask, ExplainerResult

from visualization.umap import UMAPTransform
from visualization.chord_diagram import global_layout
from visualization.attention_percentile import log_attention_percentiles
from visualization.confusion_matrix import plot_confusion_with_confidence
from visualization.plot_correlation import ConsistencyTracker
from visualization.new_matrix_dist import build_matrix


class Visualizer:
  def __init__(
    self, 
    run: Run, 
    graph: nx.DiGraph, 
    graph_traverse: GraphTraverse,
    model_umap: UMAPTransform,
    translator_umap: UMAPTransform
  ):

    self.run = run
    self.graph = graph
    self.graph_traverse = graph_traverse
    self.model_umap = model_umap
    self.translator_umap = translator_umap

  def _fit_or_plot_umap(
      self,
      umap: UMAPTransform,
      embeddings: np.ndarray,
      labels: list,
      epoch: int,
      split: str,
  ) -> None:
      if not umap.is_fitted():
          umap.fit(embeddings, labels)
      umap.plot(embeddings, labels, self.run, epoch, split=split)

  def _log_attention(
      self,
      label_map: dict,
      labels: list,
      predictions: list,
      attention_weights: list,
      epoch: int,
      split_prefix: str,
  ) -> None:
      correct_attn: dict = defaultdict(list)
      incorrect_attn: dict = defaultdict(list)
      for label, pred, attn in zip(labels, predictions, attention_weights):
          bucket = correct_attn if pred == label else incorrect_attn
          bucket[pred].append(attn.numpy())

      log_attention_percentiles(
          correct_attn, label_map, self.run, epoch,
          split=f"{split_prefix}/correct",
      )
      log_attention_percentiles(
          incorrect_attn, label_map, self.run, epoch,
          split=f"{split_prefix}/incorrect",
      )

  def log_subgraph_edge_masks(
    self,
    label_map: dict,
    class_results: dict[int, ExplainerResult],
    cons_tracker: ConsistencyTracker,
    epoch: int,
    split: Literal['train', 'val', 'test'],
    output_dir: str = "outputs"
  ):

    assert cons_tracker.split == split, f"Tracker split '{cons_tracker.split}' does not match expected split '{split}'"

    # NOTE: per-class networkX renders (plot_network / log_network_figure) are no
    # longer pushed to W&B -- they were too noisy to be a useful training signal.
    # The implementations still live in graph_vis.py for ad-hoc/local use.

    correlation_line_plot = cons_tracker.update(epoch, class_results)
    
    matrix_fig = build_matrix(class_results, label_map)
    matrix_fig.write_html(f"./{output_dir}/{split}_matrix_{epoch}.html")
  
  def visualize(
      self,
      epoch: int,
      embeddings: np.ndarray,
      label_map: dict,
      labels: list,
      predictions: list,
      pred_confidence: list,
      split: Literal['train', 'val', 'test'],
      layer: Literal['model', 'translator'],
      edge_masks: list[EdgeMask] = [],
  ) -> None:
      if layer == 'model':
          self._fit_or_plot_umap(
              self.model_umap, embeddings, labels, epoch=epoch, split=f"{split}/{layer}"
          )
      
      if layer == 'translator':
          self._fit_or_plot_umap(
            self.translator_umap, embeddings, labels, epoch=epoch, split=f"{split}/{layer}"
          )

      plot_confusion_with_confidence(
          y_true=labels,
          y_pred=predictions,
          y_conf=pred_confidence,
          class_names=list(label_map.values()),
          run=self.run,
          epoch=epoch,
          mat_type=split,
      )
      if edge_masks:
          attention_weights = [edge_mask.edge_attention for edge_mask in edge_masks]
          self._log_attention(
              label_map, labels, predictions, attention_weights, epoch, split_prefix=split
          )