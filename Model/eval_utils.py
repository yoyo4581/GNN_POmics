import logging
import random
from collections import defaultdict
from typing import Literal

from pydantic import parse_obj_as
import torch
from torch.utils.data import Subset
from torch.utils.data.sampler import SequentialSampler
from torch_geometric.loader import DataLoader

import numpy as np
import networkx as nx

from Model.wrappers import build_explainer
from Model.data_model import ModelResults, ExplainerResult, tissue_descriptions, EdgeMask
from Dataset.utils import GraphTraverse


from wandb import Run
import wandb

from visualization.umap import UMAPTransform
from visualization.confusion_matrix import plot_confusion_with_confidence
from visualization.chord_diagram import global_layout, subgraph_chord, log_bokeh_figures
from visualization.graph_vis import plot_network, log_network_figure
from visualization.new_matrix_dist import build_matrix
from visualization.plot_correlation import ConsistencyTracker
from visualization.attention_percentile import log_attention_percentiles


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_correct_by_class(labels: torch.Tensor, pred: torch.Tensor, dataset_index: list) -> dict[int, list[int]]:
    """Returns a mapping of class_idx -> list of dataset indices that were correctly predicted."""
    correct_by_class = defaultdict(list)
    for idx in range(len(labels)):
        if labels[idx] == pred[idx]:
            correct_by_class[labels[idx]].append(dataset_index[idx])
    return dict(correct_by_class)


def _subsample_by_class(correct_by_class: dict[int, ModelResults], n: int) -> list[int]:
    """Randomly subsamples up to n indices per class from correctly predicted samples."""
    subsampled = []
    for result in correct_by_class.values():
        indices = result.dataset_index
        subsampled.extend(random.sample(indices, min(n, len(indices))))
    return subsampled


def _run_explainer(model, loader: DataLoader, num_classes: int) -> dict[int, list[EdgeMask]]:
    """Runs the explainer over a loader and returns per-class edge masks."""
    device = _get_device()
    explainer = build_explainer(model)
    class_edge_masks = defaultdict(list)

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        explanation = explainer(
            x=data.x.unsqueeze(1),
            edge_index=data.edge_index,
            batch=data.batch,
        )

        assert data.edge_index.shape[1] % len(data.y) == 0, \
            "Graphs in batch must have an equal number of edges"

        num_edges = data.edge_index.shape[1] // len(data.y)
        for idx in range(len(data.y)):
            edge_mask = explanation.edge_mask[idx * num_edges: (idx + 1) * num_edges].cpu()
            edge_index = explanation.edge_index[idx * num_edges: (idx +1) *num_edges].cpu()
            class_edge_masks[data.y[idx].item()].append(
              EdgeMask(
                edge_attention = edge_mask,
                edge_indices = edge_index,
                source = 'GNNExplainer'
              )
            )

        logger.debug(f"Explained batch {batch_idx} | classes: {data.y.tolist()}")

    return dict(class_edge_masks)

def _avg_edge_masks(masks: list[EdgeMask], edge_index: torch.Tensor, edge_to_idx: dict) -> EdgeMask:
    """
    Average edge attention across masks, aligned to the global edge_index.
    Edges not present in a mask contribute 0 to the average.
    
    Args:
        masks: List of EdgeMask, each with .edge_indices (subset of global) and .attention.
        edge_index: Global edge index, shape [2, num_edges].
    
    Returns:
        Tensor of shape [num_edges] with averaged attention values.
    """
    edge_accumulator = torch.zeros(len(edge_to_idx))
    node_accumulator = torch.zeros(len(masks[0].node_scores))
    for mask in masks:
        local_num_edges = mask.edge_indices.shape[1]
        for local_i in range(local_num_edges):
            src = mask.edge_indices[0, local_i].item()
            dst = mask.edge_indices[1, local_i].item()
            global_i = edge_to_idx.get((src, dst))
            if global_i is not None:
                edge_accumulator[global_i] += mask.edge_attention[local_i].mean()
        
        node_accumulator += mask.node_scores

    n = len(masks) if masks else 1

    return EdgeMask(
      edge_attention = edge_accumulator / n,
      edge_indices = edge_index,
      node_scores = node_accumulator / n,
      source = masks[0].source
    )


def _aggregate_class_results(
  results: ModelResults,
  class_edge_masks: dict[int, list[EdgeMask]],
  edge_index: torch.Tensor
) -> dict[int, ExplainerResult]:
  """
  Takes results and calculates class_based confidence, avg edge mask, accuracy.

  Args:
    results: Contains most data in lists, may not contain EdgeMask.
    class_edge_masks: subsetted edge masks per class.

  Returns:
    dict mapping class_idx -> ExplainerResult
  """
  num_edges = edge_index.shape[1]
  # Build a lookup from (src, dst) -> global edge position
  edge_to_idx = {
      (edge_index[0, i].item(), edge_index[1, i].item()): i
      for i in range(num_edges)
  }

  class_results = {}
  for class_idx, masks in class_edge_masks.items():
    avg_mask = _avg_edge_masks(masks, edge_index, edge_to_idx) if masks else None

    # get results belong to a ground truth label
    results_c = results.for_class(class_idx)

    # of those only find correct ones.
    results_c_correct = results_c.correct()

    avg_confidence = (
      torch.tensor(results_c_correct.pred_confidence).median().item()
      if results_c_correct.pred_confidence else 0.0
    )

    correct = len(results_c_correct.labels)
    total = len(results_c.labels)

    result = ExplainerResult(
            accuracy=correct / total if total > 0 else 0.0,
            avg_confidence=avg_confidence,
            avg_edge_mask=avg_mask,
            edge_masks = masks if masks else None,
            n_samples=total,
            n_correct=correct,
        )
    class_results[class_idx] = result
  
  return class_results




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_subsamples(
    model,
    results: ModelResults,
    loader: DataLoader,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, ExplainerResult]:
    """
    Subsamples correctly predicted examples per class, runs the explainer,
    and returns per-class edge masks.

    Args:
        model: trained GNN model
        results: ModelResults
        loader: Any loader sampler would work.
        batch_size: batch size for the explainer loader

    Returns:
        class_edge_masks: class_idx -> list of edge mask tensors
    """

    num_classes = len(set(results.labels))
    edge_index = loader.dataset[0].edge_index

    # Post-Hoc explanation
    if results.edge_masks:
      class_edge_masks = {class_idx: results.for_class(class_idx).correct().edge_masks for class_idx in range(num_classes)}
    else:
      correct_by_class = {class_idx: results.for_class(class_idx).correct() for class_idx in range(num_classes)}
      subsampled_indices = _subsample_by_class(correct_by_class, num_classes)

      correct_subset = Subset(loader.dataset, subsampled_indices)
      correct_loader = DataLoader(correct_subset, batch_size=batch_size, shuffle=False)

      class_edge_masks = _run_explainer(model, correct_loader, num_classes)
    
    class_results = _aggregate_class_results(results, class_edge_masks, edge_index)
    return class_results






def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    run: Run,
    epoch: int,
    umap: UMAPTransform,
    verbose: bool = True,
) -> ModelResults:
    """
    Runs inference over the validation set.

    Args:
        model: trained GNN model
        validation_loader: DataLoader over the validation set
        verbose: if True, logs progress per batch

    Returns:
        ModelResults with predictions, confidences, and labels
    """
    if verbose:
        logger.info("===== Evaluation =====")

    model.eval()
    device = _get_device()
    num_batches = len(loader)
    predictions, pred_confidence, labels, projections = [], [], [], []
    node_scores_all, attention_weights_all, edge_indices = [], [], []
    total_loss = total_acc = total_correct_conf = total_incorrect_conf = all_conf = 0.0

    for batch_idx, data in enumerate(loader):
      data = data.to(device)
      out, proj = model.predict(
          data.x.unsqueeze(1),
          data.edge_index,
          data.batch,
      )
      loss, acc, conf_avgs, pred_idx, all_confidence = model.loss(out, data.y.squeeze())

      total_loss += loss.item()
      total_acc += acc
      total_correct_conf += conf_avgs['correct']
      total_incorrect_conf += conf_avgs['incorrect']

      projections.append(proj)
      labels += data.y.squeeze().cpu().tolist()

      predictions += [p.cpu().item() for p in pred_idx]
      pred_confidence += [c.cpu().item() for c in all_confidence]

      if "attention" in out:
        for (edge_idx, attn), node_scores in zip(out["attention"], out["node_scores"]):
          edge_indices.append(edge_idx.detach().cpu())
          attention_weights_all.append(attn.detach().cpu())
          node_scores_all.append(node_scores.detach().cpu())
        
      if verbose:
          logger.info(f"Evaluated batch {batch_idx}")

    embeddings = np.vstack(projections)
    umap.plot(embeddings, labels, run, epoch, split='val')
    plot_confusion_with_confidence(
      y_true=labels,
      y_pred=predictions,  
      y_conf=pred_confidence, 
      class_names=tissue_descriptions.values(),
      run = run,
      epoch= epoch, 
      mat_type='val'
    )
    correct_class_attention = defaultdict(list)
    incorrect_class_attention = defaultdict(list)
    for label, prediction, attention in zip(labels, predictions, attention_weights_all):
      if prediction == label:
        correct_class_attention[prediction].append(attention.numpy())
      else:
        incorrect_class_attention[prediction].append(attention.numpy())

    log_attention_percentiles(
      correct_class_attention,
      tissue_descriptions, 
      run, 
      epoch, 
      split="valid/correct")

    log_attention_percentiles(
      incorrect_class_attention,
      tissue_descriptions, 
      run, 
      epoch, 
      split="valid/incorrect")      

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    avg_correct_conf = total_correct_conf / num_batches
    avg_incorrect_conf = total_incorrect_conf / num_batches

    print(f"  Epoch {epoch}  val_loss={avg_loss:.4f}")

    run.log({"valid/loss": avg_loss,
            "valid/acc": avg_acc,
            "valid/correct_conf": avg_correct_conf,
            "valid/incorrect_conf": avg_incorrect_conf},
            step=epoch)

    edge_masks = []
    for edge_index, attention, node_scores in zip(edge_indices, attention_weights_all, node_scores_all):
      edge_masks.append(
        EdgeMask(
          edge_attention = attention,
          edge_indices = edge_index,
          node_scores = node_scores,
          source = 'Attention'
        )
      )

    return ModelResults(
        predictions=predictions,
        pred_confidence=pred_confidence,
        edge_masks = edge_masks,
        graph_embeddings = torch.tensor(embeddings),
        labels=labels,
        dataset_index = list(loader.sampler),
        loss= avg_loss,
        dataset='val'
    )


