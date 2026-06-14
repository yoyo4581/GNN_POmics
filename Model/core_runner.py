import torch
import numpy as np
from collections import defaultdict
from wandb import Run

from Model.data_model import ModelResults, TranslatorResults, tissue_descriptions, EdgeMask
from visualization.umap import UMAPTransform
from visualization.confusion_matrix import plot_confusion_with_confidence
from visualization.attention_percentile import log_attention_percentiles
from typing import Literal


class EdgeMaskExplainer:
  def _avg_edge_masks(self, masks: list[EdgeMask], edge_index: torch.Tensor, edge_to_idx: dict) -> EdgeMask:
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
    self,
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
      avg_mask = self._avg_edge_masks(masks, edge_index, edge_to_idx) if masks else None

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

  def explain_subsamples(
      self,
      model,
      results: ModelResults,
      loader: DataLoader
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
      batch_size = loader.batch_size
      edge_index = loader.dataset[0].edge_index

      # Post-Hoc explanation
      if results.edge_masks:
        class_edge_masks = {class_idx: results.for_class(class_idx).correct().edge_masks for class_idx in range(num_classes)}
      
      class_results = self._aggregate_class_results(results, class_edge_masks, edge_index)
      return class_results


class CoreRunner:
  def __init__(
      self,
      model: torch.nn.Module,
      optimizer: torch.optim.Optimizer,
      device: torch.device
  ):
    self.model = model
    self.optimizer = optimizer
    self.device = device

  def _forward(self, batch):
    return self.model(batch.x.unsqueeze(1), batch.edge_index, batch.batch)

  def _predict(self, batch):
    return self.model.predict(batch.x.unsqueeze(1), batch.edge_index, batch.batch)

  def _compute_losses(self, model_out, translator_out, y):
        translator_loss, translator_metrics = self.model.head.loss(translator_out, y)
        model_loss, model_metrics = self.model.loss(model_out, y)
        return (model_loss, model_metrics), (translator_loss, translator_metrics)

  def _backward_step(
      self, model_loss: torch.Tensor, translator_loss: torch.Tensor
  ) -> None:
      """Zero grads, backpropagate both losses, then update weights.

      Gradients accumulate naturally on any shared parameters (equivalent
      to jointly optimising model_loss + translator_loss).
      If both losses share a computation graph, swap the first call to
      model_loss.backward(retain_graph=True).
      """
      self.optimizer.zero_grad()
      (model_loss + translator_loss).backward()
      self.optimizer.step()

  # ------------------------------------------------------------------ #
  # Per-batch collection                                                #
  # ------------------------------------------------------------------ #

  @staticmethod
  def _collect_attention(model_out) -> tuple[list, list, list]:
      """Extract detached attention tensors; returns empty lists if absent."""
      edge_indices, attention_weights, node_scores = [], [], []
      if "attention" in model_out and "node_scores" in model_out:
          for (edge_idx, attn), scores in zip(
              model_out["attention"], model_out["node_scores"]
          ):
              edge_indices.append(edge_idx.detach().cpu())  # fixed: was .detach.cpu()
              attention_weights.append(attn.detach().cpu())
              node_scores.append(scores.detach().cpu())
      return edge_indices, attention_weights, node_scores


  # ------------------------------------------------------------------ #
  # Post-epoch visualisation / logging                                  #
  # ------------------------------------------------------------------ #


  def _run_batch(self, batch):
    model_out, translator_out = self._forward(batch)
    y = batch.y.squeeze()
    (model_loss, model_metrics), (translator_loss, translator_metrics) = (
      self._compute_losses(model_out, translator_out, y)
    )
    self._backward_step(model_loss, translator_loss)
    return model_out, translator_out, y, model_loss, model_metrics, translator_loss, translator_metrics


  def forward_pass(self, data_loader):
    batch_size = data_loader.batch_size
    num_batches = len(data_loader)

    # Scalar accumulators
    model_loss_sum = 0.0
    translator_loss_sum = 0.0

    # Collection lists
    model_predictions: list = []
    model_pred_confidence: list = []
    model_emb_list: list = []
    
    translator_predictions: list = []
    translator_pred_confidence: list = []
    translator_emb_list: list = []

    all_labels: list = []
    all_edge_indices: list = []
    all_attention_weights: list = []
    all_node_scores: list = []

    for batch_num, batch in enumerate(data_loader):
        batch = batch.to(self.device)

        (model_out, translator_out, y,
         model_loss, model_metrics,
         translator_loss, translator_metrics) = self._run_batch(batch)

        # Scalars
        model_loss_sum += model_loss.item()
        translator_loss_sum += translator_loss.item()

        # Embeddings & labels
        model_emb_list.append(model_out["graph_embedding"].detach().cpu())
        translator_emb_list.append(translator_out["graph_embedding"].detach().cpu())
        all_labels += y.cpu().tolist()

        # Attention
        edge_idx, attn_w, node_sc = self._collect_attention(model_out)
        all_edge_indices += edge_idx
        all_attention_weights += attn_w
        all_node_scores += node_sc

        # Predictions
        model_predictions += [p.cpu().item() for p in model_metrics["pred_idx"]]
        model_pred_confidence += [
            c.cpu().item() for c in model_metrics["all_confidences"]
        ]

        translator_predictions += [p.cpu().item() for p in translator_metrics["pred_idx"]]
        translator_pred_confidence += [
            c.cpu().item() for c in translator_metrics["all_confidences"]
        ]


    # --- Post-epoch ---
    model_embeddings = np.vstack(model_emb_list)
    translator_embeddings = np.vstack(translator_emb_list)

    edge_masks = [
        EdgeMask(
            edge_attention=attn,
            edge_indices=edge_idx,
            node_scores=node_sc,
            source="Attention",
        )
        for edge_idx, attn, node_sc in zip(
            all_edge_indices, all_attention_weights, all_node_scores
        )
    ]

    return ModelResults(
        predictions=model_predictions,
        pred_confidence=model_pred_confidence,
        edge_masks=edge_masks,
        graph_embeddings=torch.tensor(model_embeddings),
        labels=all_labels,
        dataset_index=list(data_loader.sampler),
        loss=model_loss_sum/num_batches,
    ), TranslatorResults(
        predictions = translator_predictions,
        pred_confidence = translator_pred_confidence,
        graph_embedding=torch.tensor(translator_embeddings),
        labels=all_labels,
        dataset_index = list(data_loader.sampler),
        loss=translator_loss_sum/num_batches
    )