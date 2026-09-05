import torch
import numpy as np
from collections import defaultdict
from wandb import Run
import importlib

import Model.data_model
importlib.reload(Model.data_model)

from Model.data_model import ModelResults, TranslatorResults, EdgeMask, ExplainerResult
from torch_geometric.loader import DataLoader

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
        # Batch-convert to Python lists once (fast) instead of .item() per edge
        srcs = mask.edge_indices[0].tolist()
        dsts = mask.edge_indices[1].tolist()

        # Resolve global indices in one pass
        global_indices, local_indices = zip(*[
            (edge_to_idx[k], i)
            for i, k in enumerate(zip(srcs, dsts))
            if k in edge_to_idx
        ]) if any(k in edge_to_idx for k in zip(srcs, dsts)) else ([], [])

        if global_indices:
            g_idx = torch.tensor(global_indices, dtype=torch.long)
            l_idx = torch.tensor(local_indices, dtype=torch.long)
            attn = mask.edge_attention[l_idx]
            if attn.dim() > 1:
                attn = attn.mean(dim=-1)
            edge_accumulator.scatter_add_(0, g_idx, attn)  # vectorized accumulation

        node_accumulator += mask.node_scores

    n = len(masks) if masks else 1
    return EdgeMask(
        edge_attention=edge_accumulator / n,
        edge_indices=edge_index,
        node_scores=node_accumulator / n,
        source=masks[0].source
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
          model: trained GNN model, used for its full label_map (results may
              not contain every class, e.g. a rare class absent from this split).
          results: ModelResults
          loader: Any loader sampler would work.

      Returns:
          class_edge_masks: class_idx -> list of edge mask tensors
      """

      num_classes = len(model.label_map)
      edge_index = loader.dataset[0].edge_index

      # Post-Hoc explanation
      if results.edge_masks:
        class_edge_masks = {class_idx: results.for_class(class_idx).correct().edge_masks for class_idx in range(num_classes)}
      else:
        class_edge_masks = {class_idx: [] for class_idx in range(num_classes)}

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
    cosface_loss_sum = 0.0
    infonce_loss_sum = 0.0
    infonce_batch_count = 0

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
        cosface_loss_sum += model_metrics["cosface_loss"]
        # infonce_loss is nan until warmup completes -- average only over the
        # batches where it was actually computed, instead of poisoning the mean.
        if not np.isnan(model_metrics["infonce_loss"]):
            infonce_loss_sum += model_metrics["infonce_loss"]
            infonce_batch_count += 1

        # Embeddings & labels
        model_emb_list.append(model_out["graph_embedding"].detach().cpu())
        translator_emb_list.append(translator_out["label_embedding"].detach().cpu())
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
        cosface_loss=cosface_loss_sum/num_batches,
        infonce_loss=infonce_loss_sum/infonce_batch_count if infonce_batch_count > 0 else float("nan"),
    ), TranslatorResults(
        predictions = translator_predictions,
        pred_confidence = translator_pred_confidence,
        graph_embeddings=torch.tensor(translator_embeddings),
        labels=all_labels,
        dataset_index = list(data_loader.sampler),
        loss=translator_loss_sum/num_batches
    )

  # ------------------------------------------------------------------ #
  # Explanation fidelity                                                #
  # ------------------------------------------------------------------ #

  @torch.no_grad()
  def fidelity_check(self, dataset, top_frac: float = 0.1, max_samples: int = 100) -> dict:
    """
    Edge-ablation fidelity check for the model's attention-based subgraph explanation.

    For up to `max_samples` graphs: run the normal forward pass to get a
    baseline prediction and this graph's (second-layer) attention weights,
    then rerun the model twice more on the SAME graph with its edges
    restricted to either the top `top_frac` by attention ("topk") or
    everything else ("complement"). A genuinely discriminative attention map
    should mean high topk accuracy (the kept subgraph is sufficient on its
    own) and low complement accuracy (the discarded subgraph isn't) --
    conditioned on the samples the model actually classifies correctly in
    the first place, since ablating an already-wrong prediction says
    nothing about explanation quality.

    Each graph is run as its own single-graph "batch" (batch index all
    zeros), so its attention/edge_index never need batch-offset bookkeeping.

    Caveat: GATv2Conv adds self-loops internally on every forward call,
    regardless of what edge_index is passed in, so a node can always attend
    to itself even in the "complement" ablation. Self-loop edges are
    excluded from the top-k/complement ranking below since they aren't part
    of the actual gene-interaction graph, but they still mean complement
    accuracy is not a true zero-information baseline.

    Args:
        dataset: iterable of individual PyG Data graphs (e.g. loader.dataset).
        top_frac: Fraction of highest-attention (non-self-loop) edges kept
            in the "topk" ablation.
        max_samples: Cap on graphs examined -- each one costs up to 3
            forward passes, so this is a periodic diagnostic, not a
            per-batch metric.

    Returns:
        dict: baseline_acc (over all examined graphs), topk_acc and
            complement_acc (over the baseline-correct subset only),
            n_samples, n_baseline_correct.
    """
    self.model.eval()

    n_total = 0
    n_baseline_correct = 0
    n_topk_correct = 0
    n_complement_correct = 0

    for data in dataset:
      if n_total >= max_samples:
        break
      n_total += 1

      x = data.x.unsqueeze(1).to(self.device)
      edge_index = data.edge_index.to(self.device)
      label = int(data.y.squeeze().item())
      batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
      label_tensor = torch.tensor([label], device=self.device)

      model_out, _ = self.model.predict(x, edge_index, batch)
      _, metrics = self.model.cosface_loss(model_out["graph_embedding"], label_tensor)
      if metrics["pred_idx"].item() != label:
        continue
      n_baseline_correct += 1

      local_edge_index, attn = model_out["attention"][0]
      local_edge_index = local_edge_index.to(self.device)
      attn = attn.to(self.device)

      non_self_loop = local_edge_index[0] != local_edge_index[1]
      local_edge_index = local_edge_index[:, non_self_loop]
      attn = attn[non_self_loop]
      if attn.numel() == 0:
        continue

      k = max(1, int(top_frac * attn.numel()))
      order = torch.argsort(attn, descending=True)
      topk_edge_index = local_edge_index[:, order[:k]]
      complement_edge_index = local_edge_index[:, order[k:]]

      for ablated_edge_index, is_topk in [(topk_edge_index, True), (complement_edge_index, False)]:
        ablated_out, _ = self.model.predict(x, ablated_edge_index, batch)
        _, ablated_metrics = self.model.cosface_loss(ablated_out["graph_embedding"], label_tensor)
        if ablated_metrics["pred_idx"].item() == label:
          if is_topk:
            n_topk_correct += 1
          else:
            n_complement_correct += 1

    return {
        "baseline_acc": n_baseline_correct / n_total if n_total > 0 else float("nan"),
        "topk_acc": n_topk_correct / n_baseline_correct if n_baseline_correct > 0 else float("nan"),
        "complement_acc": n_complement_correct / n_baseline_correct if n_baseline_correct > 0 else float("nan"),
        "n_samples": n_total,
        "n_baseline_correct": n_baseline_correct,
    }