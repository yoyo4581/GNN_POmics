import torch
import importlib
import Model.core_utils

importlib.reload(Model.core_utils)
from Model.core_utils import CoreRunner, EdgeMaskExplainer
from Model.data_model import ModelResults, ExplainerResult, EdgeMask, TranslatorResults
from torch_geometric.loader import DataLoader

from visualization.visualizer import Visualizer
from visualization.plot_correlation import ConsistencyTracker


from collections import defaultdict

import numpy as np
from typing import Literal
import networkx as nx
from wandb import Run

    

class Evaluater(CoreRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        run: Run,
        visualizer: Visualizer,
        consistency_tracker: ConsistencyTracker
    ):
        super().__init__(model, optimizer, device)
        self.consistency_tracker = consistency_tracker
        self.run = run
        self.visualizer = visualizer

        self.split_name = 'val'
        self.edge_explainer = EdgeMaskExplainer()


    def _log_metrics(self, epoch: int, avg_metrics: dict) -> None:
        key = f"{self.split_name}/model_loss"
        print(f"  Epoch {epoch}  model_loss={avg_metrics[key]:.4f}")
        self.run.log(avg_metrics, step=epoch)


    def _run_batch(self, batch):
      model_out, translator_out = self._predict(batch)
      y = batch.y.squeeze()
      (model_loss, model_metrics), (translator_loss, translator_metrics) = (
        self._compute_losses(model_out, translator_out, y)
      )
      return model_out, translator_out, y, model_loss, model_metrics, translator_loss, translator_metrics

    def edge_mask_explain(self, model_results: ModelResults, loader: DataLoader)->dict:
      class_results = self.edge_explainer.explain_subsamples(self.model, model_results, loader)
      return class_results

    def log_fidelity(self, loader: DataLoader, epoch: int, top_frac: float = 0.1, max_samples: int = 100) -> dict:
      """
      Run CoreRunner.fidelity_check on `loader.dataset` and log the result to
      W&B. This is a periodic diagnostic (each graph costs up to 3 forward
      passes) -- call it every N epochs from the training loop, not every one.
      """
      result = self.fidelity_check(loader.dataset, top_frac=top_frac, max_samples=max_samples)
      self.run.log({
          f"fidelity/{self.split_name}/baseline_acc": result["baseline_acc"],
          f"fidelity/{self.split_name}/topk_acc": result["topk_acc"],
          f"fidelity/{self.split_name}/complement_acc": result["complement_acc"],
      }, step=epoch)
      return result


    def visualize_all(self, epoch: int, model_results: ModelResults, translator_results: TranslatorResults, class_results: dict):
      self.visualizer.visualize(
            epoch,
            model_results.graph_embeddings,
            self.model.label_map,
            model_results.labels,
            model_results.predictions,
            model_results.pred_confidence,
            self.split_name,
            "model",
            model_results.edge_masks,
        )
      self.visualizer.visualize(
          epoch,
          translator_results.graph_embeddings,
          self.model.label_map,
          translator_results.labels,
          translator_results.predictions,
          translator_results.pred_confidence,
          self.split_name,
          "translator",
      )
      self.visualizer.log_subgraph_edge_masks(self.model.label_map, class_results, self.consistency_tracker, epoch, split=self.split_name)


    def evaluate(self, val_loader, epoch: int) -> tuple[ModelResults, TranslatorResults]:
        print("===== Evaluation =====")

        self.model.eval()
        num_batches = len(val_loader)
        model_results, translator_results = self.forward_pass(val_loader)


        model_correct_mask = (torch.tensor(model_results.predictions) == torch.tensor(model_results.labels))
        model_accuracy = model_correct_mask.float().mean().item()

        translator_correct_mask = (torch.tensor(translator_results.predictions) == torch.tensor(translator_results.labels))
        translator_accuracy = translator_correct_mask.float().mean().item()

        avg_metrics = {
            f"{self.split_name}/model_loss": model_results.loss,
            f"{self.split_name}/model_acc": model_accuracy,
            f"{self.split_name}/cosface_loss": model_results.cosface_loss,
            f"{self.split_name}/infonce_loss": model_results.infonce_loss,
            f"{self.split_name}/translator_loss": translator_results.loss,
            f"{self.split_name}/translator_acc": translator_accuracy,
        }
        self._log_metrics(epoch, avg_metrics)


        return model_results, translator_results
    