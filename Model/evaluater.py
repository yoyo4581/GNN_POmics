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
        self.consistency_tracker = ConsistencyTracker(self.model.label_map, split='val')
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
          'val',
          "translator",
      )
      self.visualizer.log_subgraph_edge_masks(self.model.label_map, class_results, self.consistency_tracker, epoch, split=self.split_name)


    def evaluate(self, val_loader, epoch: int) -> ModelResults:
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
            f"{self.split_name}/translator_loss": translator_results.loss,
            f"{self.split_name}/translator_acc": translator_accuracy,
        }
        self._log_metrics(epoch, avg_metrics)


        return model_results, translator_results
    