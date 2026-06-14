import torch
import importlib
import numpy as np
from collections import defaultdict
from wandb import Run
from torch_geometric.loader import DataLoader
import Model.core_utils

importlib.reload(Model.core_utils)

from Model.data_model import ModelResults, EdgeMask, TranslatorResults
from Model.core_utils import EdgeMaskExplainer, CoreRunner



from visualization.umap import UMAPTransform
from visualization.confusion_matrix import plot_confusion_with_confidence
from visualization.attention_percentile import log_attention_percentiles
from visualization.visualizer import Visualizer
from visualization.plot_correlation import ConsistencyTracker


from typing import Literal

class Trainer(CoreRunner):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        run: Run,
        visualizer: Visualizer,
        consistency_tracker: ConsistencyTracker,
    ):
        super().__init__(model, optimizer, device)
        self.consistency_tracker = consistency_tracker
        self.run = run
        self.visualizer = visualizer
        
        self.split_name = 'train'
        self.edge_explainer = EdgeMaskExplainer()

    # ------------------------------------------------------------------ #
    # Forward / backward                                                  #
    # ------------------------------------------------------------------ #

    def _update_memory(
        self, results: ModelResults
    ):
        self.model.proto_mem.update(results.graph_embeddings, results.labels)

    def sync_classes_from(self, label_json_path: str = None, label_map: dict = None):
      import json
      
      if not label_json_path and not label_map:
        print("Must provide JSON or label_map dictionary to sync classes from")
        return

      if not label_map and label_json_path:
        with open(label_json_path) as f:
            label_map = json.load(f)
            label_map = {int(k):v for k,v in label_map.items()}


      self.model.warmup_complete = False
      for label, description in label_map.items():
          if label not in self.model.label_map:
              self.model.add_class(label, description)
              self.consistency_tracker.add_class(label, description)
              print(f'Added {label}: {description} class to model')

    # ------------------------------------------------------------------ #
    # Post-epoch visualisation / logging                                  #
    # ------------------------------------------------------------------ #


    def _log_metrics(self, epoch: int, avg_metrics: dict) -> None:
        key = f"{self.split_name}/model_loss"
        print(f"  Epoch {epoch}  model_loss={avg_metrics[key]:.4f}")
        self.run.log(avg_metrics, step=epoch)

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
          self.split_name,
          "translator",
      )
      self.visualizer.log_subgraph_edge_masks(self.model.label_map, class_results, self.consistency_tracker, epoch, split=self.split_name)


    # ------------------------------------------------------------------ #
    # Main training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(self, train_loader, epoch: int) -> ModelResults:

        if epoch %20 ==0:
            self.model.warmup_complete = True

        if epoch % 10 == 0:
            # Update prototypes.
            self.model.proto_mem.agglomerative_cluster()

        self.model.train()
        num_batches = len(train_loader)
        model_results, translator_results = self.forward_pass(train_loader)

        # Update memory
        self._update_memory(model_results)

      
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

