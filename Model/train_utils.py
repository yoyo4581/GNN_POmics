import torch
from wandb import Run
from Model.data_model import ModelResults, tissue_descriptions, EdgeMask
from visualization.umap import UMAPTransform
from visualization.confusion_matrix import plot_confusion_with_confidence
from visualization.attention_percentile import log_attention_percentiles
import numpy as np
from collections import defaultdict
from layers3.proto_mem import Prototype_Memory

class Trainer:
  def __init__(self,
              model: torch.nn.Module,
              model_optimizer: torch.optim.Optimizer,
              translator_optimizer: torch.optim.Optimizer,
              device: torch.device,
              run: Run,
              model_umap: UMAPTransform,
              translator_umap: UMAPTransform):
    
    self.model = model
    self.model_optimizer = model_optimizer
    self.translator_optimizer = translator_optimizer
    self.device = device
    self.run = run
    self.model_umap = model_umap
    self.translator_umap = translator_umap
    

  def train(self, train_loader, epoch):

    batch_size = train_loader.batch_size
    
    # Heuristic to determine the batch_num to start updating the memory queue
    batch_num_retain = self.model.proto_mem.total_embeddings // batch_size
    num_batches = len(train_loader)

    node_scores_all, attention_weights_all, edge_indices = [], [], []
    predictions, pred_confidence, labels = [], [], []
    model_projections, translator_projections = [], []
    total_translator_loss = total_translator_acc = correct_translator_conf = incorrect_translator_conf = all_translator_conf = 0.0
    total_model_loss = total_model_acc = correct_model_conf = incorrect_model_conf = all_model_conf = 0.0

    if epoch > 20:
      self.model.warmup_complete = True

    self.model.train()
    for batch_num, batch in enumerate(train_loader):
      batch = batch.to(self.device)
  
      self.model_optimizer.zero_grad()
      self.translator_optimizer.zero_grad()

      model_out, translator_out = self.model(batch.x.unsqueeze(1), batch.edge_index, batch.batch)
      if batch_num > batch_num_retain:
        self.model.proto_mem.update(model_out["graph_embedding"], batch.y.squeeze())

      translator_loss, translator_metrics = self.model.head.loss(translator_out, batch.y.squeeze())
      model_loss, model_metrics = self.model.loss(model_out, batch.y.squeeze())

      translator_loss.backward()
      model_loss.backward()
      self.translator_optimizer.zero_grad()
      self.model_optimizer.zero_grad()

      total_model_loss += model_loss.item()
      total_model_acc += model_metrics.get("accuracy")
      correct_model_conf += model_metrics.get("conf_avgs").get("correct")
      incorrect_model_conf += model_metrics.get("conf_avgs").get("incorrect")
      all_model_conf += model_metrics.get("all_confidences")

      total_translator_loss += translator_loss.item()
      total_translator_acc += translator_metrics.get("accuracy")
      correct_translator_conf += translator_metrics.get("conf_avgs").get("correct")
      incorrect_translator_conf += translator_metrics.get("conf_avgs").get("incorrect")
      all_translator_conf += translator_metrics.get("all_confidences")

      model_projections.append(model_out["graph_embedding"])
      translator_projections.append(translator_out["graph_embedding"])
      labels += batch.y.squeeze().cpu().tolist()

      if "attention" in model_out and "node_scores" in model_out:
        for (edge_idx, attn), node_scores in zip(model_out["attention"], model_out["node_scores"]):
          edge_indices.append(edge_idx.detach.cpu())
          attention_weights_all.append(attn.detach().cpu())
          node_scores_all.append(node_scores.detach().cpu())

      predictions += [p.cpu().item() for p in model_metrics.get("pred_idx")]
      pred_confidence += [c.cpu().item() for c in model_metrics.get("all_confidences")]

    model_embeddings = np.vstack(model_projections)
    if not self.umap.is_fitted():
      self.umap.fit(model_embeddings, labels)





def train(model: torch.nn.Module,
          loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epoch: int,
          run:Run,
          umap: UMAPTransform,
        )->ModelResults:


  model.train()
  num_batches = len(loader)

  node_scores_all, attention_weights_all, edge_indices = [], [], []
  predictions, pred_confidence, labels, projections = [], [], [], []
  total_loss = total_acc = total_correct_conf = total_incorrect_conf = all_conf = 0.0

  for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()

    out, proj = model(batch.x.unsqueeze(1), batch.edge_index, batch.batch)
    loss, acc, conf_avgs, pred_idx, all_confidence = model.loss(out, batch.y.squeeze())

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_acc += acc
    total_correct_conf += conf_avgs['correct']
    total_incorrect_conf += conf_avgs['incorrect']

    projections.append(proj)
    labels += batch.y.squeeze().cpu().tolist()

    if "attention" in out and "node_scores" in out:
      for (edge_idx, attn), node_scores in zip(out["attention"], out["node_scores"]):
        edge_indices.append(edge_idx.detach().cpu())
        attention_weights_all.append(attn.detach().cpu())
        node_scores_all.append(node_scores.detach().cpu())

    predictions += [p.cpu().item() for p in pred_idx]
    pred_confidence += [c.cpu().item() for c in all_confidence]

    
  embeddings=np.vstack(projections)

  if not umap.is_fitted():
    umap.fit(embeddings, labels)

  if epoch % 5 == 0:
    umap.plot(embeddings, labels, run, epoch, split="train")
    plot_confusion_with_confidence(
      y_true=labels,
      y_pred=predictions,  
      y_conf=pred_confidence, 
      class_names=tissue_descriptions.values(),
      run= run,
      epoch= epoch, 
      mat_type='train'
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
      split="train/correct")

    log_attention_percentiles(
      incorrect_class_attention,
      tissue_descriptions, 
      run, 
      epoch, 
      split="train/incorrect")      
    

  
  avg_loss = total_loss / num_batches
  avg_acc = total_acc / num_batches
  avg_correct_conf = total_correct_conf / num_batches
  avg_incorrect_conf = total_incorrect_conf / num_batches

  print(f"  Epoch {epoch}  loss={avg_loss:.4f}")

  run.log({"train/loss": avg_loss,
          "train/acc": avg_acc,
          "train/correct_conf": avg_correct_conf,
          "train/incorrect_conf": avg_incorrect_conf}, step=epoch)

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
      edge_masks= edge_masks,
      graph_embeddings = torch.tensor(embeddings),
      labels=labels,
      dataset_index = list(loader.sampler),
      loss = avg_loss,
      dataset='train'
    )

