from collections import deque
from Model.data_model import PrototypeEmbedding
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np

class Prototype_Memory:
  def __init__(self, class_num: int, device: torch.device, capacity: int = 100):
    self.capacity = capacity
    self.class_num = class_num
    self.memory_queue = {x: deque(maxlen=capacity) for x in range(class_num)}
    self.queue_cache = {}
    self.history_clusters = {}
    self.device = device
    self.clustering_count = 0
    self.temperature = 0.3

  @property
  def total_embeddings(self):
    return self.capacity * len(self.memory_queue)

  def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
    """
    Takes in data incoming from batch and queues it in the memory_queue.
    Only the last few batches (most updated) graph embeddings are reliable.\
    """
    for i, label in enumerate(labels):
      self.memory_queue[label].append(embeddings[i])

  def is_clustering_stable(self, threshold=0.9):
    if len(self.history_clusters) < 2:
      return False

    curr_cluster_labels = self.history_clusters[self.clustering_count]
    prev_cluster_labels = self.history_clusters[self.clustering_count - 1]
    ari_scores = [
      adjusted_rand_score(prev_cluster_labels[c], curr_cluster_labels[c])
      for c in curr_cluster_labels
      if c in prev_cluster_labels
    ]
    print(np.mean(ari_scores))
    return np.mean(ari_scores) > threshold

  def agglomerative_cluster(self):
    self.prototypes = []
    self.clustering_count += 1
    
    for class_label, queue in self.memory_queue.items():
        if len(queue) == 0:
            continue
            
        embeddings_tensor = torch.stack(list(queue))
        clustering = AgglomerativeClustering(
            metric='cosine',
            linkage='average',
            distance_threshold=0.05,
            n_clusters=None,
            compute_full_tree=True
        )

        cluster_labels = clustering.fit_predict(embeddings_tensor.numpy())
        self.history_clusters[self.clustering_count] = {class_label: cluster_labels}
        
        for cluster_id in range(cluster_labels.max() + 1):
            mask = cluster_labels == cluster_id
            centroid = embeddings_tensor[mask].mean(dim=0)

            similarities = embeddings_tensor[mask] @ centroid
            closest_idx = similarities.argmax()

            self.prototypes.append(
                PrototypeEmbedding(
                    class_label=class_label,
                    centroid = centroid,
                    embedding=embeddings_tensor[mask][closest_idx]
                )
            )
    stable_score = self.is_clustering_stable()
    self.queue_cache = {
      lab: torch.stack(list(q)).to(device=self.device)
      for lab, q in self.memory_queue.items() if len(q) > 0
    }



  def InfoNCELoss(self, graph_embeddings: torch.Tensor, labels: torch.Tensor):
    """
    Takes in batch wise graph embeddings and their accompanying ground-truth labels.
    Iterate through graph_embeddings fetch the correct and closest prototype using its centroid.
    Use that prototype's embedding (closest neighbor embedding) as positive.
    Use the graph embedding in query as the anchor.
    Compute loss as:
    
    """

    def L_InfoNCE(z_i, z_p, z_n):
      """
      For a given anchor and positive example identify the loss from all other non cluster negative examples.
      """
      pos_sim = (z_i @ z_p) / self.temperature
      neg_sims = (z_n @ z_i) / self.temperature
      
      all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
      loss = -(pos_sim - torch.logsumexp(all_sims, dim=0))
      return loss

    total_loss = 0.0
    for g_emb, label in zip(graph_embeddings, labels):
      label = label.item()
      label_prototypes = [prot for prot in self.prototypes if prot.class_label == label]

      centroids = torch.stack([p.centroid for p in label_prototypes]).to(
        dtype=g_emb.dtype, device=g_emb.device
      )
      closest_prototype = label_prototypes[(centroids @ g_emb).argmax().item()]
      z_p = closest_prototype.embedding.to(dtype=g_emb.dtype, device=g_emb.device)

      z_n = torch.cat([
        v for lab, v in self.queue_cache.items() if lab != label
      ])

      loss = L_InfoNCE(g_emb, z_p, z_n)
      total_loss += loss
    
    return total_loss / len(labels)