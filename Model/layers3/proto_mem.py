"""Prototype_Memory: per-class embedding replay buffer, clustering, and InfoNCE loss.

Shape contract:
    memory_queue[c]: deque of up to `capacity` Tensor[D] embeddings for class c
    queue_cache[c]:  Tensor[N_c, D] — memory_queue[c] stacked, refreshed each clustering pass
    prototypes:      list of PrototypeEmbedding, each holding a Tensor[D] centroid + Tensor[D] embedding
    InfoNCELoss:
        input  graph_embeddings: [B, D]
        input  labels:           [B]
        output loss:             []  (scalar, mean over the batch)
"""

from collections import deque
from Model.data_model import PrototypeEmbedding
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np


class Prototype_Memory:
    """Per-class replay buffer that clusters embeddings into prototypes for contrastive training.

    Each class keeps a fixed-capacity FIFO queue of its most recent graph
    embeddings (`memory_queue`). Periodically, `agglomerative_cluster()`
    groups each class's queued embeddings into cosine-similarity clusters
    and keeps one representative "prototype" per cluster — the member
    closest to its cluster's centroid. `InfoNCELoss` then treats each
    incoming embedding as an anchor, its class's closest prototype as the
    positive, and every other class's queued embeddings as negatives.

    Shapes:
        embeddings pushed via `update`: [D]  (one per sample, per call)
        memory_queue[c]: up to `capacity` Tensor[D] entries
        queue_cache[c]:  Tensor[N_c, D]
        prototypes[i].centroid / .embedding: Tensor[D]

    Args:
        class_num: Number of classes; also the number of memory queues created.
        device: Device that `queue_cache` tensors are moved to.
        capacity: Max number of embeddings retained per class.
        temperature: Softmax temperature used in `InfoNCELoss`.
    """

    def __init__(self, class_num: int, device: torch.device, capacity: int = 100, temperature: float = 0.3):
        self.capacity = capacity
        self.class_num = class_num
        self.memory_queue = {x: deque(maxlen=capacity) for x in range(class_num)}
        self.queue_cache = {}
        self.history_clusters = {}
        self.prototypes = []
        self.device = device
        self.clustering_count = 0
        self.temperature = temperature

    @property
    def total_embeddings(self):
        """int: Total embedding capacity across all classes (capacity * class_num)."""
        return self.capacity * len(self.memory_queue)

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Push a batch of graph embeddings into their class's memory queue.

        Only the most recent `capacity` embeddings per class are kept —
        older entries are evicted first (FIFO), since the most recently
        seen embeddings are considered the most representative of the
        model's current behavior.

        Args:
            embeddings: Tensor[B, D] — graph embeddings for this batch.
            labels: Tensor[B] — ground-truth class index per embedding.
        """
        for i, label in enumerate(labels):
            self.memory_queue[label].append(embeddings[i])

    def is_clustering_stable(self, threshold=0.9):
        """Check whether the two most recent clustering passes agree (Adjusted Rand Index).

        Args:
            threshold: Minimum mean ARI across classes to call clustering stable.

        Returns:
            bool: True once at least two clustering passes exist and their
                mean per-class ARI exceeds `threshold`.
        """
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
        """Recompute per-class prototypes by clustering each class's queued embeddings.

        For every class with at least one queued embedding: run cosine-linkage
        agglomerative clustering (distance_threshold=0.05, so the number of
        clusters is discovered rather than fixed), then keep one
        `PrototypeEmbedding` per cluster — its centroid, plus its member
        closest to that centroid by cosine similarity.

        Side effects:
            self.prototypes: rebuilt from scratch (list[PrototypeEmbedding]).
            self.history_clusters: appended with this pass's per-class cluster labels.
            self.queue_cache: refreshed — Tensor[N_c, D] per class, moved to self.device.
        """
        self.prototypes = []
        self.clustering_count += 1

        for class_label, queue in self.memory_queue.items():
            if len(queue) == 0:
                continue

            embeddings_tensor = torch.stack(list(queue))  # [N_c, D]
            clustering = AgglomerativeClustering(
                metric='cosine',
                linkage='average',
                distance_threshold=0.05,
                n_clusters=None,
                compute_full_tree=True
            )

            cluster_labels = clustering.fit_predict(embeddings_tensor.numpy())  # [N_c]
            self.history_clusters[self.clustering_count] = {class_label: cluster_labels}

            for cluster_id in range(cluster_labels.max() + 1):
                mask = cluster_labels == cluster_id
                centroid = embeddings_tensor[mask].mean(dim=0)  # [D]

                # Cosine similarity (not raw dot product) so the closest member is
                # picked by direction, not by whichever embedding has the largest norm.
                similarities = F.normalize(embeddings_tensor[mask], dim=-1) @ F.normalize(centroid, dim=0)  # [N_cluster]
                closest_idx = similarities.argmax()

                self.prototypes.append(
                    PrototypeEmbedding(
                        class_label=class_label,
                        centroid=centroid,                              # [D]
                        embedding=embeddings_tensor[mask][closest_idx],  # [D]
                    )
                )

        self.is_clustering_stable()
        self.queue_cache = {
            lab: torch.stack(list(q)).to(device=self.device)  # [N_lab, D]
            for lab, q in self.memory_queue.items() if len(q) > 0
        }

    def InfoNCELoss(self, graph_embeddings: torch.Tensor, labels: torch.Tensor):
        """Prototype-contrastive InfoNCE loss over a batch of graph embeddings.

        For each embedding in the batch: find its class's closest prototype
        (by cosine similarity to prototype centroids) as the positive, treat
        every other class's queued embeddings as negatives, and compute a
        standard InfoNCE loss. All similarities are cosine (every vector is
        L2-normalized before comparison), so embedding magnitude never
        influences the loss — only direction does.

        Assumes `agglomerative_cluster()` has already populated `self.prototypes`
        and `self.queue_cache` for every class present in `labels` — this holds
        by construction once warmup epochs have run (see the training loop).

        Args:
            graph_embeddings: Tensor[B, D] — anchor embeddings for this batch.
            labels: Tensor[B] — ground-truth class index per embedding.

        Returns:
            Tensor[]: scalar loss, averaged over the batch.
        """

        def L_InfoNCE(z_i, z_p, z_n):
            """One anchor's InfoNCE loss against its positive and all negatives.

            Args:
                z_i: Tensor[D] — L2-normalized anchor embedding.
                z_p: Tensor[D] — L2-normalized positive (closest same-class prototype).
                z_n: Tensor[N_neg, D] — L2-normalized negative embeddings (other classes).

            Returns:
                Tensor[]: scalar loss for this one anchor.
            """
            pos_sim = (z_i @ z_p) / self.temperature   # []
            neg_sims = (z_n @ z_i) / self.temperature   # [N_neg]

            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + N_neg]
            loss = -(pos_sim - torch.logsumexp(all_sims, dim=0))
            return loss

        total_loss = 0.0
        for g_emb, label in zip(graph_embeddings, labels):
            label = label.item()
            z_i = F.normalize(g_emb, dim=-1)  # [D]
            label_prototypes = [prot for prot in self.prototypes if prot.class_label == label]

            centroids = F.normalize(
                torch.stack([p.centroid for p in label_prototypes]).to(
                    dtype=g_emb.dtype, device=g_emb.device
                ),
                dim=-1,
            )  # [N_proto, D]
            closest_prototype = label_prototypes[(centroids @ z_i).argmax().item()]
            z_p = F.normalize(
                closest_prototype.embedding.to(dtype=g_emb.dtype, device=g_emb.device), dim=-1
            )  # [D]

            z_n = F.normalize(torch.cat([
                v for lab, v in self.queue_cache.items() if lab != label
            ]), dim=-1)  # [N_neg, D]

            total_loss += L_InfoNCE(z_i, z_p, z_n)

        return total_loss / len(labels)
