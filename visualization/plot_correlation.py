import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from itertools import combinations
from typing import Literal
from Model.data_model import ExplainerResult, EdgeMask





class ConsistencyTracker:
  def __init__(self, label_map: dict, split= Literal['train', 'val', 'test']):
    """
    Call every evaluation
    class_idx_to_label: e.g. {0: 'Cancer', ...}
      10 randomly sampled edge weight vectors per class at this epoch.
    """
    self.label_map = label_map
    self.split = split

    self.intra_coverage = {}
    self.intra_similarity = {}
    self.inter_coverage = {}
    self.inter_similarity = {}

    # Between-class divergence at a single epoch (not per-class: one aggregate
    # value per epoch, since it's a comparison *across* classes). Unlike
    # intra_*/inter_* above (which we want trending toward agreement/stability),
    # these should trend DOWN over training -- low overlap between different
    # classes' explanation subgraphs is what "discriminative" actually means.
    self.interclass_similarity = []
    self.interclass_coverage = []

    self.history = {}

    for idx in label_map:
      self._init_class(idx)

  def _init_class(self, idx: int):
    self.intra_coverage[idx] = []
    self.intra_similarity[idx] = []
    self.inter_coverage[idx] = []
    self.inter_similarity[idx] = []
    self.history[idx] = []

  def add_class(self, idx: int):
    """Called inside the trainer when adding class"""
    if idx not in self.intra_coverage:
      self._init_class(idx)


  def _mean_pairwise_cosine(self, vectors: list[EdgeMask]) -> float:
    """
    vectors: (n_samples_per_class, n_edges)
    Returns mean pairwise cosine similarity across all sample pairs.
    Not all samples are turned to graphs only a random subset of 10 samples.

    Will generate the mean cosine similarity between all attention vectors of a certain class.
    """
    # Filter out zero vectors to avoid cosine divide-by-zero
    valid_vectors = [
        v for v in vectors
        if np.linalg.norm(v.edge_attention.numpy()) > 0
    ]
    n = len(valid_vectors)
    if n < 2:
        return float('nan')

    corrs = [
        1 - cosine(valid_vectors[i].edge_attention.numpy(), valid_vectors[j].edge_attention.numpy())
        for i, j in combinations(range(n), 2)
    ]

    # Guard against all-nan slice (shouldn't happen now, but keeps it safe)
    return float(np.nanmean(corrs)) if corrs else float('nan')


  def _compute_intra_coverage(self, vectors: list[EdgeMask], percentiles = None):
    if percentiles is None:
      percentiles = (100 - np.geomspace(1, 50, 50)).astype(int)
      percentiles = np.unique(percentiles)
    
    attentions = [v.edge_attention.numpy() for v in vectors]
    n_edges = len(attentions[0])

    jaccard_scores = {}
    for p in percentiles:
      # Binarize each mask locally against its own percentile threshold
      binary_masks = [
          attn >= np.percentile(attn, p)
          for attn in attentions
      ]
      
      # Intersection: edge must be True in ALL masks
      intersection = np.ones(n_edges, dtype=bool)
      for mask in binary_masks:
        intersection &= mask
      
      # Union: edge must be True in AT LEAST ONE mask
      union = np.zeros(n_edges, dtype=bool)
      for mask in binary_masks:
        union |= mask
      
      union_size = union.sum()
      jaccard = intersection.sum() / union_size if union_size > 0 else 0.0
      jaccard_scores[p] = jaccard
    
    return jaccard_scores

  def _compute_inter_coverage(self, vector_a: EdgeMask, vector_b: EdgeMask, percentiles = None):
    if percentiles is None:
      percentiles = (100 - np.geomspace(1, 50, 50)).astype(int)
      percentiles = np.unique(percentiles)

    attention_a = vector_a.edge_attention.numpy()
    attention_b = vector_b.edge_attention.numpy()
    n_edges = len(attention_a)

    jaccard_scores = {}
    for p in percentiles:
      binary_mask_a = attention_a > np.percentile(attention_a, p)
      binary_mask_b = attention_b > np.percentile(attention_b, p)

      intersection = np.ones(n_edges, dtype=bool)
      intersection &= binary_mask_a
      intersection &= binary_mask_b

      union = np.zeros(n_edges, dtype=bool)
      union |= binary_mask_a
      union |= binary_mask_b

      union_size = union.sum()
      jaccard = intersection.sum() / union_size if union_size > 0 else 0.0
      jaccard_scores[p] = jaccard

    return jaccard_scores

  def _compute_interclass_divergence(self, avg_masks: list[tuple[int, EdgeMask]]):
    """
    Compare avg_edge_mask across DIFFERENT classes at the same epoch.

    Reuses the same pairwise helpers as the intra-class / temporal-inter-class
    checks above, just applied to a different pairing: every (class_a, class_b)
    combination at this epoch, instead of every sample-pair within one class or
    one class across two epochs. Low similarity / low coverage between classes
    is the signal that the model is attending to genuinely class-specific
    subgraphs rather than the same generic one for everybody.

    Args:
      avg_masks: list of (class_idx, avg_edge_mask) for classes that had at
        least one correct prediction this epoch.

    Returns:
      tuple: (mean_cosine_similarity: float, mean_jaccard_by_percentile: dict)
        or (nan, {}) if fewer than 2 classes are available to compare.
    """
    if len(avg_masks) < 2:
      return float('nan'), {}

    vectors = [mask for _, mask in avg_masks]
    mean_similarity = self._mean_pairwise_cosine(vectors)

    pairwise_coverage = [
      self._compute_inter_coverage(mask_a, mask_b)
      for (_, mask_a), (_, mask_b) in combinations(avg_masks, 2)
    ]
    percentiles = pairwise_coverage[0].keys()
    mean_coverage = {
      p: float(np.mean([scores[p] for scores in pairwise_coverage]))
      for p in percentiles
    }

    return mean_similarity, mean_coverage



  def update(self, epoch: int, class_edge_masks: dict[int, ExplainerResult]):
    """
    Call this every epoch.

    class_edge_masks: {class_idx: ExplainerResult (class-wise edge masks and avg edge_mask)}
      Edge weight vectors for each sample of that class at this epoch.

      This method will:
      1. Calculate coverage:
        - Jaccard Score Intra (within samples of a class)
        - Jaccard Score Inter (between epochs average of a certain class)
      
      2. Calculate within-class similarity (cosine similarity).

    """
    from collections import defaultdict

    
    for class_idx, class_results in class_edge_masks.items():
      if class_results.edge_masks is None:
        continue #skip classes with no correct predictions
        
      score = self._mean_pairwise_cosine(class_results.edge_masks)
      self.intra_similarity[class_idx].append((epoch, score))

      intra_p_jaccard_scores = self._compute_intra_coverage(class_results.edge_masks)
      self.intra_coverage[class_idx].append((epoch, intra_p_jaccard_scores))

      if self.history[class_idx]:
        score = self._mean_pairwise_cosine([class_results.avg_edge_mask,
                                            self.history[class_idx][-1][1]])
        self.inter_similarity[class_idx].append((epoch, score))

        # compare the current avg edge mask with the previous.
        inter_p_jaccard_scores = self._compute_inter_coverage(class_results.avg_edge_mask, 
                                self.history[class_idx][-1][1])
        self.inter_coverage[class_idx].append((epoch, inter_p_jaccard_scores))

      self.history[class_idx].append((epoch, class_results.avg_edge_mask))

    avg_masks = [
      (class_idx, class_results.avg_edge_mask)
      for class_idx, class_results in class_edge_masks.items()
      if class_results.avg_edge_mask is not None
    ]
    similarity, coverage = self._compute_interclass_divergence(avg_masks)
    if coverage:
      self.interclass_similarity.append((epoch, similarity))
      self.interclass_coverage.append((epoch, coverage))

    self._log_to_wandb(epoch)

  def log_coverage(self, all_coverage, title_str: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    
    for class_idx, history_coverage in all_coverage.items():
      if not history_coverage:
        continue
      class_name = self.label_map[class_idx]
      latest_coverage = history_coverage[-1][1]
      
      percentiles = [p for p in latest_coverage]
      values = [v for v in latest_coverage.values()]
      
      fig.add_trace(go.Scatter(
          x=percentiles,
          y=values,
          mode="lines",
          name=class_name,  # shows in legend, color-coded automatically
      ))
    
    fig.update_layout(
        title=f"{title_str} percentiles ({self.split})",
        xaxis_title="Percentile",
        yaxis_title=f"{title_str}",
        legend_title="Class",
    )
    
    return fig

  def _log_to_wandb(self, epoch: int):
    """ Logs a native W&B line plot, updated incrementally each epoch"""
    import wandb

    log_dict = {}
    for class_idx, label in self.label_map.items():
      if self.intra_similarity[class_idx]:
        latest_score = self.intra_similarity[class_idx][-1][1]
        log_dict[f"intra_cos_similarity/{self.split}/{label}"] = latest_score
      
      if self.inter_similarity[class_idx]:
        latest_score = self.inter_similarity[class_idx][-1][1]
        log_dict[f"inter_cos_similarity/{self.split}/{label}"] = latest_score

    if any(self.inter_coverage[idx] for idx in self.label_map):
      fig = self.log_coverage(self.inter_coverage, title_str = 'Inter Coverage')
      log_dict[f"inter_coverage/{self.split}"] = wandb.Plotly(fig)
    
    if any(self.intra_coverage[idx] for idx in self.label_map):
      fig = self.log_coverage(self.intra_coverage, title_str = 'Intra Coverage')
      log_dict[f"intra_coverage/{self.split}"] = wandb.Plotly(fig)

    # Between-class divergence: unlike everything above, LOWER is the desired
    # direction here -- it means different classes' explanation subgraphs are
    # pulling apart rather than converging on the same generic pattern.
    if self.interclass_similarity:
      log_dict[f"interclass_cos_similarity/{self.split}"] = self.interclass_similarity[-1][1]

    if self.interclass_coverage:
      latest_curve = self.interclass_coverage[-1][1]
      top10_percentile = min(latest_curve, key=lambda p: abs(p - 90))
      log_dict[f"interclass_jaccard_top10pct/{self.split}"] = latest_curve[top10_percentile]

    if log_dict:
      wandb.log(log_dict, step=epoch)
