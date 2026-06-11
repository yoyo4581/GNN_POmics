import os
import matplotlib.pyplot as plt
import umap
import numpy as np
from matplotlib import cm
import wandb
import io
from PIL import Image

vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"name": "wandb"},
    "mark": {"type": "point", "filled": True},
    "encoding": {
        "x": {"field": "umap_x", "type": "quantitative"},
        "y": {"field": "umap_y", "type": "quantitative"},
        "color": {"field": "class", "type": "nominal"},
        "shape": {
            "field": "is_anchor",
            "type": "nominal",
            "scale": {
                "domain": [False, True],
                "range": ["circle", "star"]
            }
        }
    }
}

class UMAPTransform:
  def __init__(self, model):
    self.label_embeddings = model.head.label_embeddings.detach().cpu().numpy()
    self.label_names = model.head.label_names
    self.num_classes = len(self.label_names)
    self.reducer = None

  def is_fitted(self):
    return self.reducer is not None

  def fit(self, embeddings: np.ndarray, labels: list[int])-> None:
    self.reducer = umap.UMAP(
      n_neighbors=30, 
      min_dist=0.1, 
      metric="cosine",
    ).fit(embeddings)
    
  def transform(self, embeddings: np.ndarray) -> np.ndarray:
    if not self.is_fitted:
      raise RuntimeError("UMAPTransform must be fitted before calling transform.")
    return self.reducer.transform(embeddings)


  def plot(self, 
    embeddings: np.ndarray,
    labels: list[int], 
    run: wandb.Run,
    epoch: int, 
    split: str = "train"
  ) -> None:
    labels = np.array(labels)
    proj_2d = self.transform(embeddings)
    anchor_2d = self.reducer.transform(self.label_embeddings)

    # --- W&B Table (interactive pan/zoom scatter in dashboard) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for cls_idx, name in enumerate(self.label_names):
        mask = labels == cls_idx
        sc = ax.scatter(proj_2d[mask, 0], proj_2d[mask, 1], label=name, s=10)
        color = sc.get_facecolor()[0]  # grab the color matplotlib assigned
        ax.scatter(anchor_2d[cls_idx, 0], anchor_2d[cls_idx, 1], 
                  marker='*', s=200, color=color, edgecolors='black', linewidths=0.5)

    ax.legend()
    run.log({f"{split}/umap": wandb.Image(fig)}, step=epoch)
    plt.close(fig)



