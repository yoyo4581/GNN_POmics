from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from wandb import Run
from io import BytesIO



def plot_confusion_with_confidence(y_true, y_pred, y_conf, class_names, run:Run, epoch: int, mat_type=Literal['train', 'val', 'test']):
    n = len(class_names)
    count_matrix = np.zeros((n, n), dtype=int)
    conf_matrix = np.zeros((n, n))
    
    for yt, yp, yc in zip(y_true, y_pred, y_conf):
        count_matrix[yt][yp] += 1
        conf_matrix[yt][yp] += yc

    # avg confidence per cell (avoid divide by zero)
    with np.errstate(invalid='ignore'):
        avg_conf = np.where(count_matrix > 0, conf_matrix / count_matrix, 0)

    # build annotation: "count\nconf%"
    annots = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annots[i, j] = f"{count_matrix[i,j]}\n{avg_conf[i,j]:.0%}"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(count_matrix, annot=annots, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (count / avg confidence)")

    run.log({f"{mat_type}/conf_mat": wandb.Image(fig)}, step=epoch)
    plt.close(fig)