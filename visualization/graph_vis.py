import networkx as nx
import torch
import matplotlib.pyplot as plt
import math
from typing import Literal


def plot_network(edge_mask, graph):
  fig, ax = plt.subplots(figsize=(8,6))
  
  if edge_mask is None:
    ax.axis('off')
    return

  edge_index = graph.graph['edge_index']
  idx_gene = graph.graph['idx_gene']
  mask = edge_mask
  weights, indices = torch.topk(mask, 50)
  subbed_edges = edge_index[:, indices]

  sorted_idx, _ = torch.sort(subbed_edges.flatten())
  sorted_idx = sorted_idx.tolist()
  subgraph_names = [idx_gene[idx] for idx in sorted_idx]
  subgraph = graph.subgraph(subgraph_names)

  pos = nx.spring_layout(subgraph, k=0.15, iterations=20)
  subgraph_edges = list(subgraph.edges())
  for i, (u, v) in enumerate(subbed_edges.T.tolist()):
    w = weights[i]
    gene_u = idx_gene[u]
    gene_v = idx_gene[v]
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edgelist = [(gene_u, gene_v)],
        ax = ax,
        alpha = w,
        edge_color='black'
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_size=20,
        node_color='steelblue',
        alpha=0.85
    )
    nx.draw_networkx_labels(
        subgraph,
        pos,
        ax=ax,
        font_size=5,
        font_color='black'
    )
    ax.axis('off')
    plt.close(fig)

  return fig

def log_network_figure(figure, tissue_name, epoch, run, split: Literal['train', 'val', 'test']):
  import wandb

  #W&B Save
  run.log({f"eval/network/{tissue_name}.png": wandb.Image(figure)}, step=epoch)
  



  #