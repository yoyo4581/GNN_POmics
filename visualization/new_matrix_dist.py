import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import functools

def build_matrix(class_results, tissue_descriptions, topk: int = 100, top_label: int = 25):

    # ── build DataFrame ───────────────────────────────────────────────────────
    top_indices = {}
    for cls, data in class_results.items():
        mask = data.avg_edge_mask
        if mask is None:
            continue
        _, idx = torch.topk(mask.edge_attention, k=min(topk, mask.edge_attention.numel()))
        top_indices[cls] = idx.numpy()

    all_indices = functools.reduce(np.union1d, list(top_indices.values()))

    # full weights for diagonal
    full_weight_df = pd.DataFrame(index=np.arange(len(data.avg_edge_mask.edge_attention)))
    for cls, data in class_results.items():
        if data.avg_edge_mask is None:
            continue
        full_weight_df[f'Class {cls}'] = data.avg_edge_mask.edge_attention.numpy()

    full_weight_df.index.name = 'gene'
    gene_ids = full_weight_df.index.values
    classes = list(full_weight_df.columns)
    n = len(classes)

    # ── PairGrid ──────────────────────────────────────────────────────────────
    fig = make_subplots(rows=n, cols=n, shared_xaxes=False, shared_yaxes=False)

    for row_idx, row_cls in enumerate(classes):
        for col_idx, col_cls in enumerate(classes):
            row = row_idx + 1
            col = col_idx + 1

            topk_mask = np.array([g in set(all_indices.tolist()) for g in gene_ids])
            x_vals = full_weight_df[col_cls].values[topk_mask]
            y_vals = full_weight_df[row_cls].values[topk_mask]
            gene_ids_filtered = gene_ids[topk_mask]

            if row_idx == col_idx:
                # ── diagonal: histogram ────────────────────────────────────
                fig.add_trace(go.Histogram(
                    x=x_vals,
                    nbinsx=80,
                    marker_color='steelblue',
                    name=row_cls,
                    showlegend=False,
                ), row=row, col=col)

            else:
                # ── off-diagonal: scatter ──────────────────────────────────
                row_key = int(row_cls.split()[-1])
                col_key = int(col_cls.split()[-1])
                row_set = set(top_indices[row_key].tolist())
                col_set = set(top_indices[col_key].tolist())

                in_row  = np.array([g in row_set for g in gene_ids_filtered])
                in_col  = np.array([g in col_set for g in gene_ids_filtered])
                in_both = in_row & in_col
                only_row = in_row & ~in_col
                only_col = in_col & ~in_row
                neither  = ~in_row & ~in_col

                # compute top deviant genes for tooltip highlighting
                deviation = np.abs(y_vals - x_vals)
                top_pos = set(np.argpartition(deviation, -top_label)[-top_label:])

                def make_scatter(mask, color, label):
                  indices = np.where(mask)[0]
                  normal_idx = [i for i in indices if i not in top_pos]
                  deviant_idx = [i for i in indices if i in top_pos]

                  traces = []

                  if normal_idx:
                      normal_idx = np.array(normal_idx)
                      hover = [
                          f"gene: {gene_ids[i]}<br>{col_cls}: {x_vals[i]:.4f}<br>{row_cls}: {y_vals[i]:.4f}"
                          for i in normal_idx
                      ]
                      traces.append(go.Scatter(
                          x=x_vals[normal_idx],
                          y=y_vals[normal_idx],
                          mode='markers',
                          marker=dict(color=color, size=4, opacity=0.6, symbol='circle'),
                          text=hover,
                          hovertemplate="%{text}<extra></extra>",
                          name=label,
                          showlegend=False,
                      ))

                  if deviant_idx:
                      deviant_idx = np.array(deviant_idx)
                      hover = [
                          f"gene: {gene_ids[i]}<br>{col_cls}: {x_vals[i]:.4f}<br>{row_cls}: {y_vals[i]:.4f}<br><b>⭐ top deviant</b>"
                          for i in deviant_idx
                      ]
                      traces.append(go.Scatter(
                          x=x_vals[deviant_idx],
                          y=y_vals[deviant_idx],
                          mode='markers',
                          marker=dict(color=color, size=8, opacity=0.9, symbol='star'),
                          text=hover,
                          hovertemplate="%{text}<extra></extra>",
                          name=f"{label} (top deviant)",
                          showlegend=False,
                      ))

                  return traces

                for mask, color, label in [
                    (only_row, 'steelblue', row_cls),
                    (only_col, 'tomato',    col_cls),
                    (in_both,  'seagreen',  'both'),
                    (neither,  'lightgray', 'neither'),
                ]:
                  if mask.any():
                      for trace in make_scatter(mask, color, label):
                          fig.add_trace(trace, row=row, col=col)

                # ── y = x reference line ───────────────────────────────────
                lo = min(x_vals.min(), y_vals.min())
                hi = max(x_vals.max(), y_vals.max())
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi],
                    mode='lines',
                    line=dict(color='#333333', dash='dash', width=0.8),
                    showlegend=False,
                ), row=row, col=col)

    for row_idx, row_cls in enumerate(classes):
      for col_idx, col_cls in enumerate(classes):
          row = row_idx + 1
          col = col_idx + 1

          row_key = int(row_cls.split()[-1])
          col_key = int(col_cls.split()[-1])

          if col_idx == 0:
              fig.update_yaxes(title_text=tissue_descriptions[row_key], title_font=dict(size=12), row=row, col=col)
          if row_idx == n - 1:
              fig.update_xaxes(title_text=tissue_descriptions[col_key], title_font=dict(size=12), row=row, col=col)

    fig.update_layout(
        title='Edge mask matrix · blue = row only | red = col only | green = both',
        height=300 * n,
        width=300 * n,
    )

    return fig