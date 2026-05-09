import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
from collections import defaultdict
import torch
import numpy as np


def hex_to_rgba(hex_color, alpha):
    r, g, b = pc.hex_to_rgb(hex_color)
    return f'rgba({r},{g},{b},{alpha:.3f})'


def plot_class_explanation(
    class_idx: int,
    class_name: str,
    result: dict,
    G,
    graph,
    edge_index,
) -> go.Figure:
    avg_mask = result.avg_edge_mask
    accuracy = result.accuracy
    confidence = result.avg_confidence
    n_correct = result['n_correct']
    n_samples = result['n_samples']

    idx_gene = graph['idx_gene']
    pid_name = graph['pid_name']

    # ── Pathways & colors ─────────────────────────────────────
    all_pathways = set()
    for u, v, data in G.edges(data=True):
        for p in data.get('pathways', []):
            all_pathways.add(p)
    all_pathways = list(all_pathways)
    palette = px.colors.qualitative.Plotly
    pathway_color = {p: palette[i % len(palette)] for i, p in enumerate(all_pathways)}

    # ── Axes: sorted unique gene names ────────────────────────
    x = sorted({idx_gene[idx] for idx in edge_index[0].tolist()})  # from-genes (cols)
    y = sorted({idx_gene[idx] for idx in edge_index[1].tolist()})  # to-genes (rows)
    x_to_col = {gene: j for j, gene in enumerate(x)}
    y_to_row = {gene: i for i, gene in enumerate(y)}

    # ── Build z as [rows][cols] ────────────────────────────────
    n_rows, n_cols = len(y), len(x)
    z = [[None] * n_cols for _ in range(n_rows)]

    # hover_texts and edge_pathways are also [rows][cols]
    hover_texts = [[None] * n_cols for _ in range(n_rows)]
    edge_pathways = [[None] * n_cols for _ in range(n_rows)]

    edge_dict = {(u, v): i for i, (u, v) in enumerate(edge_index.T.tolist())}

    u_min, u_max = edge_index[0].min().item(), edge_index[0].max().item()
    v_min, v_max = edge_index[1].min().item(), edge_index[1].max().item()

    for v_idx in range(v_min, v_max + 1):
        to_gene = idx_gene[v_idx]
        if to_gene not in y_to_row:
            continue
        row = y_to_row[to_gene]

        for u_idx in range(u_min, u_max + 1):
            from_gene = idx_gene[u_idx]
            if from_gene not in x_to_col:
                continue
            col = x_to_col[from_gene]

            if (u_idx, v_idx) in edge_dict:
                data = G[from_gene][to_gene]
                pathways = data.get('pathways', [])
                pathway_names = [pid_name.get(p, p) for p in pathways]
                edge_type = data.get('type', 'UNKNOWN')

                edge_index_idx = edge_dict[(u_idx, v_idx)]
                raw = avg_mask[edge_index_idx].item()

                # hover_texts[row][col] = (
                #     f"<b>{from_gene} → {to_gene}</b><br>"
                #     f"Type: {edge_type}<br>"
                #     f"Mask weight: {raw:.3f}<br>"
                #     f"Pathways:<br>" + "<br>".join(f"  • {n}" for n in pathway_names)
                # )
                z[row][col] = raw
                edge_pathways[row][col] = pathways

    # ── Significant axis labels (top 25%) ─────────────────────
    flat = [v for row in z for v in row if v is not None]
    threshold = np.percentile(flat, 75)

    significant_x = sorted({col for row in range(n_rows) for col in range(n_cols)
                             if z[row][col] is not None and z[row][col] >= threshold})
    significant_y = sorted({row for row in range(n_rows) for col in range(n_cols)
                             if z[row][col] is not None and z[row][col] >= threshold})

    # ── Heatmap trace ──────────────────────────────────────────
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hoverongaps=False,
        name='all edges',
        showlegend=True,
    ))

    # ── Per-pathway scatter traces for legend toggling ─────────
    # Plotly supports multiple visible traces simultaneously via
    # 'legendonly' — clicking toggles each independently, so
    # multiple pathways can be shown at once ✓
    pathway_to_cells = defaultdict(list)
    for row in range(n_rows):
        for col in range(n_cols):
            pathways = edge_pathways[row][col]
            if pathways:
                for p in pathways:
                    pathway_to_cells[p].append((row, col))

    for pid in all_pathways:
        cells = pathway_to_cells[pid]
        name = pid_name.get(pid, pid)
        color_hex = pathway_color[pid]
        fig.add_trace(go.Scatter(
            x=[x[col] for _, col in cells],
            y=[y[row] for row, _ in cells],
            mode='markers',
            marker=dict(size=10, color=color_hex, opacity=0.9),
            # hovertext=[hover_texts[row][col] for row, col in cells],
            # hoverinfo='text',
            name=name,
            visible='legendonly',  # click to enable; multiple can be on at once
        ))

    # ── Layout (single call) ───────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"Class {class_idx}: {class_name} — "
                f"acc={accuracy:.3f} ({n_correct}/{n_samples}) | "
                f"median conf={confidence:.3f}"
            ),
            font=dict(size=14),
        ),
        width=900, height=900,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            tickvals=[x[j] for j in significant_x],
            ticktext=[x[j] for j in significant_x],
            tickangle=90,
            tickfont=dict(size=7),
            gridcolor='rgba(0,0,0,0)',
            zerolinecolor='rgba(0,0,0,0)',
        ),
        yaxis=dict(
            tickvals=[y[i] for i in significant_y],
            ticktext=[y[i] for i in significant_y],
            tickfont=dict(size=7),
            gridcolor='rgba(0,0,0,0)',
            zerolinecolor='rgba(0,0,0,0)',
        ),
        legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'),
        hovermode='closest',
    )

    return fig