from torch_geometric.data import Data
from typing import Optional, Union
import numpy as np
import plotly
from plotly.graph_objects import Figure

def visualize_prototype_graph_interactive(
    sample:     Data,
    class_id:   int,
    num_genes:  int,
    edge_mask=  None,
    save_path:  Optional[str] = None,
    thresholds: Optional[list] = None,
) -> Figure:
    """
    Interactive Plotly visualization of the prototype graph with a slider
    to adjust the edge-weight threshold (top-X% of edges retained).

    TensorBoard compatibility
    ─────────────────────────
    Option A — HTML embed (recommended):
        writer.add_text("graph/explanation", iframe_html, global_step)
        where iframe_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    Option B — Static snapshot per threshold step:
        import plotly.io as pio
        img = pio.to_image(fig, format="png")
        writer.add_image("graph/explanation", img_tensor, global_step)

    Option C — Save standalone HTML and open via TensorBoard projector tab.

    Args:
        sample:      a single Data object
        class_id:    integer class label
        num_genes:   number of nodes
        edge_mask:   per-edge importance scores (optional)
        save_path:   if given, writes a self-contained HTML file here
        thresholds:  percentile cutoffs to expose on the slider,
                     e.g. [0, 10, 25, 50, 75, 90].
                     Defaults to [0, 10, 25, 50, 75, 90].

    Returns:
        plotly Figure (also saved to save_path if provided)
    """
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np

    if thresholds is None:
        thresholds = [0, 10, 25, 50, 75, 90]

    edge_index = sample.edge_index.numpy()
    expr       = sample.x.mean(dim=1).numpy()          # [N]

    if edge_mask is not None:
        weights = edge_mask.detach().cpu().numpy().astype(float)
    else:
        weights = np.ones(edge_index.shape[1], dtype=float)

    hub_genes = {0, 1, 2}
    hi_start  = (class_id * 5) % num_genes
    lo_start  = (class_id * 5 + 5) % num_genes
    hi_block  = set(range(hi_start, min(hi_start + 5, num_genes)))
    lo_block  = set(range(lo_start, min(lo_start + 5, num_genes)))

    # ── Fixed circular layout (all nodes) ────────────────────────
    G_full = nx.DiGraph()
    G_full.add_nodes_from(range(num_genes))
    pos    = nx.circular_layout(G_full)   # {node: (x, y)}

    node_x = [pos[i][0] for i in range(num_genes)]
    node_y = [pos[i][1] for i in range(num_genes)]

    # Node colours: RdYlGn mapped to expression
    expr_norm = (expr - expr.min()) / max(expr.max() - expr.min(), 1e-8)

    def expr_to_rgb(v):
        # Manual RdYlGn: red(0) → yellow(0.5) → green(1)
        if v < 0.5:
            t = v * 2
            r, g, b = int(255), int(255 * t), 0
        else:
            t = (v - 0.5) * 2
            r, g, b = int(255 * (1 - t)), int(255), 0
        return f"rgb({r},{g},{b})"

    node_colors  = [expr_to_rgb(expr_norm[i]) for i in range(num_genes)]
    node_symbols = ["diamond" if i in hub_genes else "circle"
                    for i in range(num_genes)]
    node_sizes   = [14 if i in hub_genes else 10 for i in range(num_genes)]

    def node_label(i):
        tags = []
        if i in hub_genes: tags.append("hub")
        if i in hi_block:  tags.append("↑ hi")
        if i in lo_block:  tags.append("↓ lo")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        return f"Gene {i}{tag_str}<br>expr={expr[i]:.2f}"

    node_hover = [node_label(i) for i in range(num_genes)]

    # ── Node trace (shared across all slider steps) ───────────────
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(color="#1a1a2e", width=1.5),
        ),
        text=[str(i) if i in hub_genes else "" for i in range(num_genes)],
        textposition="middle center",
        textfont=dict(size=7, color="white", family="IBM Plex Mono"),
        hovertext=node_hover,
        hoverinfo="text",
        name="Genes",
    )

    # ── Bar chart traces (gene expression, one per category) ──────
    def bar_color(i):
        if i in hi_block:  return "#e05c5c"
        if i in lo_block:  return "#5b9bd5"
        if i in hub_genes: return "#c9a84c"
        return "#8ab4cc"

    bar_trace = go.Bar(
        x=list(range(num_genes)),
        y=expr.tolist(),
        marker_color=[bar_color(i) for i in range(num_genes)],
        hovertemplate="Gene %{x}<br>expr=%{y:.2f}<extra></extra>",
        name="Expression",
        xaxis="x2", yaxis="y2",
        showlegend=False,
    )

    # ── Build one set of edge traces per threshold step ───────────
    def build_edge_traces(pct_threshold):
        cutoff = np.percentile(weights, pct_threshold)
        keep   = weights >= cutoff

        retained = weights[keep]
        w_min, w_max = retained.min(), retained.max()
        if w_max > w_min:
            norm_w = (retained - w_min) / (w_max - w_min)
        else:
            norm_w = np.ones_like(retained)

        kept_idx = np.where(keep)[0]

        hub_ex, hub_ey, hub_ws   = [], [], []
        ring_ex, ring_ey, ring_ws = [], [], []

        for rank, i in enumerate(kept_idx):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            nw = float(norm_w[rank])

            is_hub = (u in hub_genes or v in hub_genes)
            if is_hub:
                hub_ex  += [x0, x1, None]
                hub_ey  += [y0, y1, None]
                hub_ws.append(nw)
            else:
                ring_ex  += [x0, x1, None]
                ring_ey  += [y0, y1, None]
                ring_ws.append(nw)

        avg_ring_w = float(np.mean(ring_ws)) if ring_ws else 0.5
        avg_hub_w  = float(np.mean(hub_ws))  if hub_ws  else 0.5

        ring_trace = go.Scatter(
            x=ring_ex, y=ring_ey,
            mode="lines",
            line=dict(
                color="black",
                width=1.0 + 3.0 * avg_ring_w,
            ),
            opacity=0.7,
            hoverinfo="skip",
            name="Ring edges",
            showlegend=False,
        )
        hub_trace = go.Scatter(
            x=hub_ex, y=hub_ey,
            mode="lines",
            line=dict(
                color="black",
                width=2.0 + 3.0 * avg_hub_w,
            ),
            opacity=0.85,
            hoverinfo="skip",
            name="Hub edges",
            showlegend=False,
        )
        n_kept = int(keep.sum())
        return ring_trace, hub_trace, n_kept

    # ── Assemble frames for slider ────────────────────────────────
    # Trace order in each frame: [ring_edges, hub_edges, nodes, bar]
    # Nodes and bar are static; only edge traces change.
    all_frames = []
    slider_steps = []

    for pct in thresholds:
        ring_tr, hub_tr, n_kept = build_edge_traces(pct)
        frame = go.Frame(
            data=[ring_tr, hub_tr, node_trace, bar_trace],
            name=str(pct),
        )
        all_frames.append(frame)
        slider_steps.append(dict(
            method="animate",
            args=[[str(pct)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
            label=f"{100 - pct}%  ({n_kept} edges)",
        ))

    # Initial frame = first threshold
    init_ring, init_hub, init_n = build_edge_traces(thresholds[0])

    # ── Layout ───────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=(
                f"Prototype graph — class {class_id} &nbsp;│&nbsp; "
                f"hi block: genes {hi_start}–{hi_start+4} &nbsp;│&nbsp; "
                f"lo block: genes {lo_start}–{lo_start+4}"
            ),
            font=dict(family="IBM Plex Mono", size=13, color="#1a1a2e"),
            x=0.01,
        ),
        paper_bgcolor="#f5f5f0",
        plot_bgcolor="#f5f5f0",
        font=dict(family="IBM Plex Mono", color="#1a1a2e"),

        # Left panel (graph)
        xaxis=dict(
            domain=[0, 0.58],
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="y",
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
        ),

        # Right panel (bar chart)
        xaxis2=dict(
            domain=[0.65, 1.0],
            title=dict(text="Gene index", font=dict(size=11)),
            showgrid=True, gridcolor="#ddd",
        ),
        yaxis2=dict(
            anchor="x2",
            title=dict(text="Mean expression [log2(TPM+1)]", font=dict(size=11)),
            range=[0, 4.5],
            showgrid=True, gridcolor="#ddd",
        ),

        # Reference lines on bar chart
        shapes=[
            dict(type="line", xref="x2", yref="y2",
                 x0=-1, x1=num_genes, y0=1.5, y1=1.5,
                 line=dict(color="grey", width=1, dash="dash")),
            dict(type="line", xref="x2", yref="y2",
                 x0=-1, x1=num_genes, y0=3.0, y1=3.0,
                 line=dict(color="#e05c5c", width=1, dash="dot")),
            dict(type="line", xref="x2", yref="y2",
                 x0=-1, x1=num_genes, y0=0.5, y1=0.5,
                 line=dict(color="#5b9bd5", width=1, dash="dot")),
        ],

        annotations=[
            dict(xref="x2", yref="y2", x=num_genes * 0.98, y=1.55,
                 text="background", showarrow=False,
                 font=dict(size=8, color="grey")),
            dict(xref="x2", yref="y2", x=num_genes * 0.98, y=3.05,
                 text="hi target", showarrow=False,
                 font=dict(size=8, color="#e05c5c")),
            dict(xref="x2", yref="y2", x=num_genes * 0.98, y=0.55,
                 text="lo target", showarrow=False,
                 font=dict(size=8, color="#5b9bd5")),
            # Legend annotations
            dict(xref="paper", yref="paper", x=0.0, y=-0.12,
                 text="◆ hub gene &nbsp;&nbsp; ● gene &nbsp;&nbsp; "
                      "<span style='color:#e05c5c'>■</span> hi block &nbsp;&nbsp; "
                      "<span style='color:#5b9bd5'>■</span> lo block &nbsp;&nbsp; "
                      "<span style='color:#c9a84c'>■</span> hub",
                 showarrow=False, font=dict(size=9), align="left"),
        ],

        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="Edges retained: ",
                font=dict(size=11, family="IBM Plex Mono"),
                xanchor="left",
            ),
            pad=dict(t=50, b=10),
            steps=slider_steps,
            bgcolor="#ddd",
            activebgcolor="#1a1a2e",
            bordercolor="#aaa",
            font=dict(family="IBM Plex Mono", size=9),
        )],

        margin=dict(l=20, r=20, t=70, b=100),
        height=560,
    )

    fig = go.Figure(
        data=[init_ring, init_hub, node_trace, bar_trace],
        layout=layout,
        frames=all_frames,
    )

    if save_path:
        fig.write_html(
            save_path,
            full_html=True,
            include_plotlyjs="cdn",
            auto_play=False,
        )
        print(f"Saved interactive HTML to {save_path}")

    return fig


# ── TensorBoard helper ────────────────────────────────────────────────────────

def add_graph_explanation_to_tensorboard(
    writer,
    fig,
    tag:         str = "graph/explanation",
    global_step: int = 0,
):
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    
    # Write to a temp file and embed via iframe
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_str)
        tmp_path = f.name

    # TensorBoard Text tab supports a limited iframe markdown tag
    iframe_md = f'<iframe src="file://{tmp_path}" width="100%" height="600px"></iframe>'
    writer.add_text(tag, iframe_md, global_step=global_step)