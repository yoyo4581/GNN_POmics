from bokeh.palettes import Turbo256
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from itertools import combinations
import numpy as np
import wandb
import os

def global_layout(graph, gt):
    pathway_edges = gt.compute_pathway_membership(graph)
    pathway_ids = list(pathway_edges.keys())
    n = len(pathway_ids)

    # Color palette
    palette = [Turbo256[int(i * 255 / (n - 1))] for i in range(n)]
    pathway_color = {pw: palette[i] for i, pw in enumerate(pathway_ids)}

    # Genes per pathway
    pathway_genes = {}
    for pw, edges in pathway_edges.items():
        genes = set()
        for u, v in edges:
            genes.add(u)
            genes.add(v)
        pathway_genes[pw] = sorted(genes)

    # Arc sizes proportional to number of member genes
    gene_counts = np.array([len(pathway_genes[pw]) for pw in pathway_ids], dtype=float)
    total_arc = 2 * np.pi
    arc_sizes = (gene_counts / gene_counts.sum()) * total_arc

    # Build pathway start/end angles
    start_angles = np.zeros(n)
    for i in range(1, n):
        start_angles[i] = start_angles[i - 1] + arc_sizes[i - 1]
    end_angles = start_angles + arc_sizes
    mid_angles = (start_angles + end_angles) / 2

    # Gene slot angles within each pathway arc
    gene_angle = {}
    for i, pw in enumerate(pathway_ids):
        genes = pathway_genes[pw]
        m = len(genes)
        if m == 1:
            slots = np.array([mid_angles[i]])
        else:
            padding = arc_sizes[i] * 0.05
            slots = np.linspace(start_angles[i] + padding, end_angles[i] - padding, m)
        gene_angle[pw] = {gene: slots[j] for j, gene in enumerate(genes)}

    # Build figure
    p = figure(width=1200, height=1200, x_range=(-1.8, 1.8), y_range=(-1.8, 1.8),
               toolbar_location=None)
    p.axis.visible = False
    p.grid.visible = False

    # Batch draw arcs — store the renderer so subgraph_chord can update alpha
    arc_source = ColumnDataSource(dict(
        start_angle=start_angles.tolist(),
        end_angle=end_angles.tolist(),
        line_color=palette,
        line_alpha=[0.8] * n,
    ))
    arc_renderer = p.arc(
        x=0, y=0, radius=1.0,
        start_angle='start_angle', end_angle='end_angle',
        line_color='line_color', line_alpha='line_alpha',
        line_width=12, source=arc_source,
    )

    return dict(
        p=p,
        arc_source=arc_source,
        arc_renderer=arc_renderer,
        pathway_ids=pathway_ids,
        pathway_color=pathway_color,
        pathway_genes=pathway_genes,
        gene_angle=gene_angle,
        start_angles=start_angles,
        end_angles=end_angles,
        mid_angles=mid_angles,
        graph=graph,
    )
import numpy as np
from itertools import combinations
from bokeh.models import ColumnDataSource, HoverTool


# ── helpers ───────────────────────────────────────────────────────────────────

def _lerp(a, b, t=0.5):
    return a + (b - a) * t


def _split_cubic_at_half(x0, y0, cx0, cy0, cx1, cy1, x1, y1):
    """
    Split a cubic bezier (Bokeh convention: p0, cx0, cx1, p1) at t=0.5.
    Returns two sets of cubic control points:
        first  = (x0,y0) → mid  with inner control points
        second = mid      → (x1,y1) with inner control points
    """
    p0  = np.array([x0,  y0 ])
    cp0 = np.array([cx0, cy0])
    cp1 = np.array([cx1, cy1])
    p1  = np.array([x1,  y1 ])

    p01   = _lerp(p0,  cp0)
    p12   = _lerp(cp0, cp1)
    p23   = _lerp(cp1, p1 )
    p012  = _lerp(p01,  p12)
    p123  = _lerp(p12,  p23)
    mid   = _lerp(p012, p123)   # exact midpoint on the curve

    # first half:  p0 → mid,  controls: p01, p012
    # second half: mid → p1,  controls: p123, p23
    return (
        p0[0],  p0[1],  p01[0],  p01[1],  p012[0], p012[1], mid[0], mid[1],
        mid[0], mid[1], p123[0], p123[1], p23[0],  p23[1],  p1[0],  p1[1],
    )


# ── main function ─────────────────────────────────────────────────────────────

def subgraph_chord(subgraph, gt, sub_weights, layout):

    p             = layout['p']
    arc_source    = layout['arc_source']
    pathway_ids   = layout['pathway_ids']
    pathway_color = layout['pathway_color']
    gene_angle    = layout['gene_angle']
    mid_angles    = layout['mid_angles']
    graph         = layout['graph']

    # ── dim inactive arcs ────────────────────────────────────────────────────
    sub_pathway_edges = gt.compute_pathway_membership(subgraph)
    active_pathways   = set(sub_pathway_edges.keys())

    arc_source.data['line_alpha'] = [
        0.8 if pw in active_pathways else 0.15
        for pw in pathway_ids
    ]

    # ── gene ticks ───────────────────────────────────────────────────────────
    subgraph_nodes = set(subgraph.nodes())
    tick_x0, tick_y0, tick_x1, tick_y1 = [], [], [], []
    for pw in active_pathways:
        if pw not in gene_angle:
            continue
        for gene, angle in gene_angle[pw].items():
            if gene not in subgraph_nodes:
                continue
            tick_x0.append(np.cos(angle) * 0.95)
            tick_y0.append(np.sin(angle) * 0.95)
            tick_x1.append(np.cos(angle) * 1.05)
            tick_y1.append(np.sin(angle) * 1.05)

    if tick_x0:
        p.segment(x0=tick_x0, y0=tick_y0, x1=tick_x1, y1=tick_y1,
                  line_width=1.5, line_color='white', line_alpha=0.7)

    # ── normalise weights ────────────────────────────────────────────────────
    weight_values = np.array(list(sub_weights.values()), dtype=float)
    w_min, w_max  = weight_values.min(), weight_values.max()
    w_range       = (w_max - w_min) or 1.0   # avoid div-by-zero

    def _normalised_alpha(w):
        return 0.15 + 0.65 * (w - w_min) / w_range

    # ── batch bezier data (split into two halves per chord) ──────────────────
    # Each chord produces TWO bezier segments; we store them flat in the same
    # ColumnDataSource so a single p.bezier() call draws everything.

    bx0, by0, bx1, by1           = [], [], [], []
    bcx0, bcy0, bcx1, bcy1       = [], [], [], []
    b_colors, b_alphas            = [], []
    b_sources, b_targets          = [], []
    b_pathways_col     = []
    mx_list, my_list              = [], []   # midpoints for hover

    def _add_split_bezier(x0, y0, x1, y1,
                          color_src, color_tgt,
                          alpha, src, tgt, pw_label_src, pw_label_tgt):
        """
        Append two half-bezier segments (source-color half + target-color half)
        for the chord from (x0,y0) to (x1,y1).
        Bokeh's bezier uses cx0/cy0 as the control point near p0,
        and cx1/cy1 as the control point near p1.
        We use the pull-toward-origin trick for the full curve, then split.
        """
        # Full cubic control points (same pull-to-origin as original code)
        full_cx0, full_cy0 = x1 * 0.5, y1 * 0.5   # ctrl near p0, pulled toward p1
        full_cx1, full_cy1 = x0 * 0.5, y0 * 0.5   # ctrl near p1, pulled toward p0

        (ax0, ay0, acx0, acy0, acx1, acy1, ax1, ay1,
         bX0, bY0, bcX0, bcY0, bcX1, bcY1, bX1, bY1) = _split_cubic_at_half(
            x0, y0, full_cx0, full_cy0, full_cx1, full_cy1, x1, y1
        )

        mid_x, mid_y = ax1, ay1   # == bX0, bY0

        for (sx, sy, ex, ey, ccx0, ccy0, ccx1, ccy1, color, pw_label) in [
            (ax0, ay0, ax1, ay1, acx0, acy0, acx1, acy1, color_src, pw_label_src),
            (bX0, bY0, bX1, bY1, bcX0, bcY0, bcX1, bcY1, color_tgt, pw_label_tgt),
        ]:
            bx0.append(sx);   by0.append(sy)
            bx1.append(ex);   by1.append(ey)
            bcx0.append(ccx0); bcy0.append(ccy0)
            bcx1.append(ccx1); bcy1.append(ccy1)
            b_colors.append(color)
            b_alphas.append(alpha)
            b_sources.append(src)
            b_targets.append(tgt)
            b_pathways_col.append(pw_label)
            mx_list.append(mid_x)
            my_list.append(mid_y)

    # ── iterate edges ─────────────────────────────────────────────────────────
    for edge_u, edge_v, data in subgraph.edges(data=True):
        pathways = data.get('pathways', [])
        alpha   = sub_weights[(edge_u, edge_v)]

        if len(pathways) > 1:
            # Edge spans multiple pathways → one chord per pair
            for pw_a, pw_b in combinations(pathways, 2):
                # checks that the pathways are in the chord diagram
                if pw_a not in gene_angle or pw_b not in gene_angle:
                    continue
                # checks that the start gene and the end gene are in the chord diagram
                if edge_u not in gene_angle[pw_a] or edge_v not in gene_angle[pw_b]:
                    continue
                a0 = gene_angle[pw_a][edge_u]
                a1 = gene_angle[pw_b][edge_v]
                x0, y0 = np.cos(a0), np.sin(a0)
                x1, y1 = np.cos(a1), np.sin(a1)
                pw_label_src = graph.graph['pid_name'][pw_a]
                pw_label_tgt = graph.graph['pid_name'][pw_b]
                _add_split_bezier(x0, y0, x1, y1,
                                  pathway_color[pw_a], pathway_color[pw_b],
                                  alpha, edge_u, edge_v, pw_label_src, pw_label_tgt)
        else:
            # Edge within a single pathway → self-pathway chord (still two colors,
            # both halves use the same color, but the split is consistent)
            pathway = pathways[0]
            if pathway not in gene_angle:
                continue
            if edge_u not in gene_angle[pathway] or edge_v not in gene_angle[pathway]:
                continue
            a0 = gene_angle[pathway][edge_u]
            a1 = gene_angle[pathway][edge_v]
            x0, y0 = np.cos(a0), np.sin(a0)
            x1, y1 = np.cos(a1), np.sin(a1)
            pw_label = graph.graph['pid_name'][pathway]
            color    = pathway_color[pathway]
            _add_split_bezier(x0, y0, x1, y1,
                              color, color,
                              alpha, edge_u, edge_v, pw_label, pw_label)

    # ── draw ──────────────────────────────────────────────────────────────────
    if bx0:
        bezier_source = ColumnDataSource(dict(
            x0=bx0, y0=by0, x1=bx1, y1=by1,
            cx0=bcx0, cy0=bcy0, cx1=bcx1, cy1=bcy1,
            line_color=b_colors, line_alpha=b_alphas,
            source=b_sources,
            target=b_targets,
            pathway=b_pathways_col,
            mx=mx_list,
            my=my_list,
        ))

        bezier_render = p.bezier(
            x0='x0', y0='y0', x1='x1', y1='y1',
            cx0='cx0', cy0='cy0', cx1='cx1', cy1='cy1',
            line_color='line_color', line_alpha='line_alpha',
            line_width=3, source=bezier_source,
        )

        # Invisible scatter for hover hit detection at midpoints
        p.scatter(x='mx', y='my', size=10,
                  fill_alpha=0, line_alpha=0,
                  source=bezier_source)

        hover = HoverTool(
            renderers=[bezier_render],
            tooltips=[
                ('Source',  '@source'),
                ('Target',  '@target'),
                ('Pathway', '@pathway'),
                ('Weight',  '@line_alpha'),
            ],
        )
        p.add_tools(hover)

    # ── pathway labels ────────────────────────────────────────────────────────
    active_indices    = [i for i, pw in enumerate(pathway_ids) if pw in active_pathways]
    active_names      = [graph.graph['pid_name'][pathway_ids[i]] for i in active_indices]
    active_mid_angles = mid_angles[active_indices]

    if active_names:
        label_source = ColumnDataSource(dict(
            x=(np.cos(active_mid_angles) * 1.25).tolist(),
            y=(np.sin(active_mid_angles) * 1.25).tolist(),
            names=active_names,
        ))
        p.text(x='x', y='y', text='names', source=label_source,
               text_font_size='7pt', text_align='center', text_baseline='middle')

    return p




def log_bokeh_figures(figures_by_class, epoch, run, split):
  from bokeh.embed import file_html
  from bokeh.resources import CDN
  from bokeh.plotting import save, output_file

  log_dict = {}
  for class_idx, fig in figures_by_class.items():
    html = file_html(fig, CDN)
    log_dict[f"{split}/chord_class_{class_idx}"] = wandb.Html(html)
  
  run.log(log_dict, step=epoch)
