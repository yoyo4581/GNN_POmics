import wandb
import numpy as np
import plotly.graph_objects as go

def log_attention_percentiles(class_attention_dict, tissue_descriptions, run, epoch, split="train"):
    percentiles = np.arange(0, 101, 1)
    
    fig = go.Figure()
    
    for class_idx, all_attn in class_attention_dict.items():
        class_name = tissue_descriptions[class_idx]
        values = np.percentile(np.concatenate(all_attn), percentiles)
        
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=values,
            mode="lines",
            name=class_name,  # shows in legend, color-coded automatically
        ))
    
    fig.update_layout(
        title=f"Attention weight percentiles ({split}) — Epoch {epoch}",
        xaxis_title="Percentile",
        yaxis_title="Attention Weight",
        legend_title="Class",
    )
    
    run.log({
        f"attention/percentile_curve/{split}": wandb.Plotly(fig)
    }, step=epoch)