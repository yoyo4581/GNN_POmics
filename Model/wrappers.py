import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.nn import global_mean_pool
from .model import TissueClassificationPipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig


 
class ExplainerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        
        out, proj = self.model(x, edge_index, batch)
        # 🔴 Extract ONLY what the explainer needs
        logits = out["logits"]   # <-- adjust key if needed
        
        return logits

def build_explainer(
    model:  TissueClassificationPipeline,
) -> Explainer:
    """
    Builds a PyG Explainer for a single sample using GraphMaskExplainer.
 
    The wrapper is built fresh per sample because x_normed and batch are
    baked in as buffers. This is consistent with GraphMaskExplainer's
    per-instance design — it re-trains gates for each sample anyway.
 
    Call pattern:
        explainer   = build_explainer(model, sample, batch_vec)
        explanation = explainer(x=sample.x, edge_index=sample.edge_index, target=sample.y)
 
    GraphMaskExplainer parameters:
        num_layers=2        — exactly the two GATv2Conv layers in GATv2Backbone
        epochs=200          — Lagrangian optimization steps per layer
        lr=0.01             — gate network Adam lr
        penalty_scaling=5   — scales the E[L0] sparsity term (0–10)
        init_lambda=0.55    — initial Lagrange multiplier λ
        allowance=0.03      — tolerance β: a gate is kept open only if dropping
                              it degrades the prediction by more than 3%
 
    What GraphMaskExplainer does internally:
        1. _freeze_model(wrapper)    — freezes conv1, conv2, norms, head
        2. walks wrapper.modules()  — finds conv1, conv2 (MessagePassing)
                                      collects in_channels/out_channels for
                                      gate network sizing
        3. _set_masks(...)          — initializes one gate Linear + one
                                      baseline Parameter per layer
        4. for layer in reversed([0, 1]):
               optimize: L = KL(original_out, masked_out) + λ * E[L0(z)]
               update λ via dual ascent to enforce the allowance constraint
        5. forward() → Explanation(edge_mask=[E])
                        edge_mask ∈ [0,1], values > 0.5 indicate necessary edges
    """
    # Pre-compute normalized features once — outside the explainer's scope
    wrapper = ExplainerWrapper(model)
 
    return Explainer(
        model=wrapper,
        algorithm=GNNExplainer(
            epochs=200,
            lr=0.01
        ),
        explanation_type='model',
        edge_mask_type='object',
        node_mask_type=None,
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )