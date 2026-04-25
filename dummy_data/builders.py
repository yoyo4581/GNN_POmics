import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from typing import List

def build_label_embeddings(label_names):
    """
    Stub. In practice:
 
        from sentence_transformers import SentenceTransformer
        encoder      = SentenceTransformer('all-mpnet-base-v2')
        descriptions = ["breast ductal epithelium", "liver hepatocytes", ...]`
        embeddings   = torch.tensor(encoder.encode(descriptions))
    """
    emb   = F.normalize(torch.randn(num_classes, emb_dim), dim=-1)
    names = [f"tissue_{i}" for i in range(num_classes)]
    return emb, names
 
def build_graph_topology(num_genes: int) -> torch.Tensor:
    """
    Build the shared edge_index once. Every sample uses this same structure,
    matching the biological assumption that the graph represents a fixed
    gene-gene interaction network.
 
    Topology: a scale-free-like graph built from two components:
        - A ring (each gene connected to its two neighbours) — provides a
          baseline of local connectivity across all nodes.
        - Hub connections — a small set of 'hub' genes (simulating TFs or
          highly connected regulators) connected to every 8th gene.
 
    This gives a heterogeneous degree distribution similar to real GRNs,
    where a few hub nodes dominate and most nodes are low-degree.
 
    Returns edge_index: [2, E] with edges in both directions (undirected).
    """
    edges = []
 
    # Ring backbone — every node connected to its immediate neighbours
    for i in range(num_genes):
        edges.append((i, (i + 1) % num_genes))
        edges.append(((i + 1) % num_genes, i))
 
    # Hub genes: indices 0, 1, 2 act as highly connected regulators
    hub_genes = [0, 1, 2]
    for hub in hub_genes:
        for target in range(0, num_genes, 8):
            if target != hub:
                edges.append((hub, target))
                edges.append((target, hub))
 
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index



def build_structured_data(
    num_genes:   int = 64,
    num_classes: int = 10,
    samples_per_class: int = 20,
    noise_std:   float = 0.1,
) -> List[Data]:
    """
    Generate synthetic samples with class-discriminative node feature patterns,
    all sharing the same edge_index.
 
    Each class is defined by a prototype expression vector in log2(TPM+1) space.
    The prototype has:
        - A HIGH expression block: genes [class_id*5 : class_id*5 + 5]
          set to ~3.0 (roughly 8-fold over background in TPM space)
        - A LOW expression block: genes [class_id*5+5 : class_id*5+10]
          set to ~0.5
        - Background: all other genes drawn from N(1.5, 0.2), representing
          housekeeping expression
 
    Samples are the prototype + Gaussian noise (noise_std).
 
    This gives the model a clear signal to learn: the pattern of which gene
    block is highly expressed relative to background is class-discriminative,
    and that signal propagates through the shared graph topology via attention.
    The GraphMaskExplainer should then recover the hub edges connected to the
    high-expression block as the explanatory subgraph.
 
    Args:
        num_genes:         number of nodes (genes) per graph
        num_classes:       number of tissue classes
        samples_per_class: number of samples per class
        noise_std:         Gaussian noise std added to prototypes
 
    Returns:
        List of Data objects, all sharing the same edge_index.
    """
    edge_index = build_graph_topology(num_genes)
 
    # Build class prototypes
    prototypes = []
    for cls in range(num_classes):
        proto = torch.full((num_genes,), 1.5)            # housekeeping background
        hi_start = (cls * 5) % num_genes
        lo_start = (cls * 5 + 5) % num_genes
        hi_end   = min(hi_start + 5, num_genes)
        lo_end   = min(lo_start + 5, num_genes)
        proto[hi_start:hi_end] = 3.0                     # upregulated block
        proto[lo_start:lo_end] = 0.5                     # downregulated block
        prototypes.append(proto)
 
    graphs = []
    for cls, proto in enumerate(prototypes):
        for _ in range(samples_per_class):
            # Each gene's expression is the prototype + independent noise,
            # then clamped to [0, 6] — a realistic log2(TPM+1) range.
            noise = torch.randn(num_genes) * noise_std
            x     = (proto + noise).clamp(0.0, 6.0).unsqueeze(1)  # [N, 1]
            # Expand to feature dim expected by model: repeat to num_genes columns
            # so x is [num_genes, num_genes] as in the real data (one feature per gene pair)
            # For the prototype test we use a single feature column broadcast across dims.
            x = x.expand(num_genes, num_genes).clone()
            y = torch.tensor([cls], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
 
    return graphs
 