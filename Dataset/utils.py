import torch
import Dataset.db_callers.GraphSnapshot as GraphSnapshot

class GraphTraverse:
    def __init__(self, graph_snapshot: GraphSnapshot):
        self.graph_snapshot = graph_snapshot

    def fetch_edge_index(self, batch, src_node, dst_node):
        edge_index = batch.edge_index
        src_idx = self.graph_snapshot.graph.graph['gene_idx']['hsa:'+str(src_node)]
        dst_idx = self.graph_snapshot.graph.graph['gene_idx']['hsa:'+str(dst_node)]

        node_mask = (edge_index[0] == src_idx) & (edge_index[1] == dst_idx)

        return edge_index[:,node_mask]

    def fetch_edge_shared_pathways(self, batch, src_idx, dst_idx):
        node_ids, pathway_ids = batch.pathway_index
        node_mask = (node_ids == src_idx) | (node_ids == dst_idx)

        p = pathway_ids[node_mask]
        shared = p.unique(return_counts=True)
        shared_pathways = shared[0][shared[1] > 1]

        mask = node_mask & torch.isin(pathway_ids, shared_pathways)

        pathways = batch.pathway_index[:, mask]
        return pathways