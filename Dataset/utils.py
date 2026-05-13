import torch
import Dataset.db_callers.GraphSnapshot as GraphSnapshot
import networkx as nx

class GraphTraverse:
    def __init__(self, graph_snapshot: GraphSnapshot):
        self.graph_snapshot = graph_snapshot
        self.global_graph = self.graph_snapshot.graph

    def fetch_edge_index(self, batch, src_node, dst_node):
        edge_index = batch.edge_index
        src_idx = self.global_graph.graph['gene_idx']['hsa:'+str(src_node)]
        dst_idx = self.global_graph.graph['gene_idx']['hsa:'+str(dst_node)]

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

    def subgraph_using_weights(self, edge_mask, top_k:int = 25):
      """
      Takes in the networkX main graph and edge_mask weights.
      Subgraphs it, returning the subgraph and its accompanying edgeweights.

      Return:
      subgraph: nx.diGraph()
      weights: dict[(edge_u, edge_v)]: weight
      """

      edge_index = self.global_graph.graph['edge_index']
      idx_gene = self.global_graph.graph['idx_gene']
      
      weights, indices = torch.topk(edge_mask, top_k)
      subbed_edges = edge_index[:, indices]

      edge_weights = {}
      edge_list = []
      for i, weight in enumerate(weights.tolist()):
        src = idx_gene[subbed_edges[0, i].item()]
        dst = idx_gene[subbed_edges[1, i].item()]
        edge_weights[(src, dst)] = weight
        edge_list.append((src, dst))

      # Build subgraph from only the top-k edges
      subgraph = nx.DiGraph()
      subgraph.add_edges_from(edge_list)

      # Copy over edge attributes from the original graph
      for u, v in edge_list:
          if self.global_graph.has_edge(u, v):
              subgraph[u][v].update(self.global_graph[u][v])

      return subgraph, edge_weights

    def compute_pathway_membership(self, graph: nx.DiGraph):
      from collections import defaultdict

      pathway_dict = defaultdict(set)
      for u, v, data in graph.edges(data=True):
        for pathway in data['pathways']:
          pathway_dict[pathway].add((u, v))

      return pathway_dict

    def show_cross_pathway_edges(self, graph: nx.DiGraph):
      pathway_subedges = self.compute_pathway_membership(graph)
      #Iterate through subgraph
      for edge_u, edge_v, data in graph.edges(data=True):
        pathways = data.get('pathways', [])
        # More than one pathway per edge
        if len(pathways) >1:
          is_in_subgraph = [pathway in pathway_subedges for pathway in pathways]
          if is_in_subgraph.sum() > 1:
            cross_pathway = pathways[is_in_subgraph]
            print('Edge: {edge_u}, {edge_v} \n in Pathways: {cross_pathway}')
      



