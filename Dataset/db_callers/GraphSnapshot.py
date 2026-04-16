import networkx as nx
from neo4j import GraphDatabase
import networkx as nx


class GraphSnapshot:
    def __init__(self, pickle_filepath = None):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "458458Yoyo"))
        if pickle_filepath:
            print(f"Loading graph from {pickle_filepath}...")
            self.graph = self.from_pickle(pickle_filepath)
        else:
            print("No pickle filepath provided, fetching graph from Neo4j...")
            self.graph = self.fetch_graph()

    def from_pickle(self, pickle_filepath)->nx.DiGraph:
        import pickle
        with open(pickle_filepath, 'rb') as f:
            return pickle.load(f)

    def save_to_pickle(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)

    def fetch_graph(self)->nx.DiGraph:
        G = nx.DiGraph()
        G.graph['pid_name'] = {}
        G.graph['gene_idx'] = {}
        G.graph['idx_gene'] = {}
        G.graph['pid_idx'] = {}
        try:
            self.driver.verify_connectivity()

            with self.driver.session() as session:
                # Fetch all nodes
                nodes = session.run("MATCH (n:Gene) RETURN n.name AS id, labels(n) AS labels, properties(n) AS props")
                for record in nodes:
                    G.add_node(record["id"], labels=record["labels"], **record["props"])

                # Fetch all relationships
                rels = session.run("""
                    MATCH (a:Gene)-[r]->(b:Gene)
                    RETURN a.name AS source, b.name AS target, type(r) AS type, properties(r) AS props
                """)
                for record in rels:
                    G.add_edge(record["source"], record["target"], type=record["type"], **record["props"])

                pathway_data = session.run("""
                    MATCH (p:Pathway)
                    RETURN p.name, p.title
                    ORDER BY toInteger(apoc.text.regexGroups(p.name, '(\d+)$')[0][1])
                """)
                for idx, (pathway_name, pathway_title) in enumerate(pathway_data):
                    G.graph['pid_name'].update({pathway_name: pathway_title})
                    G.graph['pid_idx'].update({pathway_name: idx})

            return G
        except Exception as e:
            print(f"Failed: {e}")
            return G
        finally:
            self.driver.close()

    def build_edge_index(self):
        # Takes a networkX graph and builds a tensor object
        # I need a mapper, basically we map hsa ids to it and keep it constant
        # idx_entrez-   index: hsa_id
        # gene_id- list referencing index, where position references node. For example gene hsa:226 has edge_index of 0, positionally in gene_id its the 0th item, and it holds the index 50 which maps to hsa_id 226

        import torch

        sorted_nodes = sorted(self.graph.nodes(data=True), key=lambda x: int(x[0].split(':')[1]))
        for idx, (node, data) in enumerate(sorted_nodes):
            self.graph.graph['gene_idx'][node] = idx
            self.graph.graph['idx_gene'][idx] = node

        pathway_index = []
        edge_index = []
        for src, dst, data in self.graph.edges(data=True):
            src_idx = self.graph.graph['gene_idx'][src]
            dst_idx = self.graph.graph['gene_idx'][dst]
            edge_index.append([src_idx, dst_idx])

            for p_name in data.get('pathways', []):
                p_idx = self.graph.graph['pid_idx'][p_name]
                pathway_index.append([src_idx, p_idx])
                pathway_index.append([dst_idx, p_idx])

        pathway_index = list(set(map(tuple, pathway_index)))

        # [2, E] tensor where top is the index of the src node, bottom is the index of the trgt node.
        self.graph.graph['edge_index'] = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.graph.graph['pathway_index'] = torch.tensor(pathway_index, dtype=torch.long).t().contiguous()



