from torch_geometric.data import Data, Dataset, Batch
from typing import List
import torch
import os
import Bio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py
from Dataset.db_callers.Ensembl import EnsemblCaller
from Dataset.db_callers.Entrez import EntrezCaller
from Dataset.db_callers.GraphSnapshot import GraphSnapshot
import harmonypy as hm
from sklearn.decomposition import PCA


class GraphDataset(Dataset):
    def __init__(self, root, split='train'):
        structure = torch.load(f"{root}/graph_structure.pt", weights_only=False)
        self.edge_index = structure['edge_index']
        self.pathway_index = structure['pathway_index']
        self.mask = structure['mask']
        self.samples = torch.load(f'{root}/{split}_samples.pt', weights_only=False)
        self.labels = [s.y for s in self.samples]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.samples[idx]
        data.edge_index = self.edge_index
        data.pathway_index = self.pathway_index
        data.mask = self.mask
        return data




class DatasetBuilder:
    def __init__(self, folder_path: str, graph_snapshot: GraphSnapshot = None, meta_df: pd.DataFrame = None):
        self.gs = graph_snapshot if graph_snapshot is not None else GraphSnapshot()
        self.gs.build_edge_index()
        self.graph_template = self.gs.graph.graph
        self.edge_index = self.gs.graph.graph['edge_index']
        self.file_dataset = os.path.join(folder_path, 'ML_Dataset')
        self.meta_df = meta_df

    def process_gct(self, gct_file, label):
        filepath = os.path.join(self.intake_folder_path, gct_file)
        
        df = pd.read_csv(filepath, sep='\t', skiprows=2, index_col=0)
        description = df['Description']
        df = df.drop(columns=['Description'])
        
        # Recreate your composite index
        df.index = description.astype(str) + '|' + df.index.astype(str)
        
        return df
    
    def standardize(self, df):
        log_tpm = np.log2(df + 1)
            # Transpose from (genes x samples) to (samples x genes) for PCA
        df_z = pd.DataFrame(log_tpm, index=df.index, columns=df.columns)
        return df_z

    def _build_entrez_info(self, all_entrez_ids: list[int]) -> dict:
        """
        Batch query mygene: entrez_id -> {'ensembl': [...], 'symbol': str}
        """
        from mygene import MyGeneInfo
        mg = MyGeneInfo()
        entrez_info = {}
        batch_size = 1000
        all_entrez_strs = [str(e) for e in all_entrez_ids]

        for start in range(0, len(all_entrez_strs), batch_size):
            batch = all_entrez_strs[start:start + batch_size]
            results = mg.querymany(
                batch,
                scopes='entrezgene',
                fields='ensembl.gene,symbol',
                species='human'
            )
            for r in results:
                eid = int(r['query'])
                ensembl_ids = []
                if 'ensembl' in r:
                    raw = r['ensembl']
                    raw = raw if isinstance(raw, list) else [raw]
                    ensembl_ids = [e['gene'] for e in raw]

                entrez_info[eid] = {
                    'ensembl': ensembl_ids,
                    'symbol': r.get('symbol', '').upper()
                }

        return entrez_info


    def process_genes(self, combined_expression: pd.DataFrame) -> pd.DataFrame:
        norm_df = self.standardize(combined_expression)

        # --- Parse table index (Symbol|EnsemblID) ---
        index_series = norm_df.index.to_series()
        table_symbols = index_series.str.split('|').str[0].str.upper()
        table_ensembl = index_series.str.split('|').str[1].str.split('.').str[0]  # strip version

        # Build O(1) lookups: first match wins
        ensembl_to_row: dict[str, int] = {}
        for i, ens in enumerate(table_ensembl):
            if ens not in ensembl_to_row:
                ensembl_to_row[ens] = i

        symbol_to_row: dict[str, int] = {}
        for i, sym in enumerate(table_symbols):
            if sym not in symbol_to_row:
                symbol_to_row[sym] = i

        # --- Collect graph node info ---
        sorted_nodes = sorted(
            self.gs.graph.nodes(data=True),
            key=lambda x: int(x[0].split(':')[1])
        )

        # primary entrez + all synonyms per node
        node_info: list[tuple[int, list[int]]] = []
        all_entrez_ids: set[int] = set()
        for node, data in sorted_nodes:
            primary = int(node.split(':')[1])
            synonyms = [int(s.split(':')[1]) for s in data.get('synonyms', [])]
            all_ids = [primary] + synonyms
            node_info.append((primary, all_ids))
            all_entrez_ids.update(all_ids)

        # --- Single batch mygene call for all entrez IDs ---
        print(f"Querying mygene for {len(all_entrez_ids)} entrez IDs...")
        entrez_info = self._build_entrez_info(list(all_entrez_ids))

        # --- Match each graph node to a table row ---
        # table original index value -> graph primary entrez (int)
        matched_table_index: dict[str, int] = {}
        matched_rows: set[int] = set()  # prevent double-assigning a table row

        for primary, all_ids in node_info:
            matched_row = None

            # 1. Try Ensembl match (primary entrez first, then synonyms)
            for eid in all_ids:
                for ens in entrez_info.get(eid, {}).get('ensembl', []):
                    row = ensembl_to_row.get(ens)
                    if row is not None and row not in matched_rows:
                        matched_row = row
                        break
                if matched_row is not None:
                    break

            # 2. Fallback: symbol match
            if matched_row is None:
                for eid in all_ids:
                    sym = entrez_info.get(eid, {}).get('symbol', '')
                    if sym:
                        row = symbol_to_row.get(sym)
                        if row is not None and row not in matched_rows:
                            matched_row = row
                            break

            if matched_row is not None:
                orig_idx = norm_df.index[matched_row]
                matched_table_index[orig_idx] = primary
                matched_rows.add(matched_row)

        print(f"Matched {len(matched_table_index)} / {len(node_info)} graph nodes to table rows")

        # --- Rename matched rows to primary entrez ID, drop unmatched ---
        norm_df = norm_df.rename(index=matched_table_index)
        primary_entrez_set = {p for p, _ in node_info}
        norm_df = norm_df[~norm_df.index.duplicated(keep='first')]

        return norm_df
    

    def split_data(self, df, df_y):
        train_idx, test_idx = train_test_split(
            range(len(df)), test_size=0.2, stratify=df_y, random_state=42
        )
        train_idx, valid_idx = train_test_split(
            train_idx, test_size=0.2, stratify=df_y.iloc[train_idx], random_state=42
        )

        return train_idx, valid_idx, test_idx

    def order_expression(self, df: pd.DataFrame):
        sorted_nodes = sorted(
            self.gs.graph.nodes(data=True),
            key=lambda x: int(x[0].split(':')[1])
        )
        ordered_entrez = [int(node.split(":")[1]) for node, _ in sorted_nodes]
        df = df.reindex(ordered_entrez)

        return df


    #Takes in feature data and with edge_index turns it into a PathwaySample.
    def create_dataset(self, intake_folder_path:str):
        self.intake_folder_path = intake_folder_path
        # open folders and do search to entrez Id only once.
        gct_files = sorted([
            f for f in os.listdir(self.intake_folder_path)
            if f.endswith(".gct")
        ])

        all_data = []
        all_labels = []

        for label, filename, in enumerate(gct_files):
            df = self.process_gct(filename, label)
            all_data.append(df)
            all_labels += [label]*len(df.columns)

        combined_expression = pd.concat(all_data, axis=1)
        combined_expression.index.name = "gene"

        assert len(combined_expression.columns) == len(all_labels)

        df_labels = pd.DataFrame(all_labels, index = combined_expression.columns, columns=['x'])

        # log2, standard scaler, and gene entrezId.
        self.train_idx, self.valid_idx, self.test_idx = self.split_data(combined_expression.T, df_labels)
        scaled_expression = self.process_genes(combined_expression)

        # Now mod the entrez gene ids so that they only contain genes with entrez ids
        # We can now reorder
        ordered_expression = self.order_expression(scaled_expression)
        self.dataset_split_create(ordered_expression, df_labels)


    def dataset_split_create(self, exprs_df: pd.DataFrame, df_labels: pd.DataFrame):
        # Features
        x = torch.tensor(exprs_df.fillna(0).to_numpy(), dtype=torch.float)
        y = torch.tensor(df_labels['x'].tolist(), dtype=torch.long)

        # Mask (which nodes actually have data)
        mask = torch.tensor(~exprs_df.isna().all(axis=1).values, dtype=torch.bool)

        train_samples = [Data(x=x[:, idx], 
                                y=y[idx]) for idx in self.train_idx]
        valid_samples = [Data(x=x[:, idx], 
                                y=y[idx]) for idx in self.valid_idx]
        test_samples = [Data(x=x[:, idx], 
                                y=y[idx]) for idx in self.test_idx]
        torch.save({
            'edge_index': self.gs.graph.graph['edge_index'],
            'pathway_index': self.gs.graph.graph['pathway_index'],
            'mask': mask
        }, os.path.join(self.file_dataset, 'graph_structure.pt'))

        torch.save(train_samples, os.path.join(self.file_dataset, 'train_samples.pt'))
        torch.save(valid_samples, os.path.join(self.file_dataset, 'valid_samples.pt'))
        torch.save(test_samples, os.path.join(self.file_dataset, 'test_samples.pt'))

        exprs_df.to_csv(os.path.join(self.file_dataset, 'expression_dataframe.csv'), index=True)
        df_labels.to_csv(os.path.join(self.file_dataset, 'sample_labels.csv'), index=True)

        self.gs.save_to_pickle(os.path.join(self.file_dataset, 'graph_snapshot.pkl'))