import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm 
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, cache_dir="cache"):
        dataset_name = os.path.basename(os.path.dirname(filename))  
        self.raw = filename
        self.cache_dir = os.path.join(cache_dir, dataset_name)  # Store per dataset
        self.cache_file = os.path.join(self.cache_dir, os.path.basename(filename) + ".pt")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load from cache if available
        if os.path.exists(self.cache_file):
            print(f"Loading cached graphs from {self.cache_file}...")
            self.graphs = torch.load(self.cache_file)
        else:
            print(f"Processing raw graphs from {filename}...")
            self.graphs = self.loadGraphs(self.raw)
            torch.save(self.graphs, self.cache_file)  # Save cache for next time
            print(f"Cached dataset saved at {self.cache_file}")

        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    edge_index = edge_index.to_sparse_coo()
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)








