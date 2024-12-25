import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GNNDataset(Dataset):
    def __init__(self, features, edge_indices, targets):
        """
        Custom dataset for GNN data in federated learning.
        
        Args:
            features: List of node feature matrices
            edge_indices: List of edge index matrices
            targets: List of target values
        """
        self.features = features
        self.edge_indices = edge_indices
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        """Return a single graph with its features and target."""
        x = self.features[idx]
        edge_index = self.edge_indices[idx]
        y = self.targets[idx]
        
        # Create a PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
