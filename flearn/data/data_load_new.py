import torch
import numpy as np
from torch_geometric.data import Data
import random

def create_federated_data(n_users=10, n_uavs=2, n_features=2):
    """Create synthetic data for federated UAV network simulation."""
    # Generate random positions for users and UAVs
    user_positions = np.random.rand(n_users, 2) * 100  # Random positions in 100x100 area
    uav_positions = np.random.rand(n_uavs, 2) * 100
    
    # Initialize lists to store data for each UAV
    features_list = []
    edge_indices_list = []
    targets_list = []
    
    # Create data for each UAV
    for uav_idx in range(n_uavs):
        # Calculate distances to all users
        distances = np.sqrt(np.sum((user_positions - uav_positions[uav_idx])**2, axis=1))
        
        # Connect to all users (create edges)
        edges = []
        for i in range(n_users):
            edges.append([0, i+1])  # UAV (node 0) to user connections
            edges.append([i+1, 0])  # User to UAV connections
        
        # Create features matrix
        features = np.zeros((n_users + 1, n_features))
        features[0] = uav_positions[uav_idx]  # UAV features
        features[1:] = user_positions  # User features
        
        # Create target values (e.g., communication quality)
        targets = 1.0 / (1.0 + distances)  # Communication quality for all users
        
        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        edge_index = torch.LongTensor(edges).t()
        targets = torch.FloatTensor(targets)
        
        # Store data
        features_list.append(features)
        edge_indices_list.append(edge_index)
        targets_list.append(targets)
    
    return features_list, edge_indices_list, targets_list
