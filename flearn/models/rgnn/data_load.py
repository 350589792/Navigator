'''
Data creation and partitioning for UAV network with federated learning support.
Main functions:
- data_create: Creates basic dataset with user pairs and UAV positions
- create_federated_data: Partitions data for federated learning clients (UAVs)
'''
import torch
import numpy as np
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from sklearn.cluster import KMeans


def data_create(N, M, device='cpu'):
    '''
    Creates synthetic data for UAV network simulation.
    
    Args:
        N: Number of user pairs
        M: Number of UAVs
        device: Device to place tensors on (default: 'cpu')
    
    Returns:
        tuple: (features, edge_index, targets)
            features: User and UAV positions [2N+M, 2] (first 2N are user coords, last M are UAV coords)
            edge_index: Edge indices [2, E] where E is number of edges
            targets: Target values [2N+M] representing communication quality for each node
    '''
    if N < 1 or M < 1:
        raise ValueError(f"Invalid network size: N={N}, M={M}")
        
    try:
        # Generate random user positions
        user_positions = torch.rand(2*N, 2, device=device)  # [2N, 2]
        
        # Use KMeans to determine UAV positions based on user distribution
        kmeans = KMeans(n_clusters=M, random_state=42)
        uav_positions = torch.tensor(
            kmeans.fit(user_positions.cpu().numpy()).cluster_centers_,
            dtype=torch.float32,
            device=device
        )  # [M, 2]
        
        # Combine user and UAV positions
        features = torch.cat([user_positions, uav_positions], dim=0)  # [2N+M, 2]
        
        # Create edges between users and UAVs
        user_count = 2*N
        edge_list = []
        
        # Add edges: UAV -> User and User -> UAV
        for uav_idx in range(M):
            uav_node = user_count + uav_idx
            for user_idx in range(user_count):
                edge_list.extend([
                    [uav_node, user_idx],    # UAV -> User
                    [user_idx, uav_node]     # User -> UAV
                ])
        
        # Convert edge list to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()  # [2, E]
        
        # Generate target values based on network topology
        targets = torch.zeros(2*N + M, device=device)
        
        # For each user, calculate coverage based on distance to nearest UAV
        for user_idx in range(2*N):
            user_pos = user_positions[user_idx].unsqueeze(0)  # [1, 2]
            distances = torch.cdist(user_pos, uav_positions)  # [1, M]
            min_distance = torch.min(distances)
            # Target is inverse of normalized distance (closer = better)
            targets[user_idx] = torch.exp(-min_distance)  # Simple decay function
            
        # For each UAV, calculate average service quality to connected users
        for uav_idx in range(M):
            uav_node = 2*N + uav_idx
            # Find connected users
            edge_mask = edge_index[0] == uav_node
            connected_users = edge_index[1][edge_mask]
            if len(connected_users) > 0:
                # Average of connected users' target values
                targets[uav_node] = torch.mean(targets[connected_users])
            else:
                targets[uav_node] = 0.0  # No connected users
        
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Features must be a tensor, got {type(features)}")
            
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"Edge index must be a tensor, got {type(edge_index)}")
            
        if features.dim() != 2 or features.shape[1] != 2:
            raise ValueError(f"Invalid features shape: {features.shape}, expected [*, 2]")
            
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape: {edge_index.shape}, expected [2, *]")
            
        return features, edge_index, targets
        
    except Exception as e:
        raise ValueError(f"Error creating network data: {str(e)}")


def create_federated_data(N, M, device='cpu'):
    """
    Creates and partitions data for federated learning with UAVs.
    
    Args:
        N: Number of user pairs
        M: Number of UAVs (federated learning clients)
        device: Device to place tensors on (default: 'cpu')
    
    Returns:
        list: List of M tuples (features, edge_index, targets), one for each UAV
              Each UAV gets all user positions but only its own UAV position
              and corresponding target values
    
    Raises:
        ValueError: If N or M are invalid, or if tensor creation fails
        TypeError: If tensor types are incorrect
    """
    if N < 1 or M < 1:
        raise ValueError(f"Invalid network size: N={N}, M={M}")
        
    # Get full network data
    try:
        features, edge_index, targets = data_create(N, M, device)
    except Exception as e:
        raise ValueError(f"Failed to create network data: {str(e)}")
    
    # Split data for each UAV
    uav_data = []
    user_count = 2 * N  # Total number of users (N pairs * 2)
    
    for uav_idx in range(M):
        # Get user features (all users for each UAV)
        user_features = features[:user_count].clone()  # Clone to avoid sharing memory
        
        # Get specific UAV features (only this UAV's position)
        uav_feature = features[user_count + uav_idx].unsqueeze(0)
        
        # Ensure feature dimensions are correct [N, 2]
        if user_features.dim() != 2 or user_features.shape[1] != 2:
            raise ValueError(f"Invalid user feature shape: {user_features.shape}")
            
        if uav_feature.dim() != 2 or uav_feature.shape[1] != 2:
            raise ValueError(f"Invalid UAV feature shape: {uav_feature.shape}")
        
        # Combine features for this UAV's view
        uav_features = torch.cat([user_features, uav_feature], dim=0)
        
        # Create bidirectional edges connecting UAV to all users
        try:
            # Create edges from UAV to users
            src_indices_out = torch.full((user_count,), user_count, device=device)
            dst_indices_out = torch.arange(user_count, device=device)
            
            # Create edges from users to UAV
            src_indices_in = torch.arange(user_count, device=device)
            dst_indices_in = torch.full((user_count,), user_count, device=device)
            
            # Combine both directions
            src_indices = torch.cat([src_indices_out, src_indices_in])
            dst_indices = torch.cat([dst_indices_out, dst_indices_in])
            
            # Stack into edge tensor [2, 2*user_count]
            uav_edges = torch.stack([src_indices, dst_indices], dim=0)
            
            if not isinstance(uav_edges, torch.Tensor):
                raise TypeError(f"Failed to create edge tensor, got {type(uav_edges)}")
                
            if uav_edges.dim() != 2 or uav_edges.shape[0] != 2:
                raise ValueError(f"Invalid edge tensor shape: {uav_edges.shape}")
                
            # Ensure feature tensor has correct shape [user_count + 1, 2]
            if uav_features.dim() != 2 or uav_features.shape[1] != 2:
                raise ValueError(f"Invalid feature tensor shape: {uav_features.shape}")
                
            # Get corresponding target values
            user_targets = targets[:user_count].clone()
            uav_target = targets[user_count + uav_idx].unsqueeze(0)
            uav_targets = torch.cat([user_targets, uav_target])
            
            uav_data.append((uav_features, uav_edges, uav_targets))
            
        except Exception as e:
            raise ValueError(f"Error creating data for UAV {uav_idx}: {str(e)}")
    
    return uav_data
