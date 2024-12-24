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
from gnn_fed_config_new import get_default_args

# Get default configuration
parser = get_default_args()
args = parser.parse_args()
from sklearn.cluster import KMeans


def data_create(N, M):
    '''
    input:
         N:用户对数
         M:无人机个数
    
    Output：
         features：用户和无人机的位置，前2N个为用户坐标，后M个为无人机坐标
         edge_index:边的索引
    
    用户位置随机生成
    无人机位置为2N个用户的KMeans聚类
    '''
    # torch.manual_seed(50)
    # random.seed(50)
    # 创建了四个张量，每个张量的形状为(N, 1)，其中N是张量中的元素个数。
    x1 = torch.rand(N, 1)
    x2 = torch.rand(N, 1)
    y1 = torch.rand(N, 1)
    y2 = torch.rand(N, 1)

    idx = list(range(N))
    random.shuffle(idx)     # 创建列表，并对idx中的元素进行随机打乱
    x1 = x1[idx]
    random.shuffle(idx)
    x2 = x2[idx]            # 根据打乱后的索引idx重新排序张量x1和x2。

    user_src = torch.column_stack([x1, y1])
    user_dst = torch.column_stack([x2, y2])         # 创建了user_src和user_dst两个张量。
    users = torch.row_stack([user_src, user_dst])   # 使用torch.row_stack()函数，将user_src和user_dst垂直堆叠起来，创建了users张量。
    # users = torch.cat([user_src, user_dst], dim=0)  # 使用 torch.cat() 函数进行垂直堆叠

    kmeans = KMeans(n_clusters=M, random_state=0).fit(users.numpy())    # 对存储在users中的数据进行K-means聚类
    uav = torch.FloatTensor(kmeans.cluster_centers_)                    # 聚类中心的坐标存储在uav中，经过将其转换为PyTorch张量。

    # 使用torch.repeat_interleave()函数将位于len(users)到len(users)+len(uav)范围内的索引范围，重复len(users)次。
    # 生成的张量进行形状调整为(1, -1)，然后通过挤压操作去除额外的维度，得到index_src。
    # index_src = torch.repeat_interleave(torch.arange(len(users), len(users)+len(uav)).unsqueeze(1), repeats=len(users)+len(uav), dim=1)
    index_src = torch.repeat_interleave(torch.arange(len(users), len(users)+len(uav)).unsqueeze(1), repeats=len(users), dim=1)
    index_src = torch.reshape(index_src, (1, -1)).squeeze()

    # 使用torch.repeat()函数将位于0到len(users)范围内的索引范围重复len(uav)次，生成index_dst。
    # index_dst = torch.arange(0, len(users)+len(uav)).repeat(len(uav))
    index_dst = torch.arange(0, len(users)).repeat(len(uav))

    # 使用torch.row_stack()函数将user_src、user_dst和uav三个张量垂直堆叠，创建了features张量。args.device = torch.device('cpu')是新增的代码
    args.device = torch.device('cpu')
    features = torch.row_stack([user_src, user_dst, uav]).to(args.device)
    edge_index = torch.row_stack([index_src, index_dst]).to(args.device)


    return features, edge_index


def create_federated_data(N, M):
    """
    Creates and partitions data for federated learning with UAVs.
    
    Args:
        N: Number of user pairs
        M: Number of UAVs (federated learning clients)
    
    Returns:
        list: List of M tuples (features, edge_index), one for each UAV
              Each UAV gets all user positions but only its own UAV position
    """
    # Get full network data
    features, edge_index = data_create(N, M)
    
    # Split data for each UAV
    uav_data = []
    user_count = 2 * N  # Total number of users (N pairs * 2)
    
    for uav_idx in range(M):
        # Get user features (all users for each UAV)
        user_features = features[:user_count]
        
        # Get specific UAV features (only this UAV's position)
        uav_feature = features[user_count + uav_idx].unsqueeze(0)
        
        # Combine features for this UAV's view
        uav_features = torch.cat([user_features, uav_feature], dim=0)
        
        # Adjust edge indices for this UAV
        # Only keep edges connected to this UAV and users
        uav_edge_mask = (edge_index[0] == (user_count + uav_idx))
        uav_edges = edge_index[:, uav_edge_mask]
        
        # Adjust indices to account for only one UAV
        uav_edges[0, :] = user_count  # All source indices point to the single UAV
        
        uav_data.append((uav_features, uav_edges))
    
    return uav_data
