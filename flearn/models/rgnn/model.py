import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch import Tensor
from flearn.models.rgnn.configs_new import default_config as args
import numpy as np
import random
from tqdm import tqdm

'''
    UAV_conv: 一层LGNN
    PathSearch: 基于BF算法的链路搜索
    BFLoss: 基于BF算法的Loss计算
    RGNNLoss: 基于RGNN的Loss计算
    MLP: 基于MLP的位置优化
    UAV_Evolution: 基于遗传算法的位置优化
'''

class UAV_conv(MessagePassing):
    '''
    mlp进行消息传递和聚合
    lstm实现需要通信用户间的交互
    lstm实现uav间的交互
    '''
    def __init__(self, hidden_dim, alpha):    # hidden_dim：隐藏层的维度，控制模型的学习能力。alpha：用于LeakyReLU激活函数中的负斜率

        super(UAV_conv, self).__init__(aggr='mean',flow='target_to_source')    # 两个参数aggr和flow
        # aggr='mean'表示节点特征的聚合方式为均值池化，flow='target_to_source'表示消息传递方向为从目标节点到源节点。
        
        self.hidden_dim = hidden_dim
        # self.weight = nn.Parameter(torch.rand(size=(edge_num, 1)))  # 
        self.linear = nn.Linear(2, hidden_dim)      # 创建一个线性层，将输入的维度是2，输出的维度是hidden_dim
        # Quality prediction handled by fc layer
        self.att = nn.Sequential(                   # 创建一个序列模块，包含一个线性层和一个LeakyReLU激活函数，用于计算注意力权重
            nn.Linear(2*hidden_dim, 1),
            nn.LeakyReLU(alpha)
        )
        

        # uav初始embedding
        self.uav_linear = nn.Linear(2, hidden_dim)  # 创建一个线性层，将输入的维度是2，输出的维度是hidden_dim。用于初始化UAV的嵌入

        # uav间通信过程. 输入维度是hidden_dim，隐藏层维度是hidden_dim/2，双向LSTM。
        self.uav_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True)
        # 需要通信的用户间的信息传递,编码. 输入维度是2，隐藏层维度是hidden_dim/2，双向LSTM。
        self.users_lstm = nn.LSTM(input_size=2, hidden_size=int(hidden_dim/2), bidirectional=True)
        # 创建一个序列模块，包含多个线性层和激活函数，用于进行全连接层的计算
        self.fc = nn.Sequential(
                    nn.Linear(2*hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 20)
        )
        # 创建一个序列模块，包含多个线性层和激活函数，用于计算信息传递的多层感知机
        self.message_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 创建一个序列模块，包含多个线性层和激活函数，用于更新节点状态的多层感知机
        self.update_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # self.agg_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True)
        self.Wq = nn.Linear(hidden_dim, hidden_dim)     # 计算更新的门控wq
        self.Wr = nn.Linear(hidden_dim, hidden_dim)

    # 一个神经网络模型的前向传播函数
    def forward(self, x, edge_index):
        """Forward pass with dynamic observation features and relay decisions."""
        # Store original edge index for relay decisions
        self.source_index = list(range(edge_index[0][0], edge_index[0][-1]+1))
        self.users_num = int(edge_index[0][0]/2)

        # Extract environmental features using LSTM encoding
        user_pairs = torch.column_stack([x[:self.users_num, :].unsqueeze(1), x[self.users_num:2*self.users_num, :].unsqueeze(1)])
        user_pairs = user_pairs.transpose(0, 1)
        users, (user_hidden, _) = self.users_lstm(user_pairs)
        users = users.transpose(0, 1)
        users = torch.row_stack([users[:, 0, :], users[:, 1, :]])

        # Enhanced UAV feature extraction with environmental context
        uavs = self.uav_linear(x[2*self.users_num:, :])
        
        # Store environmental observations
        self.environmental_features = uavs.clone()
        
        # Combine user and UAV features
        x = torch.row_stack([users, uavs])

        # Message passing with attention for relay decisions
        node_features = self.propagate(edge_index, x=x)
        
        # Generate relay decisions using attention weights
        relay_decisions = self.get_relay_decisions(edge_index)
        
        # Generate hidden and cell states for LSTM
        h = torch.zeros(1, node_features.size(0), self.hidden_dim, device=node_features.device)
        c = torch.zeros(1, node_features.size(0), self.hidden_dim, device=node_features.device)
        
        # Store relay decisions in the hidden state
        if relay_decisions is not None:
            h[0, 2*self.users_num:, :] = relay_decisions.float()
        
        # Generate probability distribution for next node selection
        prob_dist = F.softmax(self.fc(node_features), dim=1)
        
        # Return tuple format expected by FedUAVGNN
        return prob_dist, h, c, node_features

    def message(self, x_i, x_j) -> torch.Tensor:
        # 消息传播机制

        # message_mlp计算用户到uav的信息传递
        # 将接收到的节点特征x_i和发送的节点特征x_j在列维度上堆叠在一起，然后通过self.message_mlp模型计算消息传递的输出self.outputs。
        self.outputs = self.message_mlp(torch.column_stack([x_i, x_j]))

        # 计算注意力,将x_i和self.outputs分别输入到self.Wq和self.Wr模型中，并将它们在列维度上拼接在一起，计算注意力权重self.att_weight。
        self.att_weight = self.att(torch.cat([self.Wq(x_i), self.Wr(self.outputs)], dim=1))
        # self.att_weight = self.att(torch.cat([x_i, self.direction_f], dim=1))
        self.att_weight = self.att_weight.reshape(len(self.source_index), -1)
        self.att_weight = F.softmax(self.att_weight, dim=1).reshape(-1, 1)
        # 将注意力权重进行形状变换，并使用softmax函数进行归一化处理，得到最终的注意力权重。
        return self.outputs

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        # 消息聚合
        # 将接收到的节点特征inputs与注意力权重self.att_weight相乘。然后通过调用super().aggregate方法对这些特征进行聚合操作，得到聚合后的输出outputs。
        inputs = self.att_weight * inputs
        outputs = super().aggregate(inputs, index, ptr, dim_size)
        return outputs
    

    def update(self, aggr_out, x):
        """Update node features with environmental context."""
        # Combine aggregated and current features
        x = self.update_mlp(torch.column_stack([aggr_out, x]))
        
        # Enhanced UAV state update with environmental context
        uav_embeddings = x[2*self.users_num:, :].unsqueeze(1)
        uav_states, (_, _) = self.uav_lstm(uav_embeddings)
        uav_states = uav_states.squeeze(1)
        
        # Update UAV positions with environmental awareness
        x = torch.row_stack([x[:2*self.users_num, :], uav_states])
        return x
        
    def get_relay_decisions(self, edge_index):
        """Generate relay decisions based on attention weights and environmental features."""
        if not hasattr(self, 'att_weight'):
            return None
            
        # Get UAV indices
        uav_start_idx = 2 * self.users_num
        uav_indices = torch.arange(uav_start_idx, edge_index.max() + 1)
        
        # Extract UAV-to-UAV attention weights
        uav_mask = (edge_index[0] >= uav_start_idx) & (edge_index[1] >= uav_start_idx)
        uav_edges = edge_index[:, uav_mask]
        uav_attention = self.att_weight[uav_mask]
        
        # Create relay decision matrix
        num_uavs = len(uav_indices)
        relay_matrix = torch.zeros((num_uavs, num_uavs), device=edge_index.device)
        
        # Fill matrix with attention weights
        for i, (src, dst) in enumerate(uav_edges.t()):
            src_idx = src.item() - uav_start_idx
            dst_idx = dst.item() - uav_start_idx
            relay_matrix[src_idx, dst_idx] = uav_attention[i]
            
        # Get best relay for each UAV based on attention weights
        relay_decisions = torch.argmax(relay_matrix, dim=1)
        
        return relay_decisions
        # print(x.shape)
        # exit()

        # 映射到01之间
        x = self.fc(x)    # Generate quality predictions using fc layer (maps to [0,1] interval via sigmoid)

        # Ensure output has shape [20]
        if x.dim() == 2:
            if x.size(1) == 20:  # If already in correct shape
                x = x.squeeze(0)
            else:
                x = x.view(-1)[:20]  # Reshape and take first 20 elements
        else:
            x = x[:20]  # Take first 20 elements if 1D
            
        # Pad if necessary
        if x.size(0) < 20:
            x = F.pad(x, (0, 20 - x.size(0)))
            
        return x  # Return exactly [20] shaped tensor


class UAV(nn.Module):
    '''一层的卷积'''
    def __init__(self, hid_dim, alpha):
        super(UAV, self).__init__()
        self.uav_conv1 = UAV_conv(hid_dim, alpha)
        # self.uav_conv2 = UAV_conv(hid_dim, alpha)

    def forward(self, features, edge_index):
        # return self.uav_conv2(self.uav_conv1(features, edge_index), edge_index)
        return self.uav_conv1(features, edge_index)


class PathSearch:
    '''
        DFS搜索每对用户间的所有路径，返回snr
    '''
    def __init__(self, N, M) -> None:   # 构造函数__init__接受两个参数N和M，分别表示用户的数量和无人机的数量。
        # 计算无人机和用户+无人机间的距离
        self.N = N
        self.M = M                      # 这些参数被存储在类的实例变量self.N和self.M中。
    
    def search_maps(self, locations):   # 接受一个名为locations的参数
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]
        dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵（计算用户和无人机之间欧氏距离，将结果保存在dist_mat中）
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []  # 创建了一个空列表map_list，用于存储每对用户之间的图
        for i in range(self.N):
            uav_map = dist_mat[:, 2*self.N:]
            uav_user = dist_mat[:, [i, i+self.N]]
            map_i = np.concatenate([uav_user, uav_map], axis=1)
            users_map = uav_user.T
            users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
            map_i = np.concatenate([users_map, map_i], axis=0)
            # 计算出用户和无人机之间的距离，并将这些距离作为权值构建图，并存储在map_list中
            self.map_list.append(map_i)
        
        '''对所有图进行搜索'''
        best_path_snr = []     # 对于每对用户最佳路径所对应的snr（信噪比）的倒数
        for map_i in self.map_list:
            self.map_now = map_i
            self.set = np.zeros(self.M+2)
            self.top = -1
            self.stack = []
            self.path_max_edge = []     # 搜索到的路径的最长边的距离
            # self.path_list = []
            self.DFS(0)                 # DFS搜索算法
            best_path_snr.append(min(self.path_max_edge))
            # best_path_list.append(self.path_list[np.argmin(self.path_max_edge)])    # 最佳路径列表
        return np.mean(best_path_snr)   # 该函数根据用户和无人机的位置信息，搜索出用户之间的最佳路径，并返回平均SNR损失。

    def search_maps_BF(self, locations):
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]                           # 通过将输入列表中的2N以后的元素提取出来，即为无人机的位置信息。
        dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []
        for i in range(self.N):
            uav_map = dist_mat[:, 2*self.N:]
            uav_user = dist_mat[:, [i, i+self.N]]
            map_i = np.concatenate([uav_user, uav_map], axis=1)
            users_map = uav_user.T
            users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
            map_i = np.concatenate([users_map, map_i], axis=0)
            self.map_list.append(map_i)
        
        # 构建边集，创建边的起点和终点，其中起点为0，终点为每个无人机的位置。
        # 通过使用这两个数组，可以构建一张完整的图，其中边的起点和终点对应了用户和无人机的位置。这样，每个边就代表了用户和无人机之间的连接，便于后续的图搜索算法进行路径计算和信噪比评估

        edge_src = np.array([0 for _ in range(self.M)])     # edge_src数组定义了边的起点，其中0表示源节点（用户），而2到M+1表示无人机的位置
        edge_dst = np.arange(2, self.M+2)                   # edge_dst数组定义了边的终点，其中2到M+1表示无人机的位置，1表示目标节点（用户）
        edge_src = np.concatenate([edge_src, np.arange(2, self.M+2), np.repeat(np.arange(2, self.M+2), repeats=self.M, axis=0)], axis=0)
        edge_dst = np.concatenate([edge_dst, np.full_like(edge_dst, 1)], axis=0)
        uav_dst = np.repeat(np.expand_dims(np.arange(2, self.M+2), axis=0), self.M, axis=0).reshape(-1, 1).squeeze(1)
        edge_dst = np.concatenate([edge_dst, uav_dst], axis=0)

        '''对所有图进行搜索'''
        # 代码的作用是计算最佳路径和最佳信噪比
        best_path_list = []
        best_snr_list = []
        for map_i in self.map_list:
            # print(map_i)
            # 对每个节点创建max_edge表以及对应的链路
            # map_i为边权重
            max_edge = [np.inf for _ in range(self.M+2)]    # 创建一个长度为 self.M+2 的列表，初始值都为正无穷，这个列表用于存储从起点到每个节点的最大边权重
            max_edge[0] = 0                                 # 将起点的到自身的最大边权重设置为 0
            max_edge_path = [[] for _ in range(self.M+2)]   # 创建一个长度为 self.M+2 的列表，初始值都为空列表，用于存储从起点到每个节点的最佳路径
            max_edge_path[0] = [0]                          # 将起点的最佳路径初始化为只包含起点的列表
            # print(max_edge_path)
            for _ in range(self.M+2):                       # 外层 for 循环：循环执行 self.M+2 次（包括起点和终点），用于进行最短路径计算
                for i, j in zip(edge_src, edge_dst):        # 内层 for 循环：遍历边的起点和终点的数组 edge_src 和 edge_dst
                    # 松弛每条边
                    # 判断当前边是否可以松弛。条件是边的起点和终点不相等，并且从起点到终点的边权重和当前记录的最大边权重的较大值小于当前终点的最大边权重
                    if i != j and np.max([max_edge[i], map_i[i][j]]) < max_edge[j]:
                        max_edge[j] = np.max([max_edge[i], map_i[i][j]])    # 更新终点的最大边权重为较大值
                        max_edge_path[j] = max_edge_path[i].copy()          # 更新终点的最佳路径为起点的最佳路径的复制
                        max_edge_path[j].append(j)                          # 将终点添加到最佳路径的末尾
                        
            best_snr_list.append(max_edge[1])                               # 将终点的最大边权重添加到最佳信噪比列表中
            best_path_list.append(np.array(max_edge_path[1]))               # 将终点的最佳路径添加到最佳路径列表中。
        return np.mean(best_snr_list), np.array(best_path_list)             # 计算最佳信噪比的均值，并将最佳路径列表转换为 NumPy 数组作为返回值
        # np.mean() 是 NumPy 库中的一个函数，用于计算数组或指定轴上的均值。
    
    def search_maps_(self, locations):
        """

        input: 用户和无人机的位置
        output: 返回N对用户的平均loss
        
        """
        uav_location = locations[2*self.N:]
        self.dist_mat = self.EuclideanDistances(uav_location, locations)   # 距离矩阵
        # 针对每对用户创建一张图，权值为相互间的距离
        self.map_list = []
        # for i in range(self.N):
        uav_map = self.dist_mat[:, 2*self.N:]
        uav_user = self.dist_mat[:, [0, self.N]]
        map_i = np.concatenate([uav_user, uav_map], axis=1)
        users_map = uav_user.T
        users_map = np.concatenate([np.zeros((2, 2)), users_map], axis=1)
        map_i = np.concatenate([users_map, map_i], axis=0)
        # self.map_list.append(map_i)
        
        '''对所有图进行搜索'''
        best_path_snr = []     # 对于每对用户最佳路径所对应的snr的倒数
        best_path_list = []
        # for idx, map_i in enumerate(self.map_list):
        for idx in range(self.N):
            self.map_now = map_i
            self.set = np.zeros(self.M+2)
            self.top = -1
            self.stack = []
            self.path_max_edge = []     # 搜索到的路径的最长边的距离
            # 第一张图搜索所有路径
            if idx == 0:
                self.edge_dist_mat = []     # N个列表，包含N条路径的边长
                self.path_list = [] # 所有的路径
                self.DFS(0)
            else:
                self.output_path_(idx)
            best_path_snr.append(min(self.path_max_edge))
            best_path_list.append(self.path_list[np.argmin(self.path_max_edge)])    # 最佳路径列表
            # print(best_path_list)
        return np.mean(best_path_snr), best_path_list

    def output_path(self):
        edge_dist = []                      # 创建一个空列表edge_dist，用于存储边的距离
        for i in range(len(self.stack)-1):
            # 将self.map_now中self.stack[i]和self.stack[i+1]位置对应的元素加入到edge_dist列表中，表示这两个节点之间的边的距离
            edge_dist.append(self.map_now[self.stack[i], self.stack[i+1]])
        self.path_list.append(np.array(self.stack.copy()))
        self.edge_dist_mat.append(edge_dist.copy())
        self.path_max_edge.append(max(edge_dist))
    # 整体功能是在搜索过程中，记录每个找到的路径及其对应的边的距离，并将它们添加到self.path_list、self.edge_dist_mat和self.path_max_edge列表中。
    # 这样做是为了后续计算最佳路径所对应的SNR和最长边的距离。


    def output_path_(self, idx):
        # print(self.edge_dist_mat)
        # 对于self.path_list中的每个索引k和对应的路径p，循环执行以下代码
        for k, p in enumerate(self.path_list):
            self.edge_dist_mat[k][0] = self.dist_mat[p[1]-2, idx]               # 更新路径起始节点的边的距离
            self.edge_dist_mat[k][-1] = self.dist_mat[p[-2]-2, idx + self.N]    # 更新路径结束节点的边的距离
            self.path_max_edge.append(max(self.edge_dist_mat[k]))               # 记录路径中最长边的距离

    def EuclideanDistances(self, a, b):     #定义计算欧几里得距离的函数
        sq_a = a**2
        sum_sq_a = np.expand_dims(np.sum(sq_a, axis=1), axis=1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = np.sum(sq_b, axis=1)[np.newaxis, :]  # n->[1, n]
        bt = b.T
        distance = np.sqrt(abs(sum_sq_a+sum_sq_b-2*np.matmul(a, bt)))
        return distance


class BFLoss(nn.Module):
    """
    Enhanced Bellman-Ford loss incorporating relay decisions and resource constraints
    """
    def __init__(self, N, M, bandwidth_weight=0.1, compute_weight=0.1, relay_weight=0.1):
        super(BFLoss, self).__init__()
        self.N = N
        self.M = M
        self.path_search = PathSearch(N, M)
        self.bandwidth_weight = bandwidth_weight
        self.compute_weight = compute_weight
        self.relay_weight = relay_weight
    
    def forward(self, locations, relay_decisions=None, bandwidth=None, compute_speed=None):
        """
        Compute loss incorporating path quality, relay efficiency, and resource constraints
        """
        # Base path quality loss using Bellman-Ford
        locations_ = locations.cpu().data.numpy()
        _, path_list = self.path_search.search_maps_BF(locations_)
        
        best_snr = []
        relay_load = torch.zeros(self.M, device=locations.device)  # Track relay load per UAV
        
        for i in range(self.N):
            path_i = np.array(path_list[i])
            path_i[0] = i
            path_i[-1] = i + self.N
            path_i[1:-1] += 2*(self.N-1)
            uav_path = path_i[1:-1]
            
            # Calculate path distances
            path_dist = [torch.norm(locations[path_i[0]]-locations[path_i[1]]), 
                        torch.norm(locations[path_i[-1]]-locations[path_i[-2]])]
            
            # Track relay load for UAVs in path
            if relay_decisions is not None:
                for uav_idx in uav_path: 
                    relay_load[uav_idx - 2*self.N] += 1
            
            for j in range(len(uav_path)-1):
                path_dist.append(torch.norm(locations[uav_path[j]]-locations[uav_path[j+1]]))
            path_dist = torch.row_stack(path_dist)
            best_snr.append(torch.max(path_dist))
            
        best_snr = torch.row_stack(best_snr)
        path_loss = torch.mean(best_snr)
        
        # Resource constraint penalties
        bandwidth_loss = torch.tensor(0.0, device=locations.device)
        compute_loss = torch.tensor(0.0, device=locations.device)
        relay_loss = torch.tensor(0.0, device=locations.device)
        
        if bandwidth is not None:
            bandwidth_loss = torch.mean(F.relu(relay_load - bandwidth))
            
        if compute_speed is not None:
            compute_loss = torch.mean(F.relu(relay_load/compute_speed))
            
        if relay_decisions is not None:
            # Penalize long relay chains
            chain_lengths = torch.zeros_like(relay_decisions, dtype=torch.float)
            for i, decision in enumerate(relay_decisions):
                current = i
                chain_length = 0
                visited = set()
                while current != decision and current not in visited:
                    visited.add(current)
                    current = relay_decisions[current].item()
                    chain_length += 1
                chain_lengths[i] = chain_length
            relay_loss = torch.mean(chain_lengths)
        
        # Combined loss
        total_loss = path_loss + \
                    self.bandwidth_weight * bandwidth_loss + \
                    self.compute_weight * compute_loss + \
                    self.relay_weight * relay_loss
                    
        return total_loss

            

    def EuclideanDistances(self, a, b):     # 用于计算输入矩阵a和b中每对向量之间的欧氏距离,
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b, axis=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        distance = torch.sqrt(torch.abs(sum_sq_a+sum_sq_b-2*torch.mm(a, bt)))
        return distance



class RGNNLoss(nn.Module):
    def __init__(self, path_find_model, N, bandwidth_weight=0.1, compute_weight=0.1, feature_weight=0.1):
        super(RGNNLoss, self).__init__()
        self.path_find_model = path_find_model
        self.N = N  # Number of user pairs
        self.bandwidth_weight = bandwidth_weight
        self.compute_weight = compute_weight
        self.feature_weight = feature_weight
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, bandwidth=None, compute_speed=None, environmental_features=None):
        """
        Enhanced forward pass incorporating resource constraints and feature quality
        
        Args:
            outputs: Model outputs including node features and positions
            bandwidth: Available bandwidth per UAV
            compute_speed: Computation speed per UAV
            environmental_features: Ground truth environmental features if available
        """
        if isinstance(outputs, dict):
            node_features = outputs['node_features']
            relay_decisions = outputs.get('relay_decisions', None)
            pred_env_features = outputs.get('environmental_features', None)
        else:
            node_features = outputs
            relay_decisions = None
            pred_env_features = None
            
        users_src = node_features[:self.N].unsqueeze(1)
        users_dst = node_features[self.N:2*self.N].unsqueeze(1)
        uav_nodes = node_features[2*self.N:].repeat(self.N, 1, 1)
        
        uav_graph = torch.cat([users_src, uav_nodes, users_dst], dim=1)
        B = uav_graph.shape[0]
        size = uav_graph.shape[1]
        
        # Initialize path finding
        mask = torch.zeros(uav_graph.shape[0], uav_graph.shape[1]).to(node_features.device)
        mask[:, 0] = -np.inf
        x = uav_graph[:,0,:]
        max_dist = torch.zeros(uav_graph.shape[0]).to(node_features.device)
        h = None
        c = None
        
        # Path finding with resource awareness
        for k in range(size):
            if k == 0:
                mask[[i for i in range(B)], -1] = -np.inf
                Y0 = x.clone()
            if k > 0:
                mask[[i for i in range(B)], -1] = 0
            
            model_output = self.path_find_model(x=x, X_all=uav_graph, h=h, c=c, mask=mask)
            if isinstance(model_output, dict):
                output = model_output['attention_weights']
                h = model_output['h']
                c = model_output['c']
            else:
                output, h, c, _ = model_output
            output = output.detach()
            
            idx = torch.argmax(output, dim=1)
            Y1 = uav_graph[[i for i in range(B)], idx.data].clone()
            
            dist = torch.norm(Y1-Y0, dim=1)
            max_dist[dist > max_dist] = dist[dist > max_dist]
            
            Y0 = Y1.clone()
            x = uav_graph[[i for i in range(B)], idx.data].clone()
            
            mask[[i for i in range(B)], idx.data] += -np.inf
            mask[idx.data==size] = -np.inf
        
        path_loss = max_dist.mean()
        
        # Resource constraint penalties
        bandwidth_loss = torch.tensor(0.0, device=node_features.device)
        compute_loss = torch.tensor(0.0, device=node_features.device)
        feature_loss = torch.tensor(0.0, device=node_features.device)
        
        if bandwidth is not None and relay_decisions is not None:
            relay_load = torch.zeros(len(relay_decisions), device=node_features.device)
            for i, decision in enumerate(relay_decisions):
                if decision != i:  # If UAV is relaying
                    relay_load[decision] += 1
            bandwidth_loss = torch.mean(F.relu(relay_load - bandwidth))
            
        if compute_speed is not None and relay_decisions is not None:
            compute_loss = torch.mean(F.relu(1.0/compute_speed))
            
        if environmental_features is not None and pred_env_features is not None:
            feature_loss = self.mse(pred_env_features, environmental_features)
        
        # Combined loss
        total_loss = path_loss + \
                    self.bandwidth_weight * bandwidth_loss + \
                    self.compute_weight * compute_loss + \
                    self.feature_weight * feature_loss
                    
        return total_loss
# 该模型的功能是根据输入的outputs进行路径搜索，并计算路径的最大距离的平均值作为损失。模型中包含一个path_find_model，其余部分为路径搜索算法的实现。

# 该MLP模型用于接收输入特征，其中包含有关用户的信息。它将输入特征经过编码，并利用MLP模型进行位置确定。最终输出无人机的位置坐标。
class MLP(nn.Module):
    '''利用mlp进行位置确定'''
    def __init__(self, input_dim, hidden_dim, output_num, users_num):  # 输入参数
        super(MLP, self).__init__()
        self.input_dim = input_dim      # 将输入特征的维度保存为类的成员变量
        self.hidden_dim = hidden_dim
        self.users_num = users_num
        # 对节点进行编码
        self.linear = nn.Linear(input_dim, hidden_dim)      # 定义一个线性层(nn.Linear)，用于对输入特征进行编码。
        # 两层的mlp
        self.location_mlp = nn.Sequential(                  # 定义位置确定的MLP模型，通过nn.Sequential将多个层串联起来。其中包括
            nn.Linear(2*users_num*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_num*2),
            nn.ReLU(),
            nn.Tanh()                                       # Tanh激活函数(nn.Tanh)，用于将输出结果的范围限定在[-1, 1]之间
        )

    def forward(self, features):
        users = features[:2*self.users_num, :]
        users_cat = users.reshape(1, -1)
        uav_locations = self.location_mlp(users_cat)        # 通过位置MLP网络对用户信息进行编码，得到无人机位置。
        uav_locations = uav_locations.reshape(-1, 2)        # 将无人机位置进行重新形状，每2列为一个位置坐标。

        return uav_locations

class UAV_Evolution():
    def __init__(self, N, M, pop_num, total_epochs, features, Loss) -> None:
        self.flow_loss = Loss
        self.N = N
        self.M = M
        self.features = features
        self.pop_num = pop_num     # 初始种群规模
        self.retain_rate = 0.3      # 保存率
        self.mutate_rate = 0.2
        self.random_select_rate = 0.1
        # self.locations = locations
        self.total_epoch = total_epochs
        self.b = 2

    def evolution(self):
        populication = self.populication_create()   # 调用 函数声明在下面
        max_distance = []                           # 创建一个空列表max_distance，用于存储每个迭代周期中最好个体的适应度
        for i in tqdm(range(self.total_epoch)):
            # print(populication.shape)
            parents, output_i = self.selection(populication)      # 调用 函数声明在下面
            cs = self.cross_over(parents)                         # 调用 对父代个体进行交叉操作，生成一部分子代个体，并将结果赋值给变量cs
            cs = self.mutation(cs, i)                             # 调用 对子代个体进行变异操作，生成新的子代个体，并将结果赋值给变量cs
            populication = np.concatenate([parents, cs], axis=0)  # 将父代个体和子代个体按行连接起来，生成新的种群，并将结果赋值给变量populication。
            # output_i = self.adaptbility(populication)
            
            max_dist = np.min(output_i)       # 最好的一个个体对应的最大边
            
            max_distance.append(max_dist)     # 将最好个体的适应度max_dist添加到max_distance列表中。
            # print('epoch == {}，max_distance of best_one == {}'.format(i, max_dist))
        
        return np.array(max_distance)         # 将 max_distance转换为NumPy数组，并作为函数的返回值
        
    # 该方法用于进行交叉操作，通过选择、交叉和组合父代个体，生成子代个体
    def cross_over(self, parent):
        # 交叉, 单点交叉

        # 均匀交叉
        children = []
        get_child_num = self.pop_num-len(parent)
        while len(children) < get_child_num:
            i = random.randint(0, len(parent)-1)
            j = random.randint(0, len(parent)-1)
            male = parent[i]
            female = parent[j]
            select_p = np.random.rand(len(male))
            select_p[np.where(select_p < 0.5)] = 0
            select_p[np.where(select_p >= 0.5)] = 1
            child1 = select_p * male + (1-select_p) * female
            child2 = (1 - select_p) * male + select_p * female
            children.append(child1.reshape(1, len(child1)))
            children.append(child2.reshape(1, len(child2)))
            
        children = np.concatenate(children, axis=0)
        return children

    def populication_create(self):
        # 生成种群
        self.populication = np.random.rand(self.pop_num, self.M*2)
        self.users = torch.tensor(self.features[:2*self.N], device=args.device)
        
        return self.populication

    def mutation(self, cs, i):
        # 变异
        
        # 采用非一致性变异，每个位置都进行变异
        new_cs = cs.copy()
        for idx, c in enumerate(cs):
            if random.random() < self.mutate_rate:
                r = random.random()
                mut1 = (1-c)*np.random.rand(len(c))*(1-i/self.total_epoch)**self.b
                mut2 = c*np.random.rand(len(c))*(1-i/self.total_epoch)**self.b
                # print(mut1)
                if random.random() > 0.5:
                    c = c + mut1
                else:
                    c = c - mut2
                # print(c)
            new_cs[idx] = c
            # print(c)
        return new_cs
            

    def selection(self, populication):
        # 选择

        # 选择最佳的rate率的个体
        # 对种群从小到大进行排序
        adpt = self.adaptbility(populication)
        grabed = [[ad, one] for ad, one in zip(adpt, populication)]
        # print(grabed)
        # exit()
        sorted_grabed = sorted(grabed, key=lambda x: x[0])
        grabed = np.array([x[1] for x in sorted_grabed])
        index = int(len(populication)*self.retain_rate)

        live = grabed[:index]

        # 选择幸运个体
        for i in grabed[index:]:
            if random.random() < self.random_select_rate:
                live = np.concatenate([live, i.reshape(1, len(i))], axis=0)
        
        return live, adpt
    
    def adaptbility(self, populication):
        max_dist = []
        for p in torch.FloatTensor(populication).to(args.device):
            p = p.reshape(int(len(p)/2), 2)
            # p = torch.FloatTensor(p).to(device)
            users_uav = torch.cat([self.users, p], dim=0)
            max_dist.append(self.flow_loss(users_uav).cpu().data.numpy())
        return max_dist

