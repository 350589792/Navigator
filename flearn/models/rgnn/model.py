import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch import Tensor
from flearn.models.rgnn.configs_new import get_default_args

# Get default configuration
parser = get_default_args()
args = parser.parse_args()
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
        self.linear2 = nn.Linear(hidden_dim, 2)     # 创建另一个线性层，将输入的维度是hidden_dim，输出的维度是2
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
                    nn.Linear(hidden_dim, 2),
                    nn.Sigmoid()
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
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # 源节点和目标节点的索引
        self.source_index = list(range(edge_index[0][0], edge_index[0][-1]+1))  # 存储源节点索引的列表
        self.users_num = int(edge_index[0][0]/2)                                # 用户节点的数量

        # 用户间lstm实现编码,即对用户节点进行LSTM编码
        # 从输入特征x中选取前self.users_num个节点作为用户节点特征，再讲这些特征按列堆叠在一起，得到一个形状为(self.users_num, 2, feature_dim)的张量
        user_pairs = torch.column_stack([x[:self.users_num, :].unsqueeze(1), x[self.users_num:2*self.users_num, :].unsqueeze(1)])
        user_pairs = user_pairs.transpose(0, 1)
        users, (_, _) = self.users_lstm(user_pairs)
        users = users.transpose(0, 1)
        users = torch.row_stack([users[:, 0, :], users[:, 1, :]])

        # uav间编码， 对UAV节点进行线性变换，得到UAV节点的表示uavs。
        uavs = self.uav_linear(x[2*self.users_num:, :])
        # 将用户节点和UAV节点的表示按行堆叠在一起，得到最终的节点表示x。
        x = torch.row_stack([users, uavs])

        # 使用self.propagate方法进行消息传播。
        return self.propagate(edge_index, x=x)

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
        # 对节点进行更新的过程
        # 将聚合后的输出aggr_out与原始输入特征x在列维度上拼接起来，并通过self.update_mlp模型进行更新，得到更新后的节点表示x
        x = self.update_mlp(torch.column_stack([aggr_out, x]))
        
        # 聚合之后uav间进行lstm信息传递
        uav_embeddings = x[2*self.users_num:, :].unsqueeze(1)
        uav_locations, (_ ,_) = self.uav_lstm(uav_embeddings)
        uav_locations = uav_locations.squeeze(1)
        # uav_locations = self.fc(uav_.squeeze(1))
        # x = F.sigmoid(self.linear2(x))
        # x[2*self.users_num:, :] = uav_locations
        x = torch.row_stack([x[:2*self.users_num, :], uav_locations])    # 将更新后的节点表示与UAV节点的表示按行堆叠在一起，得到最终的节点表示x
        # print(x.shape)
        # exit()

        # 映射到01之间
        x = F.sigmoid(self.linear2(x))    # 通过Sigmoid函数进行映射，将节点表示映射到[0,1]区间，并返回更新后的节点表示x

        return x


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
        使用Bellman-Ford计算最佳链路
        
    """
    def __init__(self, N, M):
        super(BFLoss, self).__init__()
        self.N = N
        self.M = M
        self.path_search = PathSearch(N, M)         # 调用了PathSearch类来初始化path_search对象。
    
    def forward(self, locations):                   # forward方法是类的前向传播函数，用于计算最佳链路
        locations_ = locations.cpu().data.numpy()   # 将locations转换为NumPy数组，以便后续处理
        _, path_list= self.path_search.search_maps_BF(locations_)
        # 调用了PathSearch类的search_maps_BF方法进行路径搜索，并将搜索结果存储在path_list中
        
        best_snr = []
        for i in range(self.N):
            # 将path_i设置为path_list[i]的NumPy数组,修改path_i的首尾元素，即将起始节点和结束节点替换为特定的值
            path_i = np.array(path_list[i])
            path_i[0] = i
            path_i[-1] = i+ self.N
            path_i[1:-1] += 2*(self.N-1)        # 对path_i[1:-1]中的元素进行偏移操作，以应对特定的链路需求
            # path_i[1:-1] -= 2
            uav_path = path_i[1:-1]     # uav间的链路
            # 构建path_dist列表，其中包含了起始节点到第二个节点、结束节点到倒数第二个节点的距离
            path_dist = [torch.norm(locations[path_i[0]]-locations[path_i[1]]), torch.norm(locations[path_i[-1]]-locations[path_i[-2]])]
            # 迭代range(len(uav_path)-1)，计算uav_path中相邻节点之间的距离，并将结果添加到path_dist列表中
            for j in range(len(uav_path)-1):
                path_dist.append(torch.norm(locations[uav_path[j]]-locations[uav_path[j+1]]))
            path_dist = torch.row_stack(path_dist)
            best_snr.append(torch.max(path_dist))       # 使用torch.max函数计算path_dist中的最大值，并添加到best_snr列表中。
        best_snr = torch.row_stack(best_snr)
        return torch.mean(best_snr)                     #使用torch.mean函数计算best_snr中的均值，并作为最终结果返回。

            

    def EuclideanDistances(self, a, b):     # 用于计算输入矩阵a和b中每对向量之间的欧氏距离,
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b, axis=1).unsqueeze(0)  # n->[1, n]
        bt = b.t()
        distance = torch.sqrt(torch.abs(sum_sq_a+sum_sq_b-2*torch.mm(a, bt)))
        return distance



class RGNNLoss(nn.Module):
    def __init__(self, path_find_model, N):
        super(RGNNLoss, self).__init__()
        self.path_find_model = path_find_model
        self.N = N  # 用户对数
    
    def forward(self, outputs):
        users_src = outputs[:self.N].unsqueeze(1)                           # 把前N个outputs切片成users_src，并在第二个维度添加了一个维度
        users_dst = outputs[self.N:2*self.N].unsqueeze(1)
        uav_nodes = outputs[2*self.N:].repeat(self.N, 1, 1)
        
        uav_graph = torch.cat([users_src, uav_nodes, users_dst], dim=1)     # 通过在第二个维度上拼接users_src、uav_nodes和users_dst而创建的
        B = uav_graph.shape[0]
        size = uav_graph.shape[1]
        mask = torch.zeros(uav_graph.shape[0], uav_graph.shape[1]).to(args.device)      # 初始化为零的张量，用于记录路径搜索中已经访问过的位置
        mask[:, 0] = -np.inf                                                # 初始时，将第一个节点设置为-np.inf，表示不可访问。
        x = uav_graph[:,0,:]                                                # x初始化为uav_graph中的第一个节点（源节点）
        max_dist = torch.zeros(uav_graph.shape[0]).to(args.device)          # 用于记录每个路径的最大距离的张量
        h = None
        c = None
        # RNN模型中的隐藏状态和细胞状态，初始时设置为None
        for k in range(size):
            if k == 0:                                      # 将mask张量的最后一个元素设置为-np.inf，即该位置不可访问。并且将当前节点的副本保存到Y0。
                mask[[i for i in range(B)], -1] = -np.inf
                Y0 = x.clone()
            if k > 0:                                       # 将mask张量的最后一个元素设置为0，即该位置可以访问。
                mask[[i for i in range(B)], -1] = 0
            
            output, h, c, _ = self.path_find_model(x=x, X_all=uav_graph, h=h, c=c, mask=mask)
            output = output.detach()
            
            idx = torch.argmax(output, dim=1)         # now the idx has B elements
            # idx_list.append(idx.clone().cpu().data.numpy()[0])
            Y1 = uav_graph[[i for i in range(B)], idx.data].clone()     # 根据idx从uav_graph中选择新的节点Y1
            
            dist = torch.norm(Y1-Y0, dim=1)
            
            max_dist[dist > max_dist] = dist[dist > max_dist]
            # 更新Y0为Y1的副本,更新x为Y1的副本
            Y0 = Y1.clone()
            x = uav_graph[[i for i in range(B)], idx.data].clone()
            
            mask[[i for i in range(B)], idx.data] += -np.inf        # 将mask中的被访问过的位置置为-np.inf。
            mask[idx.data==size] = -np.inf                          # 将mask中idx等于size的位置置为-np.inf
        
        return max_dist.mean()
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

