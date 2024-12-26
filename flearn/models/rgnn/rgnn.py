import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flearn.models.rgnn.configs_new import default_config as args


class Attention(nn.Module):         #Attention模块的定义，用于在机器学习模型中进行注意力计算
    def __init__(self, n_hidden):   #初始化函数（__init__()）。它有一个参数n_hidden，用于指定隐藏层的数量。
        super(Attention, self).__init__()  #使用super()函数来调用父类的__init__()方法。super()函数返回一个临时对象，该对象允许访问父类的属性和方法。
        self.size = 0
        self.batch_size = 0     #
        self.dim = n_hidden     #用n_hidden进行初始化dim

        v  = torch.FloatTensor(n_hidden).to(args.device)
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))   #对参数v进行了均匀分布初始化

        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)   #定义了两个线性层（nn.Linear）：self.Wref和self.Wq。
        self.Wq = nn.Linear(n_hidden, n_hidden)     #两个线性层的输入和输出维度都是n_hidden，用于对输入进行线性变换。


    def forward(self, q, ref):       # query and reference，接受两个输入参数q和ref，分别表示查询和参考
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)  #获取batch的大小，并计算参考的size
        q = self.Wq(q)     # (B, dim)  对查询q和参考ref应用线性变换self.Wq和self.Wref，将其维度变为(B, dim)和(B, size, dim)。
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)

        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)，unsqueeze函数将查询的维度从(B, dim)变为(B, 1, dim)
        # v_view: (B, dim, 1) repeat()函数将其沿着第二个维度重复size次，扩展为(B, size, dim)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2) #对参数v进行扩展，使其维度与查询和参考相匹配

        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2) #通过torch.bmm()函数计算注意力权重u
        #先对相加的结果应用tanh激活函数，然后通过torch.bmm()函数对结果与v_view进行批矩阵乘法操作，最后使用squeeze()函数去掉维度为1的那一维，得到(B, size)维的注意力权重u。
        return u, ref


class LSTM(nn.Module):             #用于处理序列数据
    def __init__(self, n_hidden):  #传入一个参数n_hidden，表示LSTM隐藏状态的维度
        super(LSTM, self).__init__()

        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt) ，输入x的线性变换
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht) ，是上一个隐藏状态h的线性变换
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct) ，是上一个细胞状态c的线性变换

        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)

        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)

        # parameters for forget gate  定义了LSTM单元的输出门的权重矩阵
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)


    def forward(self, x, h, c):       # query and reference ，前向传播函数
        #对于给定的输入x，上一个隐藏状态h和细胞状态c，首先计算输入门i、遗忘门f、细胞更新值c和输出门o的激活值。
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))

        h = o * torch.tanh(c)

        return h, c  #通过输入x、前一个隐藏状态h和细胞状态c，计算出当前的隐藏状态h和细胞状态c。


class RGNN(torch.nn.Module):

    def __init__(self, n_feature, n_hidden):  #接受两个参数：n_feature（输入特征的维度）和n_hidden（隐藏层的维度）
        super(RGNN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden  #初始化了RGNN类的变量

        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)

        # pointer layer ，初始化了一个Attention模型，用于指针网络层
        self.pointer = Attention(n_hidden)

        # lstm for encoder
        self.encoder = LSTM(n_hidden)

        # trainable first hidden input ,创建了初始的隐藏状态向量h0和记忆状态向量c0，并将其转换为torch.FloatTensor类型
        h0 = torch.FloatTensor(n_hidden).to(args.device)
        c0 = torch.FloatTensor(n_hidden).to(args.device)

        # trainable latent variable coefficient ,并将其转换为torch.Tensor类型。
        alpha = torch.ones(1).to(args.device)

        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)

        self.alpha = nn.Parameter(alpha)  #将h0、c0和alpha包装为nn.Parameter对象，以便可以在优化过程中进行训练。
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)) #将h0和c0的数据进行了均匀分布的初始化

        r1 = torch.ones(1).to(args.device)
        r2 = torch.ones(1).to(args.device)
        r3 = torch.ones(1).to(args.device)
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)   #建了r1、r2和r3变量，并将其包装为nn.Parameter对象以便进行训练

        # embedding 初始化了两个线性层，用于将输入特征映射到隐藏层维度
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)


        # weights for GNN 初始化了三个线性层，用于图网络层的权重
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)

        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)


    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2) #当前城市的坐标（B，2），其中B是批次大小。
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer #来自前一层的潜在指针向量（B，size，dim），可以为None。
        '''

        #计算输入中的批次大小和城市数量，并将它们保存在模型的属性中
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)

        #将输入的城市坐标进行嵌入处理,将输入的城市坐标映射到一个更高维度的向量表示
        x = self.embedding_x(x)
        context = self.embedding_all(X_all)

        # =============================
        # process hidden variable
        # =============================
        #检查是否为第一次调用函数。如果隐藏变量h和c为空（即None），则将first_turn设置为True，否则设置为False。
        first_turn = False
        if h is None or c is None:
            first_turn = True

        if first_turn:
            # (dim) -> (B, dim)

            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()

            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))

            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)


        # =============================
        # graph neural network encoder
        # =============================

        # (B, size, dim) ,将输入的上下文张量进行形状转换，从（B，size，dim）变为（B*size，dim），其中B是批次大小，size是城市数量，dim是维度。
        context = context.view(-1, self.dim)

        #通过第一个图神经网络层进行处理，其中self.r1是一个权重系数，self.W1是一个线性变换权重矩阵，self.agg_1是一个线性变换层。
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context))

        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context))


        # LSTM encoder ,调用了LSTM编码器，接受输入坐标x、隐藏变量h和细胞门c作为参数，并返回更新后的隐藏变量h和细胞门c。
        h, c = self.encoder(x, h, c)

        # query vector ,将隐藏变量h作为查询向量q
        q = h

        # pointer ,调用指针网络模型，接受查询向量q和上下文张量context作为参数，并返回注意力分布u和其他未使用的返回值。
        u, _ = self.pointer(q, context)

        #对注意力分布u进行后续处理
        latent_u = u.clone()  #将其克隆为潜在指针向量latent_u

        u = 10 * torch.tanh(u) + mask  #通过套用双曲正切函数和线性操作将u进行缩放和偏移

        if latent is not None:
            u += self.alpha * latent   #如果潜在指针向量latent不为空，则将其加权并添加到u中
        # Compute final attention weights for relay decisions
        attention_weights = F.softmax(u, dim=1)
        
        # Ensure output shape matches target shape (B, city_size)
        prob_dist = attention_weights[:, :self.city_size]
        
        # Store relay decisions and environmental features as instance variables
        self.relay_decisions = torch.argmax(prob_dist, dim=1)
        self.environmental_features = self.embedding_all(X_all)
        
        # Reshape context for latent features
        latent_features = context.view(self.batch_size, self.city_size, -1)
        
        return prob_dist, h, c, latent_features
