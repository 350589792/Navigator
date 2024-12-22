# models/ai_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class GlassFurnaceModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(GlassFurnaceModel, self).__init__()

        # 模型参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )

        # 残差连接层
        self.residual = nn.Linear(input_size, 1)

        # 批归一化层
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def attention_mechanism(self, lstm_output):
        """注意力机制实现"""
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output, attention_weights

    def forward(self, x):
        """前向传播"""
        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 应用注意力机制
        attended_output, attention_weights = self.attention_mechanism(lstm_out)

        # 批归一化
        attended_output = self.batch_norm(attended_output)

        # 全连接层处理
        fc_out = self.fc_layers(attended_output)

        # 残差连接
        residual_out = self.residual(x[:, -1, :])

        # 组合输出
        final_output = fc_out + residual_out

        return final_output, attention_weights


class TemperaturePredictor:
    def __init__(self, model, device, learning_rate=0.001):
        """
        初始化温度预测器
        Args:
            model: GlassFurnaceModel实例
            device: 运行设备（CPU/GPU）
            learning_rate: 学习率
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def create_dataloaders(self, X_train, y_train, X_val, y_val, batch_size=32):
        """创建数据加载器"""
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def train_epoch(self, train_loader, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs, _ = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        attention_maps = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, attention = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
                attention_maps.extend(attention.cpu().numpy())

        return (
            total_loss / len(val_loader),
            np.array(predictions),
            np.array(actuals),
            np.array(attention_maps)
        )

    def predict(self, X):
        """进行预测"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions, attention = self.model(X_tensor)

        return predictions.cpu().numpy(), attention.cpu().numpy()

    def calculate_prediction_interval(self, X, n_samples=100):
        """计算预测区间"""
        self.model.train()  # 启用dropout以获得不确定性估计
        predictions = []

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)

        return mean_pred, lower_bound, upper_bound