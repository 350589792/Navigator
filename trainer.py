# training/trainer.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from pathlib import Path
import json


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, config, device):
        """
        初始化训练器
        Args:
            model: 要训练的模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = device

        self.setup_logging()
        self.setup_training_components()

    def setup_logging(self):
        """设置日志"""
        log_path = Path(self.config['paths']['log_path'])
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_training_components(self):
        """设置训练组件"""
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def prepare_data(self, X_train, y_train, X_val, y_val):
        """准备数据加载器"""
        # 转换为张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size']
        )

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch_X, batch_y in tqdm(self.train_loader, desc='Training'):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            loss = self.criterion(outputs.squeeze(), batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 参数更新
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

        return total_loss / total_samples

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in tqdm(self.val_loader, desc='Validation'):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                # 前向传播
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)

                total_loss += loss.item() * len(batch_y)
                total_samples += len(batch_y)

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

            val_loss = total_loss / total_samples
            return val_loss, np.array(predictions), np.array(actuals)

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练模型
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
        Returns:
            训练历史记录
        """
        self.prepare_data(X_train, y_train, X_val, y_val)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'predictions': [],
            'actuals': []
        }

        for epoch in range(self.config['training']['epochs']):
            # 训练epoch
            train_loss = self.train_epoch()

            # 验证
            val_loss, predictions, actuals = self.validate()

            # 更新学习率
            self.scheduler.step(val_loss)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['predictions'].append(predictions)
            history['actuals'].append(actuals)

            # 输出进度
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        save_path = Path(self.config['paths']['model_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path / f'model_checkpoint_epoch_{epoch}.pt')

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['val_loss']

    def save_training_history(self, history):
        """保存训练历史"""
        history_path = Path(self.config['paths']['output_path']) / 'training_history.json'

        # 转换numpy数组为列表
        serializable_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'predictions': [pred.tolist() for pred in history['predictions']],
            'actuals': [act.tolist() for act in history['actuals']]
        }

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f)