# utils/config.py

import yaml
import os
from pathlib import Path


class Config:
    """配置管理类"""

    def __init__(self, config_path):
        """
        初始化配置管理器
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {str(e)}")

    def get_model_config(self):
        """获取模型配置"""
        return {
            'ai_model': {
                'input_size': self.config['model']['ai']['input_size'],
                'hidden_size': self.config['model']['ai']['hidden_size'],
                'num_layers': self.config['model']['ai']['num_layers'],
                'dropout': self.config['model']['ai']['dropout'],
                'learning_rate': self.config['training']['learning_rate']
            },
            'cfd_model': {
                'nx': self.config['model']['cfd']['nx'],
                'ny': self.config['model']['cfd']['ny'],
                'dx': self.config['model']['cfd']['dx'],
                'dy': self.config['model']['cfd']['dy'],
                'dt': self.config['model']['cfd']['dt']
            },
            'hybrid_model': {
                'ai_weight': self.config['model']['hybrid']['ai_weight'],
                'cfd_weight': self.config['model']['hybrid']['cfd_weight'],
                'temp_threshold': self.config['model']['hybrid']['temp_threshold']
            }
        }

    def get_training_config(self):
        """获取训练配置"""
        return {
            'batch_size': self.config['training']['batch_size'],
            'epochs': self.config['training']['epochs'],
            'learning_rate': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            'validation_split': self.config['training']['validation_split'],
            'early_stopping_patience': self.config['training']['early_stopping_patience']
        }

    def get_data_config(self):
        """获取数据配置"""
        return {
            'data_path': self.config['data']['path'],
            'sequence_length': self.config['data']['sequence_length'],
            'target_column': self.config['data']['target_column'],
            'feature_columns': self.config['data']['feature_columns']
        }

    def get_paths(self):
        """获取路径配置"""
        return {
            'model_save_path': Path(self.config['paths']['model_save_path']),
            'log_path': Path(self.config['paths']['log_path']),
            'output_path': Path(self.config['paths']['output_path'])
        }

    def create_directories(self):
        """创建必要的目录"""
        paths = self.get_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, path):
        """保存当前配置"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)

    def update_config(self, updates):
        """更新配置"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value