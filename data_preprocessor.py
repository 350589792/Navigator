# data/data_preprocessor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from scipy import stats


class DataPreprocessor:
    def __init__(self):
        """初始化数据预处理器"""
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.feature_scalers = {}

    def handle_missing_values(self, df):
        """处理缺失值"""
        # 使用KNN填充数值型特征的缺失值
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.knn_imputer.fit_transform(df[numerical_cols])

        return df

    def remove_outliers(self, df, columns, n_std=3):
        """移除异常值"""
        for col in columns:
            # 计算Z分数
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < n_std]
        return df

    def normalize_features(self, df, feature_names):
        """特征标准化"""
        for feature in feature_names:
            if feature in df.columns:
                # 为每个特征创建单独的缩放器
                scaler = StandardScaler()
                df[feature] = scaler.fit_transform(df[[feature]])
                self.feature_scalers[feature] = scaler
        return df

    def prepare_sequences(self, data, sequence_length, target_col):
        """准备时间序列数据"""
        # 确保包含所有必要的特征列
        required_features = [
            'bottom_temperature', 'heavy_oil_flow', 'natural_gas_flow',
            'air_ratio', 'air_oil_ratio', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'vault_temperature', 'inlet_temperature', 'outlet_temperature'
        ]
        
        # 检查并创建缺失的列
        for col in required_features:
            if col not in data.columns:
                if col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
                    continue  # 这些列会在create_time_features中创建
                else:
                    data[col] = 0.0  # 用0填充缺失的列
        
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            # 提取序列，使用所有必要的特征并确保float32类型
            seq = data.iloc[i:(i + sequence_length)][required_features].astype(np.float32)
            target = np.float32(data.iloc[i + sequence_length][target_col])

            sequences.append(seq.values)
            targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """划分数据集"""
        # 计算划分点
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)

        # 划分数据
        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_time_features(self, df):
        """创建时间特征"""
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def process_temperature_data(self, df):
        """处理温度数据"""
        temp_columns = [col for col in df.columns if 'temperature' in col]

        # 移除温度异常值
        df = self.remove_outliers(df, temp_columns)

        # 标准化温度数据
        df = self.normalize_features(df, temp_columns)

        return df

    def process_flow_data(self, df):
        """处理流量数据"""
        flow_columns = [col for col in df.columns if 'flow' in col]

        # 移除流量异常值
        df = self.remove_outliers(df, flow_columns)

        # 标准化流量数据
        df = self.normalize_features(df, flow_columns)

        return df

    def inverse_transform_temperature(self, data, feature_name):
        """反向转换温度数据"""
        if feature_name in self.feature_scalers:
            return self.feature_scalers[feature_name].inverse_transform(data.reshape(-1, 1))
        return data

    def prepare_full_dataset(self, df, sequence_length, target_col):
        """准备完整数据集"""
        # 1. 处理缺失值
        df = self.handle_missing_values(df)

        # 2. 处理温度数据
        df = self.process_temperature_data(df)

        # 3. 处理流量数据
        df = self.process_flow_data(df)

        # 4. 创建时间特征
        df = self.create_time_features(df)

        # 5. 准备序列数据
        X, y = self.prepare_sequences(df, sequence_length, target_col)

        # 6. 划分数据集
        return self.split_data(X, y)
