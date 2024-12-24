# models/hybrid_model.py

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
import logging
from ai_model import TemperaturePredictor


class HybridModel(nn.Module):
    def __init__(self, ai_model, cfd_model, config, device='cpu'):
        """
        初始化混合模型
        Args:
            ai_model: GlassFurnaceModel实例
            cfd_model: 配置好的CFD模型
            config: 混合模型配置
            device: 运行设备（CPU/GPU）
        """
        super(HybridModel, self).__init__()
        self.base_model = ai_model
        self.predictor = TemperaturePredictor(ai_model, device)
        self.cfd_model = cfd_model
        self.config = config
        self.device = device

        # 配置日志
        self.setup_logging()

        # 融合权重 - 作为可训练参数
        self.ai_weight = nn.Parameter(torch.tensor(config.get('ai_weight', 0.6)))
        self.cfd_weight = nn.Parameter(torch.tensor(config.get('cfd_weight', 0.4)))

        # 温度阈值
        self.temp_threshold = config.get('temp_threshold', 2.0)

    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hybrid_model.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def forward(self, input_data):
        """
        PyTorch forward pass
        Args:
            input_data: 输入数据张量
        Returns:
            预测结果张量
        """
        try:
            # 确保输入数据是张量
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)

            # 使用基础模型进行前向传播
            output, attention = self.base_model(input_data)
            return output, attention
        except Exception as e:
            self.logger.error(f"Forward pass error: {str(e)}")
            raise

    def predict(self, input_data, return_details=False):
        """
        进行混合预测
        Args:
            input_data: 输入数据，形状为 (batch_size, sequence_length, features) 或 (batch_size, features)
            return_details: 是否返回详细信息
        Returns:
            预测结果和可选的详细信息
        """
        # 确保输入数据维度正确
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()
            
        # 如果输入是2D，添加序列维度
        if len(input_data.shape) == 2:
            input_data = input_data.reshape(input_data.shape[0], 1, -1)
        # 如果输入是1D，添加批次和序列维度
        elif len(input_data.shape) == 1:
            input_data = input_data.reshape(1, 1, -1)
        try:
            # 1. AI模型预测
            ai_pred, ai_attention = self.predictor.predict(input_data)
            self.logger.info(f"AI预测完成：{ai_pred.shape}")

            # 2. 设置CFD边界条件
            cfd_boundary = self.prepare_cfd_boundary(ai_pred)

            # 3. CFD模拟
            try:
                cfd_results = self.cfd_model.run_simulation(cfd_boundary)
                self.logger.info("CFD模拟完成")
                
                # 检查CFD结果是否有效
                if not isinstance(cfd_results, dict) or 'temperature_field' not in cfd_results:
                    self.logger.warning("CFD结果格式无效，使用AI预测结果")
                    return ai_pred if not return_details else (ai_pred, {
                        'ai_prediction': ai_pred,
                        'ai_attention': ai_attention,
                        'cfd_results': None,
                        'fusion_weights': {'ai': 1.0, 'cfd': 0.0}
                    })
                
                if np.any(np.isnan(cfd_results['temperature_field'])):
                    self.logger.warning("CFD结果包含NaN值，使用AI预测结果")
                    return ai_pred if not return_details else (ai_pred, {
                        'ai_prediction': ai_pred,
                        'ai_attention': ai_attention,
                        'cfd_results': None,
                        'fusion_weights': {'ai': 1.0, 'cfd': 0.0}
                    })

                # 4. 结果融合
                final_prediction = self.fuse_predictions(ai_pred, cfd_results)
                self.logger.info("预测融合完成")
            except Exception as e:
                self.logger.warning(f"CFD模拟失败: {str(e)}，使用AI预测结果")
                return ai_pred if not return_details else (ai_pred, {
                    'ai_prediction': ai_pred,
                    'ai_attention': ai_attention,
                    'cfd_results': None,
                    'fusion_weights': {'ai': 1.0, 'cfd': 0.0}
                })

            if return_details:
                return final_prediction, {
                    'ai_prediction': ai_pred,
                    'ai_attention': ai_attention,
                    'cfd_results': cfd_results,
                    'fusion_weights': {
                        'ai': self.ai_weight.item(),
                        'cfd': self.cfd_weight.item()
                    }
                }
            return final_prediction

        except Exception as e:
            self.logger.error(f"预测过程出错: {str(e)}")
            raise

    def prepare_cfd_boundary(self, ai_predictions):
        """
        根据AI预测准备CFD边界条件
        Args:
            ai_predictions: AI模型的预测结果
        Returns:
            CFD边界条件字典
        """
        # 提取关键温度点
        key_temperatures = {
            'top_temp': float(ai_predictions[0]),
            'bottom_temp': self.get_bottom_temperature(ai_predictions),
            'inlet_temp': self.get_inlet_temperature(ai_predictions),
            'outlet_temp': self.get_outlet_temperature(ai_predictions),
            'left_temp': self.config.get('default_left_temp', 1200.0),
            'right_temp': self.config.get('default_right_temp', 1200.0)
        }

        # 添加流动条件
        flow_conditions = {
            'inlet_velocity': self.config.get('inlet_velocity', 0.5),
            'outlet_pressure': self.config.get('outlet_pressure', 0.0)
        }

        return {**key_temperatures, **flow_conditions}

    def fuse_predictions(self, ai_pred, cfd_results):
        """
        融合AI和CFD预测结果
        Args:
            ai_pred: AI模型预测
            cfd_results: CFD模拟结果
        Returns:
            融合后的预测结果
        """
        try:
            # 1. 提取CFD温度场的关键点
            cfd_temp = cfd_results['temperature_field']
            
            # 检查CFD温度场是否有效
            if np.any(np.isnan(cfd_temp)):
                self.logger.warning("CFD温度场包含NaN值，使用AI预测结果")
                return ai_pred

            # 2. 计算置信度
            ai_confidence = self.calculate_ai_confidence(ai_pred)
            cfd_confidence = self.calculate_cfd_confidence(cfd_temp)

            # 3. 动态调整权重
            local_weights = self.calculate_local_weights(
                ai_confidence,
                cfd_confidence
            )


            # 4. 提取CFD预测点
            cfd_predictions = self.extract_cfd_predictions(cfd_temp)
            if np.any(np.isnan(cfd_predictions)):
                self.logger.warning("CFD预测点包含NaN值，使用AI预测结果")
                return ai_pred

            # 5. 融合预测
            fused_temp = (
                    local_weights['ai'] * ai_pred +
                    local_weights['cfd'] * cfd_predictions
            )

            # 6. 应用温度约束
            fused_temp = self.apply_temperature_constraints(fused_temp)
            
            # 最终验证
            if np.any(np.isnan(fused_temp)):
                self.logger.warning("融合结果包含NaN值，使用AI预测结果")
                return ai_pred
                
            return fused_temp
            
        except Exception as e:
            self.logger.error(f"融合预测失败: {str(e)}，使用AI预测结果")
            return ai_pred

    def calculate_ai_confidence(self, ai_pred):
        """计算AI预测的置信度"""
        # 确保输入数据维度正确
        if isinstance(ai_pred, torch.Tensor):
            ai_pred = ai_pred.detach().cpu().numpy()
        
        # 如果输入是2D，添加序列维度
        if len(ai_pred.shape) == 2:
            ai_pred = ai_pred.reshape(ai_pred.shape[0], 1, -1)
        # 如果输入是1D，添加批次和序列维度
        elif len(ai_pred.shape) == 1:
            ai_pred = ai_pred.reshape(1, 1, -1)
            
        # 基于预测方差计算置信度
        _, lower, upper = self.predictor.calculate_prediction_interval(ai_pred)
        confidence = 1.0 / (upper - lower + 1e-6)  # 添加小值避免除零
        return np.clip(confidence, 0, 1)

    def calculate_cfd_confidence(self, cfd_temp):
        """计算CFD结果的置信度"""
        # 基于物理约束计算置信度
        temp_gradient = np.gradient(cfd_temp)
        max_gradient = np.max(np.abs(temp_gradient))

        # 梯度越大，置信度越低
        confidence = np.exp(-max_gradient / self.config.get('gradient_scale', 1.0))
        return confidence

    def calculate_local_weights(self, ai_confidence, cfd_confidence):
        """计算局部权重"""
        # 添加小值避免除零
        epsilon = 1e-10
        
        # 处理无效值
        ai_confidence = np.nan_to_num(ai_confidence, nan=0.0)
        cfd_confidence = np.nan_to_num(cfd_confidence, nan=0.0)
        
        total_confidence = ai_confidence + cfd_confidence + epsilon

        weights = {
            'ai': (ai_confidence / total_confidence) * self.ai_weight.item(),
            'cfd': (cfd_confidence / total_confidence) * self.cfd_weight.item()
        }

        # 归一化权重
        total_weight = sum(weights.values()) + epsilon
        return {k: v / total_weight for k, v in weights.items()}

    def extract_cfd_predictions(self, cfd_temp):
        """从CFD温度场提取预测点"""
        # 提取关键位置的温度
        key_points = self.config.get('key_points', [(0.5, 0.5)])
        predictions = []

        for point in key_points:
            i, j = int(point[0] * cfd_temp.shape[0]), int(point[1] * cfd_temp.shape[1])
            predictions.append(cfd_temp[i, j])

        return np.array(predictions)

    def apply_temperature_constraints(self, temperature):
        """应用温度约束"""
        # 确保温度在物理合理范围内
        min_temp = self.config.get('min_temperature', 0)
        max_temp = self.config.get('max_temperature', 2000)

        return np.clip(temperature, min_temp, max_temp)

    def evaluate_prediction(self, prediction, actual):
        """评估预测结果"""
        mse = mean_squared_error(actual, prediction)
        rmse = np.sqrt(mse)

        max_error = np.max(np.abs(prediction - actual))
        mean_error = np.mean(np.abs(prediction - actual))

        return {
            'mse': mse,
            'rmse': rmse,
            'max_error': max_error,
            'mean_error': mean_error,
            'within_threshold': np.mean(np.abs(prediction - actual) <= self.temp_threshold)
        }

    def save_model(self, path):
        """保存模型"""
        model_state = {
            'ai_model_state': self.base_model.state_dict(),
            'config': self.config,
            'weights': {
                'ai': self.ai_weight.item(),
                'cfd': self.cfd_weight.item()
            }
        }
        torch.save(model_state, path)

    def load_model(self, path):
        """加载模型"""
        model_state = torch.load(path)
        self.base_model.load_state_dict(model_state['ai_model_state'])
        self.config = model_state['config']
        self.ai_weight = nn.Parameter(torch.tensor(model_state['weights']['ai']))
        self.cfd_weight = nn.Parameter(torch.tensor(model_state['weights']['cfd']))

    def get_bottom_temperature(self, predictions):
        """获取底部温度"""
        return float(predictions[1]) if len(predictions) > 1 else self.config.get('default_bottom_temp', 1000.0)

    def get_inlet_temperature(self, predictions):
        """获取入口温度"""
        return float(predictions[2]) if len(predictions) > 2 else self.config.get('default_inlet_temp', 800.0)

    def get_outlet_temperature(self, predictions):
        """获取出口温度"""
        return float(predictions[3]) if len(predictions) > 3 else self.config.get('default_outlet_temp', 1200.0)
