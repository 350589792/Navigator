# utils/visualization.py

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """可视化工具类"""

    def __init__(self, save_path):
        """
        初始化可视化器
        Args:
            save_path: 图像保存路径
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # 设置样式
        plt.style.use('default')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF'])

    def plot_training_history(self, history, save=True):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))

        # 损失曲线
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        if save:
            plt.savefig(self.save_path / 'training_history.png')
            plt.close()
        else:
            plt.show()

    def plot_temperature_field(self, temperature_field, title='Temperature Field'):
        """绘制温度场"""
        plt.figure(figsize=(10, 8))
        im = plt.imshow(temperature_field, cmap='hot', aspect='auto')
        plt.title(title)
        plt.colorbar(im, label='Temperature (°C)')

        plt.savefig(self.save_path / 'temperature_field.png')
        plt.close()

    def plot_attention_weights(self, attention_weights, timestamps):
        """绘制注意力权重"""
        plt.figure(figsize=(12, 6))
        im = plt.imshow(attention_weights, aspect='auto')
        plt.xticks(range(len(timestamps)), timestamps, rotation=45)
        plt.colorbar(im, label='Weight')
        plt.title('Attention Weights Distribution')
        plt.xlabel('Time')

        plt.savefig(self.save_path / 'attention_weights.png')
        plt.close()

    def plot_prediction_comparison(self, actual, predicted, uncertainty=None):
        """绘制预测比较"""
        plt.figure(figsize=(12, 6))

        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red')

        if uncertainty is not None:
            lower, upper = uncertainty
            plt.fill_between(range(len(predicted)), lower, upper,
                             color='red', alpha=0.2, label='Uncertainty')

        plt.title('Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()

        plt.savefig(self.save_path / 'prediction_comparison.png')
        plt.close()

    def plot_error_distribution(self, errors):
        """绘制误差分布"""
        plt.figure(figsize=(10, 6))

        # 确保errors是一维数组
        errors = np.ravel(errors)

        # 只使用直方图显示误差分布
        plt.hist(errors, bins=30, density=False, alpha=0.7, color='blue', label='Error Distribution')
        plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}°C')
        plt.axvline(x=np.median(errors), color='g', linestyle='--', label=f'Median Error: {np.median(errors):.2f}°C')
        
        plt.legend()
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error (°C)')
        plt.ylabel('Count')

        plt.savefig(self.save_path / 'error_distribution.png')
        plt.close()

    def create_interactive_dashboard(self, results):
        """创建交互式仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Field', 'Prediction Comparison',
                            'Error Distribution', 'Attention Weights')
        )

        # 温度场
        fig.add_trace(
            go.Heatmap(z=results['temperature_field'],
                       colorscale='Hot',
                       showscale=True),
            row=1, col=1
        )

        # 预测比较
        fig.add_trace(
            go.Scatter(y=results['actual'], name='Actual',
                       line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=results['predicted'], name='Predicted',
                       line=dict(color='red')),
            row=1, col=2
        )

        # 误差分布
        fig.add_trace(
            go.Histogram(x=results['errors'], nbinsx=30,
                         name='Error Distribution'),
            row=2, col=1
        )

        # 注意力权重
        fig.add_trace(
            go.Heatmap(z=results['attention_weights'],
                       colorscale='Viridis'),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1200, title_text="Model Analysis Dashboard")
        fig.write_html(self.save_path / 'dashboard.html')

    def plot_cfd_results(self, cfd_results):
        """绘制CFD结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 温度场
        im1 = axes[0, 0].imshow(cfd_results['temperature_field'], cmap='hot')
        axes[0, 0].set_title('Temperature Field')
        plt.colorbar(im1, ax=axes[0, 0])

        # 速度场（u分量）
        im2 = axes[0, 1].imshow(cfd_results['velocity_field_u'], cmap='coolwarm')
        axes[0, 1].set_title('Velocity Field (u)')
        plt.colorbar(im2, ax=axes[0, 1])

        # 速度场（v分量）
        im3 = axes[1, 0].imshow(cfd_results['velocity_field_v'], cmap='coolwarm')
        axes[1, 0].set_title('Velocity Field (v)')
        plt.colorbar(im3, ax=axes[1, 0])

        # 压力场
        im4 = axes[1, 1].imshow(cfd_results['pressure_field'], cmap='viridis')
        axes[1, 1].set_title('Pressure Field')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(self.save_path / 'cfd_results.png')
        plt.close()
