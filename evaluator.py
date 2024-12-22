# evaluation/evaluator.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, config):
        """
        初始化评估器
        Args:
            model: 要评估的模型
            config: 评估配置
        """
        self.model = model
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        log_path = Path(self.config['paths']['log_path'])
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(np.abs(y_true - y_pred))
        }

        # 计算温度阈值内的预测比例
        temp_threshold = self.config['model']['hybrid']['temp_threshold']
        metrics['within_threshold'] = np.mean(np.abs(y_true - y_pred) <= temp_threshold)

        return metrics

    def evaluate_model(self, test_data, test_labels):
        """评估模型性能"""
        # 模型预测
        predictions = self.model.predict(test_data)

        # 计算指标
        metrics = self.calculate_metrics(test_labels, predictions)

        # 保存评估结果
        self.save_evaluation_results(metrics, predictions, test_labels)

        return metrics, predictions

    def analyze_error_distribution(self, y_true, y_pred):
        """分析误差分布"""
        errors = y_true - y_pred

        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'q1_error': np.percentile(errors, 25),
            'q3_error': np.percentile(errors, 75),
            'skewness': pd.Series(errors).skew(),
            'kurtosis': pd.Series(errors).kurtosis()
        }

        return error_stats

    def analyze_prediction_intervals(self, test_data, confidence_level=0.95):
        """分析预测区间"""
        predictions = []

        # 多次预测以获得分布
        for _ in range(100):
            pred = self.model.predict(test_data)
            predictions.append(pred)

        predictions = np.array(predictions)

        # 计算置信区间
        lower = np.percentile(predictions, ((1 - confidence_level) / 2) * 100, axis=0)
        upper = np.percentile(predictions, (1 - (1 - confidence_level) / 2) * 100, axis=0)

        return lower, upper

    def analyze_feature_importance(self, test_data):
        """分析特征重要性"""
        feature_importance = {}
        base_pred = self.model.predict(test_data)

        for i in range(test_data.shape[2]):
            # 打乱单个特征
            shuffled_data = test_data.copy()
            shuffled_data[:, :, i] = np.random.permutation(shuffled_data[:, :, i])

            # 计算性能下降
            shuffled_pred = self.model.predict(shuffled_data)
            importance = mean_squared_error(base_pred, shuffled_pred)

            feature_importance[f'feature_{i}'] = importance

        return feature_importance

    def analyze_time_dependency(self, y_true, y_pred, timestamps):
        """分析时间依赖性"""
        errors = np.abs(y_true - y_pred)

        # 按时间段分析误差
        time_analysis = pd.DataFrame({
            'timestamp': timestamps,
            'error': errors
        })

        # 按小时分组
        hourly_errors = time_analysis.groupby(
            time_analysis['timestamp'].dt.hour
        )['error'].mean()

        # 按天分组
        daily_errors = time_analysis.groupby(
            time_analysis['timestamp'].dt.day
        )['error'].mean()

        return {
            'hourly_errors': hourly_errors,
            'daily_errors': daily_errors
        }

    def generate_evaluation_report(self, metrics, error_stats, feature_importance):
        """生成评估报告"""
        report = {
            'model_performance': metrics,
            'error_analysis': error_stats,
            'feature_importance': feature_importance
        }

        # 添加结论和建议
        report['conclusions'] = self.generate_conclusions(metrics, error_stats)
        report['recommendations'] = self.generate_recommendations(
            metrics,
            error_stats,
            feature_importance
        )

        return report

    def generate_conclusions(self, metrics, error_stats):
        """生成结论"""
        conclusions = []

        # 性能评估
        if metrics['within_threshold'] >= 0.95:
            conclusions.append("模型性能达到要求，95%以上的预测在误差阈值内")
        else:
            conclusions.append(f"模型性能未达标准，只有{metrics['within_threshold'] * 100:.1f}%的预测在误差阈值内")

        # 误差分析
        if abs(error_stats['mean_error']) < 0.1:
            conclusions.append("预测误差分布接近零均值，无明显系统性偏差")
        else:
            conclusions.append(f"存在系统性偏差，平均误差为{error_stats['mean_error']:.2f}")

        return conclusions

    def generate_recommendations(self, metrics, error_stats, feature_importance):
        """生成建议"""
        recommendations = []

        # 基于性能改进建议
        if metrics['within_threshold'] < 0.95:
            recommendations.append("建议增加训练数据量或调整模型架构")

        # 基于误差分析建议
        if abs(error_stats['mean_error']) >= 0.1:
            recommendations.append("建议检查数据预处理步骤，可能存在标准化问题")

        # 基于特征重要性建议
        important_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        recommendations.append(
            f"建议重点关注以下特征的数据质量：{[f[0] for f in important_features]}"
        )

        return recommendations

    def save_evaluation_results(self, metrics, predictions, true_values):
        """保存评估结果"""
        output_path = Path(self.config['paths']['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存指标
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # 保存预测结果
        results_df = pd.DataFrame({
            'true_values': true_values.flatten(),
            'predictions': predictions.flatten()
        })
        results_df.to_csv(output_path / 'prediction_results.csv', index=False)

        # 保存预测对比图
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='True Values')
        plt.plot(predictions, label='Predictions')
        plt.legend()
        plt.title('Predictions vs True Values')
        plt.savefig(output_path / 'predictions_comparison.png')
        plt.close()