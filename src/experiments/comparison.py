import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

from ..models.proposed_model import ProposedModel
from ..utils.metrics import compute_metrics
from ..utils.visualization import plot_waveforms, plot_spectrograms, plot_error_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparison:
    """Class for comparing different denoising models."""
    
    def __init__(self, config: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model comparison.
        
        Args:
            config: Configuration dictionary
            device: Device to run models on
        """
        self.config = config
        self.device = device
        self.results = {}
        self.models = {}
        logger.info(f"Using device: {device}")
        
    def load_baseline_models(self, model_paths: Dict[str, str]):
        """Load baseline models from paths.
        
        Args:
            model_paths: Dictionary mapping model names to their paths
        """
        for name, path in model_paths.items():
            logger.info(f"Loading {name} model from {path}")
            try:
                if name == 'cnn':
                    from ..models.cnn import CNNDenoiser
                    self.models[name] = CNNDenoiser(self.config)
                elif name == 'bilstm':
                    from ..models.bilstm import BiLSTMDenoiser
                    self.models[name] = BiLSTMDenoiser(self.config)
                elif name == 'cnn_bilstm':
                    from ..models.cnn_bilstm import CNNBiLSTMDenoiser
                    self.models[name] = CNNBiLSTMDenoiser(self.config)
                elif name == 'msc':
                    from ..models.msc import MSCDenoiser
                    self.models[name] = MSCDenoiser(self.config)
                elif name == 'msc_bilstm':
                    from ..models.msc_bilstm import MSCBiLSTMDenoiser
                    self.models[name] = MSCBiLSTMDenoiser(self.config)
                elif name == 'msc_transformer_bilstm':
                    from ..models.msc_transformer_bilstm import MSCTransformerBiLSTMDenoiser
                    self.models[name] = MSCTransformerBiLSTMDenoiser(self.config)
                else:
                    raise ValueError(f"Unknown model type: {name}")
                
                # Load weights and move to device
                self.models[name].load_state_dict(torch.load(path, map_location=self.device))
                self.models[name].to(self.device)
                self.models[name].eval()
                logger.info(f"Successfully loaded {name} model")
            except Exception as e:
                logger.error(f"Error loading {name} model: {str(e)}")
                raise
            
    def load_proposed_model(self, model_path: str):
        """Load proposed model from path."""
        logger.info(f"Loading proposed model from {model_path}")
        try:
            self.models['proposed'] = ProposedModel(self.config)
            self.models['proposed'].load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.models['proposed'].to(self.device)
            self.models['proposed'].eval()
            logger.info("Successfully loaded proposed model")
        except Exception as e:
            logger.error(f"Error loading proposed model: {str(e)}")
            raise
        
    def evaluate_models(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate all models on test data.
        
        Args:
            test_loader: Test data loader
        """
        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name} model...")
            all_preds = []
            all_targets = []
            all_inputs = []
            
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    # Ensure predictions match target shape
                    if pred.shape != y.shape:
                        pred = pred.view(y.shape)
                    all_preds.append(pred.cpu())
                    all_targets.append(y.cpu())
                    all_inputs.append(x.cpu())
            
            # Concatenate results
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_inputs = torch.cat(all_inputs, dim=0).numpy()
            
            # Compute metrics
            metrics = compute_metrics(all_targets, all_preds)
            
            # Store results
            self.results[name] = {
                'metrics': metrics,
                'sample_input': all_inputs[0],
                'sample_pred': all_preds[0],
                'sample_target': all_targets[0]
            }
            
            # Log metrics
            logger.info(f"Results for {name}:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
    def plot_metric_comparison(self, baseline_results: Dict, ablation_results: Dict, 
                               metric: str, save_path: Optional[str] = None):
        """Plot comparison of a specific metric across baseline and ablation models.
        
        Args:
            baseline_results: Results from baseline models
            ablation_results: Results from ablation variants
            metric: Metric to plot
            save_path: Path to save the plot
        """
        # Prepare data
        models = []
        values = []
        
        # Add baseline results
        for model_name, result in baseline_results.items():
            models.append(model_name)
            values.append(result['metrics'][metric])
        
        # Add ablation results
        for variant_name, result in ablation_results.items():
            models.append(f"Ablated\n{variant_name}")
            values.append(result['metrics'][metric])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values)
        
        # Color ablation variants differently
        n_baselines = len(baseline_results)
        for i in range(n_baselines, len(bars)):
            bars[i].set_color('lightcoral')
        
        plt.title(f'{metric.upper()} Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.upper())
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved {metric} comparison plot to {save_path}")
        plt.close()
        
    def plot_sample_comparisons(self, save_dir: Optional[str] = None):
        """Plot sample waveform and spectrogram comparisons."""
        for name, result in self.results.items():
            logger.info(f"Generating plots for {name} model...")
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot waveforms
            plot_waveforms(
                result['sample_target'],
                result['sample_input'],
                result['sample_pred'],
                title=f'Waveform Comparison - {name}',
                save_path=save_dir / f'waveform_{name}.png' if save_dir else None
            )
            
            # Plot spectrograms
            plot_spectrograms(
                result['sample_target'],
                result['sample_input'],
                result['sample_pred'],
                title=f'Spectrogram Comparison - {name}',
                save_path=save_dir / f'spectrogram_{name}.png' if save_dir else None
            )
            
            # Plot error distribution
            plot_error_distribution(
                result['sample_target'],
                result['sample_pred'],
                title=f'Error Distribution - {name}',
                save_path=save_dir / f'error_{name}.png' if save_dir else None
            )
            
    def generate_report(self, save_dir: str):
        """Generate comprehensive comparison report.
        
        Args:
            save_dir: Directory to save report files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating report in {save_dir}")
        
        # Plot metric comparisons
        self.plot_metric_comparison(save_dir)
        
        # Plot sample comparisons
        self.plot_sample_comparisons(save_dir)
        
        # Save detailed metrics
        metrics_file = save_dir / 'detailed_metrics.txt'
        with open(metrics_file, 'w') as f:
            for model_name, result in self.results.items():
                f.write(f"\nResults for {model_name}:\n")
                f.write("-" * 50 + "\n")
                for metric, value in result['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
        logger.info(f"Saved detailed metrics to {metrics_file}")
        
    def load_model(self, model_type):
        """Load model based on type.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            Loaded model instance
        """
        if model_type == 'proposed':
            return ProposedModel(self.config)
        elif model_type == 'cnn':
            # Import and return CNN model
            from ..models.cnn import CNNDenoiser
            return CNNDenoiser(self.config)
        elif model_type == 'bilstm':
            # Import and return BiLSTM model
            from ..models.bilstm import BiLSTMDenoiser
            return BiLSTMDenoiser(self.config)
        elif model_type == 'cnn_bilstm':
            # Import and return CNN-BiLSTM model
            from ..models.cnn_bilstm import CNNBiLSTMDenoiser
            return CNNBiLSTMDenoiser(self.config)
        elif model_type == 'msc':
            # Import and return MSC model
            from ..models.msc import MSCDenoiser
            return MSCDenoiser(self.config)
        elif model_type == 'msc_bilstm':
            # Import and return MSC-BiLSTM model
            from ..models.msc_bilstm import MSCBiLSTMDenoiser
            return MSCBiLSTMDenoiser(self.config)
        elif model_type == 'msc_transformer_bilstm':
            # Import and return MSC-Transformer-BiLSTM model
            from ..models.msc_transformer_bilstm import MSCTransformerBiLSTMDenoiser
            return MSCTransformerBiLSTMDenoiser(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train_and_evaluate(self, model_type, train_data, test_data):
        """Train and evaluate a specific model.
        
        Args:
            model_type: Type of model to train
            train_data: Training data tuple (input, target)
            test_data: Test data tuple (input, target)
            
        Returns:
            Dictionary of evaluation metrics
        """
        model = self.load_model(model_type)
        model.compile()
        model.train(*train_data)
        # Evaluate model on test data
        model.eval()
        with torch.no_grad():
            x_test, y_test = test_data
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            y_pred = model(x_test)
            metrics = compute_metrics(y_test.cpu(), y_pred.cpu())
        return metrics
        
    def run_comparison(self, test_data_path: str, model_paths: dict, save_dir: Optional[str] = None):
        """Run comparison between pre-trained models.
        
        Args:
            test_data_path: Path to test data (.npz file with 'noisy_data' and 'original_data')
            model_paths: Dictionary mapping model names to their saved weights paths
            save_dir: Directory to save comparison results
        """
        logger.info("Starting model comparison...")
        save_dir = Path(save_dir) if save_dir else Path("comparison_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        test_data = np.load(test_data_path)
        x_test = torch.tensor(test_data['noisy_data'], dtype=torch.float32)
        y_test = torch.tensor(test_data['original_data'], dtype=torch.float32)
        
        # Normalize data
        x_max = torch.max(torch.abs(x_test))
        y_max = torch.max(torch.abs(y_test))
        x_test = x_test / x_max
        y_test = y_test / y_max
        
        # Create test loader
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4
        )
        
        # Load and evaluate models
        logger.info("Loading and evaluating models...")
        self.load_baseline_models(model_paths)
        
        if 'proposed' in model_paths:
            self.load_proposed_model(model_paths['proposed'])
        
        # Evaluate all models
        self.evaluate_models(test_loader)
        
        # Generate comprehensive report
        logger.info("Generating comparison report...")
        
        # 1. Overall metrics comparison
        self.plot_metric_comparison(save_dir)
        
        # 2. Sample-wise comparisons
        self.plot_sample_comparisons(save_dir)
        
        # 3. Save detailed metrics
        metrics_file = save_dir / 'detailed_metrics.txt'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"\n{model_name} 模型结果:\n")
                f.write("-" * 30 + "\n")
                metrics = result['metrics']
                
                # Basic metrics
                f.write(f"均方误差 (MSE): {metrics['mse']:.6f}\n")
                f.write(f"平均绝对误差 (MAE): {metrics['mae']:.6f}\n")
                f.write(f"信噪比 (SNR): {metrics['snr']:.2f} dB\n")
                f.write(f"峰值信噪比 (PSNR): {metrics['psnr']:.2f} dB\n")
                
                # Variance metrics
                f.write(f"\n方差指标:\n")
                f.write(f"误差方差: {metrics['error_variance']:.6f}\n")
                f.write(f"预测信号方差: {metrics['prediction_variance']:.6f}\n")
                f.write(f"目标信号方差: {metrics['target_variance']:.6f}\n")
                f.write(f"方差比: {metrics['variance_ratio']:.6f}\n")
                
                # Spectral metrics
                f.write(f"\n频谱指标:\n")
                f.write(f"频谱MSE: {metrics['spectral_mse']:.6f}\n")
                f.write(f"频谱MAE: {metrics['spectral_mae']:.6f}\n")
                f.write(f"频谱SNR: {metrics['spectral_snr']:.2f} dB\n")
                f.write("\n" + "=" * 50 + "\n")
        
        logger.info(f"Results saved to {save_dir}")
        logger.info("Model comparison completed successfully")
        return self.results
