import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ..models.proposed_model import ProposedModel, AdaptiveFeatureExtractor
from ..utils.metrics import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns

class AblationModel(ProposedModel):
    """Model variant for ablation studies."""
    
    def __init__(self, config, disable_channel_attention=False,
                 disable_temporal_attention=False,
                 disable_adaptive_weights=False,
                 disable_perceptual_loss=False):
        """Initialize ablation model.
        
        Args:
            config: Model configuration
            disable_channel_attention: Whether to disable channel attention
            disable_temporal_attention: Whether to disable temporal attention
            disable_adaptive_weights: Whether to disable adaptive feature weighting
            disable_perceptual_loss: Whether to disable perceptual loss
        """
        super().__init__(config)
        self.disable_channel_attention = disable_channel_attention
        self.disable_temporal_attention = disable_temporal_attention
        self.disable_adaptive_weights = disable_adaptive_weights
        self.disable_perceptual_loss = disable_perceptual_loss
        
        
        # Replace adaptive feature extractor if needed
        if disable_adaptive_weights:
            self.feature_extractor = SimpleFeatureExtractor(1, 32)
            
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Attention mechanisms
        if not self.disable_channel_attention:
            features = self.dual_attention.channel_attention(features)
        if not self.disable_temporal_attention:
            features = self.dual_attention.temporal_attention(features)
            
        # Final processing
        return self.final_layers(features)
        
    def loss_function(self, y_pred, y_true):
        # Time domain loss
        time_loss = nn.functional.mse_loss(y_pred, y_true)
        
        # Frequency domain loss
        y_true_fft = torch.abs(torch.fft.fft(y_true))
        y_pred_fft = torch.abs(torch.fft.fft(y_pred))
        freq_loss = nn.functional.mse_loss(y_pred_fft, y_true_fft)
        
        # Perceptual loss (optional)
        if not self.disable_perceptual_loss:
            true_features = self.compute_perceptual_features(y_true)
            pred_features = self.compute_perceptual_features(y_pred)
            perceptual_loss = nn.functional.mse_loss(pred_features, true_features)
        else:
            perceptual_loss = torch.tensor(0.0, device=y_pred.device)
        
        # Combine losses
        total_loss = (
            self.config['loss_weights']['time'] * time_loss +
            self.config['loss_weights']['spectral'] * freq_loss
        )
        
        if not self.disable_perceptual_loss:
            total_loss += self.config['loss_weights']['perceptual'] * perceptual_loss
            
        return total_loss

class SimpleFeatureExtractor(nn.Module):
    """Simple feature extractor without adaptive weights."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        
    def forward(self, x):
        out1 = nn.functional.relu(self.conv1(x))
        out2 = nn.functional.relu(self.conv2(x))
        out3 = nn.functional.relu(self.conv3(x))
        return (out1 + out2 + out3) / 3

class AblationStudy:
    """Class for conducting ablation experiments."""
    
    def __init__(self, config, device='cuda'):
        """Initialize ablation study.
        
        Args:
            config: Model configuration
            device: Device to run experiments on
        """
        self.config = config
        self.device = device
        self.results = {}
        
    def train_model(self, model, train_loader, val_loader, epochs=50):
        """Train a model and return best validation loss.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            
        Returns:
            Best validation loss achieved
        """
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config['params']['learning_rate'])
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = model.loss_function(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    loss = model.loss_function(pred, y)
                    val_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return best_val_loss, train_losses, val_losses
        
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        
        return compute_metrics(all_preds, all_targets)
        
    def run_experiment(self, name, model_config, train_loader, val_loader, test_loader):
        """Run a single ablation experiment.
        
        Args:
            name: Experiment name
            model_config: Model configuration for this experiment
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        print(f"\nRunning experiment: {name}")
        
        model = AblationModel(self.config, **model_config).to(self.device)
        best_val_loss, train_losses, val_losses = self.train_model(
            model, train_loader, val_loader
        )
        
        metrics = self.evaluate_model(model, test_loader)
        self.results[name] = {
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"Results for {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    def run_all_experiments(self, train_loader, val_loader, test_loader):
        """Run all ablation experiments.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        experiments = {
            'baseline': {},
            'no_channel_attention': {'disable_channel_attention': True},
            'no_temporal_attention': {'disable_temporal_attention': True},
            'no_adaptive_weights': {'disable_adaptive_weights': True},
            'no_perceptual_loss': {'disable_perceptual_loss': True},
            'no_attention': {
                'disable_channel_attention': True,
                'disable_temporal_attention': True
            }
        }
        
        for name, config in experiments.items():
            self.run_experiment(name, config, train_loader, val_loader, test_loader)
            
        self.plot_results()
        return self.results
        
    def plot_results(self):
        """Plot comparison of results across experiments."""
        metrics = ['mse', 'mae', 'snr', 'psnr']
        
        # Prepare data for plotting
        plot_data = {metric: [] for metric in metrics}
        experiments = []
        
        for exp_name, result in self.results.items():
            experiments.append(exp_name)
            for metric in metrics:
                plot_data[metric].append(result['metrics'][metric])
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for ax, metric in zip(axes, metrics):
            sns.barplot(x=experiments, y=plot_data[metric], ax=ax)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('ablation_results.png')
        plt.close()
