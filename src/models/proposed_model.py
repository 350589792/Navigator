import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .attention import DualAttention

class AdaptiveFeatureExtractor(nn.Module):
    """Adaptive multi-scale feature extractor."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        
        # Adaptive weights for each branch
        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # Get normalized weights
        weights = self.softmax(self.weights)
        
        # Apply convolutions
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        
        # Weighted combination
        return weights[0] * out1 + weights[1] * out2 + weights[2] * out3

class ProposedModel(BaseModel):
    """Proposed multi-branch adaptive denoising model with dual attention."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = AdaptiveFeatureExtractor(1, 32)
        
        # Dual attention
        self.dual_attention = DualAttention(
            channels=32,
            reduction_ratio=16,
            hidden_dim=64
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Feature extractor for perceptual loss
        self.feature_network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        features = self.feature_extractor(x)
        
        # Apply dual attention
        attended = self.dual_attention(features)
        
        # Final processing
        return self.final_layers(attended)
        
    def compute_perceptual_features(self, x):
        """Extract features for perceptual loss."""
        return self.feature_network(x)
        
    def loss_function(self, y_pred, y_true):
        """Multi-objective loss combining time, frequency, and perceptual losses."""
        # Time domain loss
        time_loss = F.mse_loss(y_pred, y_true)
        
        # Frequency domain loss
        y_true_fft = torch.abs(torch.fft.fft(y_true))
        y_pred_fft = torch.abs(torch.fft.fft(y_pred))
        freq_loss = F.mse_loss(y_pred_fft, y_true_fft)
        
        # Perceptual loss
        true_features = self.compute_perceptual_features(y_true)
        pred_features = self.compute_perceptual_features(y_pred)
        perceptual_loss = F.mse_loss(pred_features, true_features)
        
        # Combine losses
        total_loss = (
            self.config['loss_weights']['time'] * time_loss +
            self.config['loss_weights']['spectral'] * freq_loss +
            self.config['loss_weights']['perceptual'] * perceptual_loss
        )
        
        return total_loss
