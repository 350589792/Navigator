import torch
import torch.nn as nn
from .base_model import BaseModel

class CNNDenoiser(BaseModel):
    """CNN model for seismic data denoising."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.hidden_channels = 64  # Fixed size to match pretrained weights
        
        self.network = nn.Sequential(
            nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure input is in the correct format (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
            
        return self.network(x)
