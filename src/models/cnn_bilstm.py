import torch
import torch.nn as nn
from .base_model import BaseModel

class CNNBiLSTMDenoiser(BaseModel):
    """Combined CNN-BiLSTM model for seismic data denoising."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.hidden_channels = 64  # Fixed size to match pretrained weights
        self.num_layers = 2     # Fixed number of layers
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=self.hidden_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(self.hidden_channels * 2, 1)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure input is in the correct format (B, C, L) for CNN
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
            
        # CNN feature extraction
        cnn_out = self.cnn(x)
        
        # Prepare for LSTM (B, L, C)
        lstm_in = cnn_out.transpose(1, 2)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Final output
        return self.fc(lstm_out)

