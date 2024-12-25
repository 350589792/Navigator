import torch
import torch.nn as nn
from .base_model import BaseModel

class MSCBiLSTMDenoiser(BaseModel):
    """Multi-Scale CNN with BiLSTM model for seismic data denoising."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.hidden_channels = 64  # Fixed size to match pretrained weights
        self.lstm_hidden = 64    # Fixed size for BiLSTM
        self.num_layers = 2      # Fixed number of layers
        
        # Multi-scale convolution branches
        self.conv1 = nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, self.hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, self.hidden_channels, kernel_size=7, padding=3)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.hidden_channels * 3,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(self.lstm_hidden * 2, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure input is in the correct format (B, C, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
            
        # Multi-scale feature extraction
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        
        # Concatenate features
        merged = torch.cat([out1, out2, out3], dim=1)
        
        # Prepare for LSTM (B, L, C)
        lstm_in = merged.transpose(1, 2)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Final output
        return self.fc(lstm_out)
