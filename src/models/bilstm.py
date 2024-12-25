import torch
import torch.nn as nn
from .base_model import BaseModel

class BiLSTMDenoiser(BaseModel):
    """BiLSTM model for seismic data denoising."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Match the architecture from bilstm_pytorch.py
        self.hidden_size = 64  # Fixed size to match pretrained weights
        self.num_layers = 2    # Fixed layers to match pretrained weights
        
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(self.hidden_size * 2, 1)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure input is in the correct format (B, L, C)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3 and x.shape[-1] != 1:
            x = x.transpose(1, 2)
            
        output, _ = self.bilstm(x)
        return self.fc(output)
