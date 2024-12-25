import torch
import torch.nn as nn
from .base_model import BaseModel

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # Feed-forward network
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        
        return x

class MSCTransformerBiLSTMDenoiser(BaseModel):
    """Multi-Scale CNN with Transformer and BiLSTM model for seismic data denoising."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        
        self.hidden_channels = 64   # Fixed size to match pretrained weights
        self.lstm_hidden = 64     # Fixed size for BiLSTM
        self.num_layers = 2       # Fixed number of layers
        self.nhead = 4           # Fixed number of attention heads
        self.dim_feedforward = 128  # Fixed feedforward dimension
        
        # Multi-scale convolution branches
        self.conv1 = nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, self.hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1, self.hidden_channels, kernel_size=7, padding=3)
        
        # Transformer encoder
        self.transformer = TransformerEncoderBlock(
            d_model=self.hidden_channels * 3,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward
        )
        
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
        
        # Prepare for Transformer (L, B, C)
        transformer_in = merged.transpose(1, 2).transpose(0, 1)
        
        # Transformer processing
        transformer_out = self.transformer(transformer_in)
        
        # Prepare for LSTM (B, L, C)
        lstm_in = transformer_out.transpose(0, 1)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Final output
        return self.fc(lstm_out)
