import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel attention module for focusing on important feature channels."""
    
    def __init__(self, channels, reduction_ratio=16):
        """Initialize channel attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Channel reduction ratio for attention
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Tensor of shape (batch_size, channels, time_steps) with channel attention applied
        """
        b, c, _ = x.size()
        
        # Average pooling branch
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        # Max pooling branch
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention module for focusing on important time steps."""
    
    def __init__(self, input_dim, hidden_dim=64):
        """Initialize temporal attention module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Tensor of shape (batch_size, channels, time_steps) with temporal attention applied
        """
        # Transpose for attention: (batch, time_steps, channels)
        x = x.transpose(1, 2)
        
        # Linear transformations
        Q = self.query(x)  # (batch, time_steps, hidden_dim)
        K = self.key(x)    # (batch, time_steps, hidden_dim)
        V = self.value(x)  # (batch, time_steps, hidden_dim)
        
        # Scaled dot-product attention
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Transpose back: (batch, channels, time_steps)
        return out.transpose(1, 2)

class DualAttention(nn.Module):
    """Combined channel and temporal attention module."""
    
    def __init__(self, channels, reduction_ratio=16, hidden_dim=64):
        """Initialize dual attention module.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Channel reduction ratio for channel attention
            hidden_dim: Hidden dimension for temporal attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.temporal_attention = TemporalAttention(channels, hidden_dim)
        
    def forward(self, x):
        """Forward pass applying both channel and temporal attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Tensor with both attention mechanisms applied
        """
        x = self.channel_attention(x)
        x = self.temporal_attention(x)
        return x
