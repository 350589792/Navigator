import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    
    def __init__(self, config=None):
        """Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config if config is not None else {}
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def predict(self, x):
        """Make predictions.
        
        Args:
            x: Input tensor or numpy array
            
        Returns:
            Model predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            output = self(x)
            return output.cpu().numpy()
    
    def save(self, path):
        """Save the model.
        
        Args:
            path: Save path
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load the model.
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            self.config = checkpoint['config']
    
    def get_name(self):
        """Get model name."""
        return self.__class__.__name__
    
    def custom_loss(self):
        """Get default loss function."""
        return nn.MSELoss()
