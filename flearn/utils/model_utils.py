import torch
import numpy as np

class Metrics(object):
    def __init__(self, clients):
        self.clients = clients
        self.train_losses = []
        self.train_acc = []
        self.train_times = []
        self.comm_overhead = []
        
    def write(self):
        metrics = {
            'train_loss': self.train_losses,
            'train_accuracy': self.train_acc,
            'train_time': self.train_times,
            'communication_overhead': self.comm_overhead
        }
        return metrics

def read_data(dataset):
    """Read data from dataset."""
    return None  # Placeholder - we're using create_federated_data instead

def read_user_data(dataset, user_idx):
    """Read data for a specific user."""
    return None  # Placeholder - we're using create_federated_data instead

def configure_device(model, device='cpu'):
    """Configure a PyTorch model to run on a specific device.
    
    Args:
        model (nn.Module): PyTorch model to configure
        device (str): Target device ('cpu' or 'cuda')
    
    Returns:
        nn.Module: Model configured for specified device
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = 'cpu'
    
    device = torch.device(device)
    model = model.to(device)
    return model

def get_default_device():
    """Get the default device for PyTorch computations.
    
    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
