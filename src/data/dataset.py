import numpy as np
import segyio
from pathlib import Path
from .noise_generator import (
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
    add_salt_and_pepper_noise
)

class SeismicDataset:
    """Dataset class for handling seismic data."""
    
    def __init__(self, config):
        """Initialize SeismicDataset with configuration.
        
        Args:
            config: Dictionary containing dataset configuration
        """
        self.data_path = Path(config['data']['path']).expanduser()
        self.noise_type = config['noise']['type']
        self.noise_params = config['noise']['params'][self.noise_type]
        self.adaptive = config['noise']['adaptive']
        self.normalize = config['data'].get('normalize', True)
        
    def load_data(self):
        """Load seismic data from SEG-Y file.
        
        Returns:
            Numpy array of seismic traces
        """
        with segyio.open(str(self.data_path), 'r', ignore_geometry=True) as f:
            # Get basic file info
            n_traces = f.tracecount
            samples_per_trace = len(f.samples)
            
            # Read all traces
            data = np.zeros((n_traces, samples_per_trace))
            for i in range(n_traces):
                data[i] = f.trace[i]
                
            if self.normalize:
                data = (data - np.mean(data)) / np.std(data)
                
            return data
        
    def add_noise(self, data):
        """Add noise to data based on configuration.
        
        Args:
            data: Input data array
            
        Returns:
            Noisy data array
        """
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(
                data,
                noise_ratio=self.noise_params['base_noise_ratio'],
                adaptive=self.adaptive
            )
        elif self.noise_type == 'poisson':
            return add_poisson_noise(
                data,
                scale=self.noise_params['base_scale'],
                adaptive=self.adaptive
            )
        elif self.noise_type == 'uniform':
            return add_uniform_noise(
                data,
                noise_ratio=self.noise_params['base_noise_ratio'],
                adaptive=self.adaptive
            )
        elif self.noise_type == 'salt_and_pepper':
            return add_salt_and_pepper_noise(
                data,
                prob=self.noise_params['base_prob'],
                adaptive=self.adaptive
            )
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
            
    def prepare_data(self, train_ratio=0.8):
        """Prepare data for training/testing.
        
        Args:
            train_ratio: Ratio of training data
            
        Returns:
            Training and testing data
        """
        data = self.load_data()
        noisy_data = self.add_noise(data)
        
        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        train_noisy = noisy_data[:split_idx]
        test_data = data[split_idx:]
        test_noisy = noisy_data[split_idx:]
        
        return (train_data, train_noisy), (test_data, test_noisy)
