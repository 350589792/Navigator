"""
Analog precision proxy model for quick architecture evaluation with hardware-aware features
"""
from typing import Dict
import networkx as nx
import numpy as np
import math

class ProxyModel:
    """Hardware-aware proxy model for architecture evaluation with IMC-specific features."""
    
    def __init__(self, config: Dict):
        """Initialize proxy model with configuration."""
        self.max_resources = config['hardware']['max_resources']
        # IMC device configuration
        self.device_type = config['hardware'].get('device_type', 'RRAM')  # RRAM, PCM, or MRAM
        self.temperature = config['hardware'].get('temperature', 25)  # Operating temperature in Celsius
        self.enable_drift = config['hardware'].get('drift_modeling', True)
        self.precision_bits = config['hardware'].get('precision_bits', 8)
        
    def _calculate_drift_penalty(self, total_resources: float, time_factor: float = 1.0) -> float:
        """Calculate drift penalty based on device type and temperature."""
        # Temperature coefficient (higher penalty at higher temperatures)
        temp_factor = 1.0 + max(0, (self.temperature - 25) / 50)  # Normalized to 1.0 at 25°C
        
        # Device-specific drift characteristics
        drift_coefficients = {
            'RRAM': 0.05,  # Highest drift
            'PCM': 0.03,   # Medium drift
            'MRAM': 0.01   # Lowest drift
        }
        
        base_drift = drift_coefficients.get(self.device_type, 0.03)
        # Logarithmic drift model based on technical document
        drift_penalty = base_drift * total_resources * math.log(1 + time_factor) * temp_factor
        return drift_penalty
    
    def _calculate_precision_factor(self) -> float:
        """Calculate precision impact on performance."""
        # Higher precision requires more hardware resources
        return self.precision_bits / 8.0  # Normalized to 1.0 at 8-bit precision
    
    def estimate_performance(self, architecture: nx.DiGraph, time_factor: float = 0.0) -> float:
        """Estimate architecture performance considering hardware constraints."""
        # Calculate total hardware resource consumption with precision scaling
        total_resources = 0
        precision_factor = self._calculate_precision_factor()
        
        for node in architecture.nodes:
            node_data = architecture.nodes[node]
            if node_data.get('type') == 'conv':
                # Enhanced resource estimation considering precision
                channels = node_data.get('channels', 0)
                kernel_size = node_data.get('kernel_size', 3)
                groups = node_data.get('groups', 1)
                
                # More accurate resource calculation
                ops = (channels * channels * kernel_size * kernel_size) / groups
                total_resources += ops * precision_factor
        
        # Apply hardware constraints
        if total_resources > self.max_resources:
            return 0.0
        
        # Calculate base performance score
        num_nodes = len(architecture.nodes)
        avg_channels = np.mean([
            architecture.nodes[n].get('channels', 0)
            for n in architecture.nodes
            if architecture.nodes[n].get('type') == 'conv'
        ])
        
        # Normalize metrics
        depth_score = min(1.0, num_nodes / 50)  # Assume max depth of 50
        width_score = min(1.0, avg_channels / 512)  # Assume max width of 512
        
        # Calculate drift penalty if enabled, using time factor
        drift_penalty = self._calculate_drift_penalty(total_resources, time_factor) if self.enable_drift else 0.0
        
        # Combined score with hardware-aware adjustments
        base_score = 0.4 * depth_score + 0.6 * width_score
        final_score = base_score - drift_penalty
        
        return max(0.0, final_score)  # Ensure non-negative score
