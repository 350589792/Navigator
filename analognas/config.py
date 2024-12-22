"""
Configuration management for AnalogNAS
"""
import os
import yaml
from pathlib import Path

def get_config(config_path=None):
    """Load configuration from file or return default configuration."""
    default_config = {
        'search_space': {
            'min_depth': 3,
            'max_depth': 20,
            'min_width': 16,
            'max_width': 128,
            'max_branches': 2,
        },
        'evolution': {
            'population_size': 50,
            'generations': 30,
            'mutation_prob': 0.1,
            'crossover_prob': 0.5,
        },
        'hardware': {
            'max_acu': 1000,  # Arbitrary unit for analog compute
            'target_platform': 'analog',
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    config[key] = {**value, **config[key]}
        return config
    
    return default_config
