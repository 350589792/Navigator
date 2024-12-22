"""
Main entry point for AnalogNAS
"""
from pathlib import Path
from .config import get_config
from .evolution import Evolution

def run_search(config_path: str = None) -> None:
    """Run architecture search with given configuration."""
    # Load configuration
    config = get_config(config_path)
    
    # Initialize evolution search
    evolution = Evolution(config)
    
    # Perform search
    best_architecture = evolution.search()
    
    return best_architecture

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_search(config_path)
