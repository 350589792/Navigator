"""Example run script to demonstrate cross-platform compatibility."""
import os
from analognas.config import get_config
from analognas.evolution import Evolution

def main():
    """Run a small example search to verify cross-platform compatibility."""
    # Use smaller configuration for quick verification
    config = get_config()
    config['evolution']['population_size'] = 5  # Small population for quick test
    config['evolution']['generations'] = 2      # Few generations for quick test
    config['hardware']['max_acu'] = 500        # Limited ACU for efficiency
    
    print(f"Platform: {os.name}")  # Show which platform we're running on
    print("Starting AnalogNAS search with minimal configuration...")
    
    evolution = Evolution(config)
    best_architecture = evolution.search()
    
    print("\nSearch completed successfully!")
    print(f"Best architecture has {len(best_architecture.nodes)} nodes")
    print(f"Architecture properties:")
    node_types = {}
    for node in best_architecture.nodes:
        node_type = best_architecture.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in node_types.items():
        print(f"- {node_type}: {count} nodes")

if __name__ == "__main__":
    main()
