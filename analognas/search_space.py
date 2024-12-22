"""
ResNet-like search space implementation for AnalogNAS
"""
from typing import Dict, List, Optional
import networkx as nx
import numpy as np

class SearchSpace:
    """ResNet-like architecture search space."""
    
    def __init__(self, config: Dict):
        """Initialize search space with configuration."""
        self.min_depth = config['search_space']['min_depth']
        self.max_depth = config['search_space']['max_depth']
        self.min_width = config['search_space']['min_width']
        self.max_width = config['search_space']['max_width']
        self.max_branches = config['search_space']['max_branches']
        
    def create_random_architecture(self) -> nx.DiGraph:
        """Create a random architecture within the search space."""
        depth = np.random.randint(self.min_depth, self.max_depth + 1)
        graph = nx.DiGraph()
        
        # Input node
        graph.add_node(0, type='input', channels=64)
        
        # Hidden layers
        current_node = 0
        for i in range(depth):
            width = np.random.randint(self.min_width, self.max_width + 1)
            branches = np.random.randint(1, self.max_branches + 1)
            
            # Add branches
            for b in range(branches):
                next_node = len(graph.nodes)
                graph.add_node(next_node, type='conv', channels=width)
                graph.add_edge(current_node, next_node)
            
            current_node = next_node
            
        # Output node
        out_node = len(graph.nodes)
        graph.add_node(out_node, type='output')
        graph.add_edge(current_node, out_node)
        
        return graph
