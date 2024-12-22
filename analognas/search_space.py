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
        
    def mutate_layer(self, graph: nx.DiGraph, node_id: int) -> nx.DiGraph:
        """Mutate layer-level parameters."""
        if graph.nodes[node_id]['type'] == 'conv':
            # Modify layer parameters
            new_width = np.random.randint(self.min_width, self.max_width + 1)
            graph.nodes[node_id]['channels'] = new_width
            
            # Randomly change kernel size
            kernel_sizes = [1, 3, 5]
            graph.nodes[node_id]['kernel_size'] = np.random.choice(kernel_sizes)
            
            # Randomly add/remove batch normalization
            graph.nodes[node_id]['batch_norm'] = np.random.choice([True, False])
        
        return graph
    
    def mutate_block(self, graph: nx.DiGraph, start_node: int) -> nx.DiGraph:
        """Mutate block-level structure."""
        if graph.nodes[start_node]['type'] != 'output':
            # Modify number of branches
            successors = list(graph.successors(start_node))
            if len(successors) > 1:
                # Randomly remove a branch
                to_remove = np.random.choice(successors)
                graph.remove_node(to_remove)
            elif len(successors) < self.max_branches:
                # Add a new branch
                new_node = len(graph.nodes)
                width = np.random.randint(self.min_width, self.max_width + 1)
                graph.add_node(new_node, type='conv', channels=width)
                graph.add_edge(start_node, new_node)
                graph.add_edge(new_node, successors[0])
        
        return graph
    
    def mutate_connections(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Mutate neuron-level connections."""
        nodes = list(graph.nodes())
        for node in nodes:
            if graph.nodes[node]['type'] == 'conv':
                # Randomly add dropout
                graph.nodes[node]['dropout'] = np.random.uniform(0, 0.3)
                
                # Randomly add skip connections
                if node > 0 and node < len(nodes) - 1:
                    for target in nodes[node+1:-1]:
                        if graph.nodes[target]['type'] == 'conv':
                            if np.random.random() < 0.2:  # 20% chance to add skip connection
                                graph.add_edge(node, target)
        
        return graph
    
    def apply_multi_granular_mutation(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Apply mutations at multiple granularities."""
        # Deep copy the graph to avoid modifying the original
        mutated = graph.copy()
        
        # Randomly select mutation types to apply
        mutation_types = ['layer', 'block', 'connections']
        selected_mutations = np.random.choice(mutation_types, 
                                           size=np.random.randint(1, len(mutation_types) + 1),
                                           replace=False)
        
        for mutation in selected_mutations:
            if mutation == 'layer':
                # Mutate random conv layers
                conv_nodes = [n for n, d in mutated.nodes(data=True) 
                            if d.get('type') == 'conv']
                if conv_nodes:
                    node_to_mutate = np.random.choice(conv_nodes)
                    mutated = self.mutate_layer(mutated, node_to_mutate)
            
            elif mutation == 'block':
                # Mutate random blocks
                non_output_nodes = [n for n, d in mutated.nodes(data=True) 
                                  if d.get('type') != 'output']
                if non_output_nodes:
                    block_to_mutate = np.random.choice(non_output_nodes)
                    mutated = self.mutate_block(mutated, block_to_mutate)
            
            elif mutation == 'connections':
                # Mutate connections throughout the network
                mutated = self.mutate_connections(mutated)
        
        return mutated
