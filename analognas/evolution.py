"""
Evolution-based architecture search implementation
"""
from typing import Dict, List
import networkx as nx
import numpy as np
from .search_space import SearchSpace
from .proxy_model import ProxyModel

class Evolution:
    """Evolution-based neural architecture search."""
    
    def __init__(self, config: Dict):
        """Initialize evolution search with configuration."""
        self.config = config
        self.population_size = config['evolution']['population_size']
        self.generations = config['evolution']['generations']
        self.mutation_prob = config['evolution']['mutation_prob']
        self.crossover_prob = config['evolution']['crossover_prob']
        
        self.search_space = SearchSpace(config)
        self.proxy_model = ProxyModel(config)
        
    def initialize_population(self) -> List[nx.DiGraph]:
        """Initialize random population of architectures."""
        return [
            self.search_space.create_random_architecture()
            for _ in range(self.population_size)
        ]
    
    def mutate(self, architecture: nx.DiGraph) -> nx.DiGraph:
        """Mutate an architecture."""
        # Simple mutation: regenerate architecture
        if np.random.random() < self.mutation_prob:
            return self.search_space.create_random_architecture()
        return architecture
    
    def crossover(self, parent1: nx.DiGraph, parent2: nx.DiGraph) -> nx.DiGraph:
        """Perform crossover between two parent architectures."""
        # Simple crossover: choose one parent
        if np.random.random() < self.crossover_prob:
            return parent1
        return parent2
    
    def search(self) -> nx.DiGraph:
        """Perform architecture search."""
        population = self.initialize_population()
        best_architecture = None
        best_score = -float('inf')
        
        for _ in range(self.generations):
            # Evaluate population
            scores = [
                self.proxy_model.estimate_performance(arch)
                for arch in population
            ]
            
            # Update best architecture
            max_score_idx = np.argmax(scores)
            if scores[max_score_idx] > best_score:
                best_score = scores[max_score_idx]
                best_architecture = population[max_score_idx]
            
            # Selection with scores as weights
            scores = np.array(scores)
            # Ensure non-negative scores for probability calculation
            scores = scores - np.min(scores) + 1e-6
            probs = scores / np.sum(scores)
            
            # Selection using weighted probabilities
            selected = []
            for _ in range(self.population_size):
                idx = np.random.choice(len(population), p=probs)
                selected.append(population[idx])
            
            # Create new population
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, len(selected) - 1)]
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_architecture
