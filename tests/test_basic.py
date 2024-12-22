"""Basic tests for AnalogNAS components."""
import unittest
import networkx as nx
from analognas.config import get_config
from analognas.search_space import SearchSpace
from analognas.proxy_model import ProxyModel
from analognas.evolution import Evolution

class TestAnalogNAS(unittest.TestCase):
    """Test basic functionality of AnalogNAS components."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = get_config()
        
    def test_search_space(self):
        """Test search space creation."""
        search_space = SearchSpace(self.config)
        arch = search_space.create_random_architecture()
        self.assertIsInstance(arch, nx.DiGraph)
        self.assertGreater(len(arch.nodes), 0)
        
    def test_proxy_model(self):
        """Test proxy model evaluation."""
        search_space = SearchSpace(self.config)
        proxy_model = ProxyModel(self.config)
        arch = search_space.create_random_architecture()
        score = proxy_model.estimate_performance(arch)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_evolution(self):
        """Test evolution search."""
        evolution = Evolution(self.config)
        population = evolution.initialize_population()
        self.assertEqual(len(population), evolution.population_size)
        for arch in population:
            self.assertIsInstance(arch, nx.DiGraph)

if __name__ == '__main__':
    unittest.main()
