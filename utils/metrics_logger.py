import os
import csv
import time
from datetime import datetime
import json

class MetricsLogger:
    """Utility class for logging federated learning metrics."""
    
    def __init__(self, log_dir, experiment_name=None):
        """Initialize the metrics logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Optional name for the experiment
        """
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize CSV files
        self.files = {
            'training': self._init_csv('training_metrics.csv', 
                ['timestamp', 'round', 'train_loss', 'train_acc', 'test_loss', 'test_acc']),
            'communication': self._init_csv('communication_metrics.csv',
                ['timestamp', 'round', 'param_size_mb', 'transfer_time']),
            'resource': self._init_csv('resource_metrics.csv',
                ['timestamp', 'round', 'runtime_seconds', 'memory_mb'])
        }
        
        # Keep track of experiment metadata
        self.metadata = {
            'start_time': time.time(),
            'experiment_name': experiment_name,
            'metrics_logged': []
        }
    
    def _init_csv(self, filename, headers):
        """Initialize a CSV file with headers."""
        filepath = os.path.join(self.experiment_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return filepath
    
    def log_training_metrics(self, round_num, train_loss, train_acc, test_loss, test_acc):
        """Log training metrics for a round."""
        with open(self.files['training'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), round_num, train_loss, train_acc, test_loss, test_acc])
        self.metadata['metrics_logged'].append(f'training_round_{round_num}')
    
    def log_communication_metrics(self, round_num, param_size_mb, transfer_time):
        """Log communication overhead metrics."""
        with open(self.files['communication'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), round_num, param_size_mb, transfer_time])
        self.metadata['metrics_logged'].append(f'communication_round_{round_num}')
    
    def log_resource_metrics(self, round_num, runtime_seconds, memory_mb):
        """Log resource usage metrics."""
        with open(self.files['resource'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), round_num, runtime_seconds, memory_mb])
        self.metadata['metrics_logged'].append(f'resource_round_{round_num}')
    
    def save_metadata(self):
        """Save experiment metadata."""
        self.metadata['end_time'] = time.time()
        self.metadata['total_duration'] = self.metadata['end_time'] - self.metadata['start_time']
        
        metadata_file = os.path.join(self.experiment_dir, 'experiment_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def get_experiment_dir(self):
        """Get the experiment directory path."""
        return self.experiment_dir
