import os
import csv
import json
import time

class MetricsLogger:
    def __init__(self, log_dir, prefix):
        """Initialize metrics logger with directory and file prefix."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create CSV files for different metrics
        self.training_file = os.path.join(log_dir, f"{prefix}_training.csv")
        self.comm_file = os.path.join(log_dir, f"{prefix}_communication.csv")
        self.resource_file = os.path.join(log_dir, f"{prefix}_resources.csv")
        self.metadata_file = os.path.join(log_dir, f"{prefix}_metadata.json")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        # Store metadata
        self.metadata = {
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'log_dir': log_dir,
            'prefix': prefix
        }
        
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Training metrics
        with open(self.training_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Train_Loss', 'Train_Accuracy', 'Test_Loss', 'Test_Accuracy'])
            
        # Communication metrics
        with open(self.comm_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Overhead_MB', 'Time_Seconds'])
            
        # Resource metrics
        with open(self.resource_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Time_Seconds', 'Memory_MB'])
            
    def log_training_metrics(self, round_num, train_loss, train_acc, test_loss, test_acc):
        """Log training metrics for a round."""
        with open(self.training_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, train_loss, train_acc, test_loss, test_acc])
            
    def log_communication_metrics(self, round_num, overhead_mb, time_seconds):
        """Log communication metrics for a round."""
        with open(self.comm_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, overhead_mb, time_seconds])
            
    def log_resource_metrics(self, round_num, time_seconds, memory_mb):
        """Log resource usage metrics for a round."""
        with open(self.resource_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num, time_seconds, memory_mb])
            
    def save_metadata(self):
        """Save metadata and final statistics."""
        self.metadata['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
