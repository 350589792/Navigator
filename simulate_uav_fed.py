#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import psutil
import logging
from datetime import datetime
from pathlib import Path
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Import federated learning components
try:
    from flearn.servers.serverfedl import FEDL
    from flearn.models.models import FedUAVGNN
    from flearn.data.data_load_new import create_federated_data
    from flearn.utils.metrics_logger import MetricsLogger
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    raise

def run_simulation(n_users, n_uavs, args):
    """Run federated learning simulation with specified number of UAVs."""
    logger.info(f"\nStarting simulation with {n_uavs} UAVs and {n_users} users")
    logger.info("Configuration:")
    logger.info(f"- Network: {n_users} users, {n_uavs} UAVs")
    logger.info(f"- Training: batch_size={args.batch_size}, learning_rate={args.learning_rate}")
    logger.info(f"- Rounds: {args.num_rounds} global rounds, {args.local_epochs} local epochs")
    logger.info(f"- Model: hidden_dim={args.hidden_dim}, alpha={args.alpha}")
    logger.info(f"- Device: {args.device}")
    logger.info(f"- Client Sample Ratio: {args.client_sample_ratio}")
    logger.info(f"- Evaluation Interval: {args.eval_interval}")
    logger.info(f"- Random Seed: {args.seed}")
    
    # Create model configuration
    class ModelConfig:
        def __init__(self, args, n_users, n_uavs):
            # Training parameters
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.device = args.device
            self.hidden_dim = args.hidden_dim
            self.alpha = args.alpha
            self.train_num = args.train_num
            self.num_rounds = args.num_rounds
            self.local_epochs = args.local_epochs
            self.epochs = args.epochs
            self.hyper_learning_rate = args.hyper_learning_rate
            self.L = args.L
            
            # Network configuration
            self.n_users = n_users
            self.n_uavs = n_uavs
            
            # Additional parameters
            self.seed = args.seed
            self.eval_interval = args.eval_interval
            self.client_sample_ratio = args.client_sample_ratio
    
    model_config = ModelConfig(args, n_users, n_uavs)
    
    # Get federated data and initialize server
    uav_data = create_federated_data(n_users, n_uavs)
    
    # Split data into train/test sets for each UAV (80/20 split)
    train_data = []
    test_data = []
    for uav_features, uav_edges in uav_data:
        n_samples = uav_edges.shape[1]
        n_train = int(0.8 * n_samples)
        
        # Randomly shuffle indices
        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # Split features and edges
        train_data.append((
            uav_features,
            uav_edges[:, train_idx]
        ))
        test_data.append((
            uav_features,
            uav_edges[:, test_idx]
        ))
    
    # Initialize server with configuration
    server = FEDL(
        dataset="uav_network",
        algorithm="fedl",
        model_config=model_config,
        batch_size=model_config.batch_size,
        learning_rate=model_config.learning_rate,
        hyper_learning_rate=model_config.hyper_learning_rate,
        L=model_config.L,
        num_glob_iters=model_config.num_rounds,
        local_epochs=model_config.local_epochs,
        optimizer="fedl",
        num_users=model_config.n_users,
        rho=1.0,
        times=1,
        hidden_dim=model_config.hidden_dim,
        train_data=train_data,
        test_data=test_data
    )
    
    # Record start time and initial resource usage
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Train the model using server's built-in training method
    server.train()
    
    # Calculate final metrics
    total_time = time.time() - start_time
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    metrics = {
        'training_time': total_time,
        'memory_usage': final_memory - initial_memory,
        'num_users': n_users,
        'num_uavs': n_uavs,
        'train_loss_history': server.rs_train_loss,
        'train_acc_history': server.rs_train_acc,
        'test_loss_history': server.rs_test_loss,
        'test_acc_history': server.rs_test_acc,
        'communication_overhead': server.communication_overhead,
        'training_times': server.training_times,
        'final_loss': server.rs_train_loss[-1] if server.rs_train_loss else None,
        'final_accuracy': server.rs_train_acc[-1] if server.rs_train_acc else None,
        'convergence_round': len(server.rs_train_loss) if server.rs_train_loss else None
    }
    
    return metrics

def plot_results(results, save_dir):
    """Plot and save simulation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    uav_counts = sorted(results.keys())
    metrics = ['training_time', 'communication_overhead', 'final_loss', 'convergence_round']
    
    # Create plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [results[n][metric] for n in uav_counts]
        plt.plot(uav_counts, values, 'o-')
        plt.title(f'{metric.replace("_", " ").title()} vs Number of UAVs')
        plt.xlabel('Number of UAVs')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

def parse_arguments():
    """Parse command line arguments."""
    try:
        import os
        import sys
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import and use the argument parser from gnn_fed_config_new
        from gnn_fed_config_new import get_default_args
        parser = get_default_args()
        args = parser.parse_args()
        print("\nParsed arguments:", vars(args))
        return args
    except ImportError as e:
        print(f"Error importing configuration: {e}")
        print(f"Python path: {sys.path}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def main():
    """Main function for UAV Network Federated Learning Simulation."""
    print("\nStarting UAV Network Federated Learning Simulation")
    
    # Get configuration
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create required directories
    try:
        os.makedirs(os.path.dirname(args.path_model), exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        print("\nDirectories created successfully")
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        raise
        
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.log_dir, f'simulation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'simulation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log configuration
    logging.info("Configuration:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    
    # Run simulations for different network sizes
    results = {}
    network_configs = [
        (args.n_users_small, args.n_uavs_small, "small"),
        (args.n_users_medium, args.n_uavs_medium, "medium"),
        (args.n_users_large, args.n_uavs_large, "large")
    ]
    
    for n_users, n_uavs, size in network_configs:
        network_dir = os.path.join(results_dir, f'network_{size}')
        os.makedirs(network_dir, exist_ok=True)
        
        logging.info(f"\nRunning simulation for {size} network:")
        logging.info(f"Users: {n_users}, UAVs: {n_uavs}")
        
        # Run simulation and collect metrics
        metrics = run_simulation(n_users, n_uavs, args)
        results[f"{size}_{n_uavs}"] = metrics
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(metrics['train_loss_history'], label='Train')
        plt.plot(metrics['test_loss_history'], label='Test')
        plt.title(f'{size.title()} Network - Loss History')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(metrics['train_acc_history'], label='Train')
        plt.plot(metrics['test_acc_history'], label='Test')
        plt.title(f'{size.title()} Network - Accuracy History')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Communication overhead plot
        plt.subplot(2, 2, 3)
        plt.plot(metrics['communication_overhead'])
        plt.title(f'{size.title()} Network - Communication Overhead')
        plt.xlabel('Round')
        plt.ylabel('Bytes')
        plt.grid(True)
        
        # Training time plot
        plt.subplot(2, 2, 4)
        plt.plot(metrics['training_times'])
        plt.title(f'{size.title()} Network - Training Time')
        plt.xlabel('Round')
        plt.ylabel('Seconds')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(network_dir, 'training_history.png'))
        plt.close()
        
        # Save network metrics
        with open(os.path.join(network_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Network {size} results saved to {network_dir}")
    
    # Plot comparison results
    plot_results(results, results_dir)
    
    # Save overall results
    with open(os.path.join(results_dir, 'overall_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"\nSimulation complete. All results saved to {results_dir}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\nError occurred:")
        print(e)
        import traceback
        traceback.print_exc()
