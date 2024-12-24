#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
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
    print(f"\nStarting simulation with {n_uavs} UAVs and {n_users} users")
    print(f"Configuration: batch_size={args.batch_size}, learning_rate={args.learning_rate}, epochs={args.epochs}")
    
    # Record start time and initial resource usage
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
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
            
            # Network configuration
            self.num_users = n_users
            self.num_uavs = n_uavs
            
            # Paths
            self.checkpoint_dir = args.checkpoint_dir
            self.log_dir = args.log_dir
            self.path_model = args.path_model
    
    model_config = ModelConfig(args, n_users, n_uavs)
    
    # Initialize server with configuration
    server = FEDL(
        dataset="uav_network",
        algorithm="fedl",
        model_config=model_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hyper_learning_rate=0.01,
        L=0.1,
        num_glob_iters=args.num_rounds,
        local_epochs=args.local_epochs,
        optimizer="fedl",
        num_users=n_users,  # Total number of users in the network
        num_uavs=n_uavs,   # Number of UAV nodes
        rho=1.0,
        times=1,
        hidden_dim=args.hidden_dim,
        train_num=args.train_num,
        alpha=args.alpha,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
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
    parser = argparse.ArgumentParser(description='UAV Network Federated Learning Simulation')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_num', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--path_model', type=str, default='./model/rgnn_fed.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    # Network configuration
    parser.add_argument('--n_users_small', type=int, default=10)
    parser.add_argument('--n_uavs_small', type=int, default=2)
    parser.add_argument('--n_users_medium', type=int, default=20)
    parser.add_argument('--n_uavs_medium', type=int, default=5)
    parser.add_argument('--n_users_large', type=int, default=50)
    parser.add_argument('--n_uavs_large', type=int, default=10)
    
    # Parse known args only
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")
    
    return args

def main():
    """Main function for UAV Network Federated Learning Simulation."""
    print("\nStarting UAV Network Federated Learning Simulation")
    
    # Get configuration
    args = parse_arguments()
    
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'simulation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log configuration
    logging.info("Configuration:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    
    # Create results directory
    results_dir = os.path.join(args.log_dir, f'simulation_{int(time.time())}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run simulations for different UAV counts
    results = {}
    uav_configs = [
        (args.n_users_small, args.n_uavs_small),
        (args.n_users_medium, args.n_uavs_medium),
        (args.n_users_large, args.n_uavs_large)
    ]
    
    for n_users, n_uavs in uav_configs:
        results[n_uavs] = run_simulation(n_users, n_uavs, args)
    
    # Plot results
    plot_results(results, results_dir)
    
    # Save raw results
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSimulation complete. Results saved to {results_dir}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\nError occurred:")
        print(e)
        import traceback
        traceback.print_exc()
