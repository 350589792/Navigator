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
    from flearn.models.rgnn.data_load import create_federated_data
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
            
            # Network size configurations
            self.n_users_small = args.n_users_small
            self.n_uavs_small = args.n_uavs_small
            self.n_users_medium = args.n_users_medium
            self.n_uavs_medium = args.n_uavs_medium
            self.n_users_large = args.n_users_large
            self.n_uavs_large = args.n_uavs_large
            
            # UAV Resource Configuration
            self.uav_compute_speed = args.uav_compute_speed
            self.uav_bandwidth = args.uav_bandwidth
            self.max_relay_distance = args.max_relay_distance
            
            # Additional parameters
            self.seed = args.seed
            self.eval_interval = args.eval_interval
            self.client_sample_ratio = args.client_sample_ratio
    
    model_config = ModelConfig(args, n_users, n_uavs)
    
    # Define scenarios to run
    scenarios = [
        {
            'name': 'direct',
            'description': '直接传输 (Direct Transmission)',
            'config': {'use_relay': False, 'use_resource_opt': False}
        },
        {
            'name': 'resource_opt',
            'description': '资源优化 (Resource Optimization)',
            'config': {'use_relay': False, 'use_resource_opt': True}
        },
        {
            'name': 'feduavgnn',
            'description': 'FedUAVGNN',
            'config': {'use_relay': True, 'use_resource_opt': True}
        },
        {
            'name': 'bellman_ford',
            'description': 'Bellman-Ford路径选择',
            'config': {'use_relay': True, 'use_resource_opt': False}
        }
    ]
    
    all_metrics = {}
    for scenario in scenarios:
        logger.info(f"\nRunning scenario: {scenario['description']}")
        
        # Initialize server with scenario-specific configuration
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
            num_users=n_uavs,
            rho=1.0,
            times=1,
            hidden_dim=model_config.hidden_dim,
            use_relay=scenario['config']['use_relay'],
            use_resource_opt=scenario['config']['use_resource_opt']
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
        'scenario': scenario['name'],
        'description': scenario['description'],
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
        'convergence_round': len(server.rs_train_loss) if server.rs_train_loss else None,
        'use_relay': scenario['config']['use_relay'],
        'use_resource_opt': scenario['config']['use_resource_opt'],
        'total_communication_overhead': sum(server.communication_overhead) / (1024 * 1024) if server.communication_overhead else 0  # Convert to MB
    }
    
    # Store metrics for this scenario
    all_metrics[scenario['name']] = metrics
    
    return all_metrics

def plot_results(results, save_dir):
    """Plot and save simulation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    scenarios = sorted(results.keys())
    network_sizes = sorted(list(results[scenarios[0]].keys()))
    metrics = {
        'total_time': 'training_time',
        'total_overhead': 'total_communication_overhead',
        'final_loss': 'final_loss',
        'final_accuracy': 'final_accuracy'
    }
    
    # Create plots
    for metric_name, metric_key in metrics.items():
        plt.figure(figsize=(12, 6))
        for scenario in scenarios:
            values = [results[scenario][size][metric_key] for size in network_sizes]
            plt.plot(network_sizes, values, 'o-', label=results[scenario][network_sizes[0]]['description'])
        
        plt.title(f'{metric_name.replace("_", " ").title()} 对比', fontproperties='Noto Sans CJK JP')
        plt.xlabel('网络规模', fontproperties='Noto Sans CJK JP')
        plt.ylabel(metric_name.replace('_', ' ').title(), fontproperties='Noto Sans CJK JP')
        plt.xticks(rotation=45)
        plt.legend(prop={'family': 'Noto Sans CJK JP'})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric_name}.png'))
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
        
        # Run simulation and collect metrics for all scenarios
        metrics = run_simulation(n_users, n_uavs, args)
        
        # Save scenario metrics
        for scenario_name, scenario_data in metrics.items():
            scenario_dir = os.path.join(network_dir, f'scenario_{scenario_name}')
            os.makedirs(scenario_dir, exist_ok=True)
            
            # Save metrics to JSON
            metrics_file = os.path.join(scenario_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(scenario_data, f, indent=4)
            
            # Store results by scenario for plotting
            if scenario_name not in results:
                results[scenario_name] = {}
            results[scenario_name][f"{size}_{n_uavs}"] = scenario_data
            
        # Generate scenario comparison plots
        from analyze_metrics import plot_scenario_comparison
        plot_scenario_comparison(metrics, network_dir)
        
        # Plot training history for each scenario
        for scenario_name, scenario_data in metrics.items():
            scenario_dir = os.path.join(network_dir, f'scenario_{scenario_name}')
            
            # Plot loss history
            plt.figure(figsize=(10, 6))
            plt.plot(scenario_data['train_loss_history'], label='训练损失')
            plt.plot(scenario_data['test_loss_history'], label='测试损失')
            plt.title(f'{size.title()} 网络 - {scenario_data["description"]} - 损失历史', fontproperties='Noto Sans CJK JP')
            plt.xlabel('轮次', fontproperties='Noto Sans CJK JP')
            plt.ylabel('损失', fontproperties='Noto Sans CJK JP')
            plt.legend(prop={'family': 'Noto Sans CJK JP'})
            plt.grid(True)
            plt.savefig(os.path.join(scenario_dir, 'loss_history.png'))
            plt.close()

            # Plot accuracy history
            plt.figure(figsize=(10, 6))
            plt.plot(scenario_data['train_acc_history'], label='训练准确率')
            plt.plot(scenario_data['test_acc_history'], label='测试准确率')
            plt.title(f'{size.title()} 网络 - {scenario_data["description"]} - 准确率历史', fontproperties='Noto Sans CJK JP')
            plt.xlabel('轮次', fontproperties='Noto Sans CJK JP')
            plt.ylabel('准确率', fontproperties='Noto Sans CJK JP')
            plt.legend(prop={'family': 'Noto Sans CJK JP'})
            plt.grid(True)
            plt.savefig(os.path.join(scenario_dir, 'accuracy_history.png'))
            plt.close()

            # Plot combined metrics
            plt.figure(figsize=(15, 10))
            
            # Communication overhead plot
            plt.subplot(2, 1, 1)
            plt.plot(scenario_data['communication_overhead'])
            plt.title(f'{size.title()} 网络 - {scenario_data["description"]} - 通信开销', fontproperties='Noto Sans CJK JP')
            plt.xlabel('轮次', fontproperties='Noto Sans CJK JP')
            plt.ylabel('字节', fontproperties='Noto Sans CJK JP')
            plt.grid(True)
            
            # Training time plot
            plt.subplot(2, 1, 2)
            plt.plot(scenario_data['training_times'])
            plt.title(f'{size.title()} 网络 - {scenario_data["description"]} - 训练时间', fontproperties='Noto Sans CJK JP')
            plt.xlabel('轮次', fontproperties='Noto Sans CJK JP')
            plt.ylabel('秒', fontproperties='Noto Sans CJK JP')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(scenario_dir, 'performance_metrics.png'))
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
