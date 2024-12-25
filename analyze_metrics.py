import json
import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def analyze_network_metrics(network_type: str) -> Dict:
    """Analyze metrics for a specific network configuration."""
    log_dirs = glob.glob(f'./logs/{network_type}_network_*')
    if not log_dirs:
        return None
    
    latest_dir = max(log_dirs, key=os.path.getctime)
    sim_dir = max(glob.glob(os.path.join(latest_dir, 'simulation_*')), key=os.path.getctime)
    metrics_file = os.path.join(sim_dir, f'network_{network_type}', 'metrics.json')
    
    if not os.path.exists(metrics_file):
        return None
        
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
        # Extract metrics
        train_losses = data.get('train_loss_history', [])
        train_accuracies = data.get('train_acc_history', [])
        test_losses = data.get('test_loss_history', [])
        test_accuracies = data.get('test_acc_history', [])
        
        # Calculate convergence rate (loss decrease per round)
        convergence_rate = (train_losses[0] - train_losses[-1]) / len(train_losses) if len(train_losses) > 1 else 0
        
        # Convert communication overhead from bytes to MB
        comm_overhead = sum(overhead / (1024 * 1024) for overhead in data.get('communication_overhead', []))
        
        # Get training time
        training_time = data.get('training_time', 0)
        
        return {
            'convergence_rate': convergence_rate,
            'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0,
            'final_test_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'total_communication_overhead': comm_overhead,
            'training_time': training_time,
            'train_loss_history': train_losses,
            'train_acc_history': train_accuracies,
            'test_loss_history': test_losses,
            'test_acc_history': test_accuracies,
            'memory_usage': data.get('memory_usage', 0),
            'num_users': data.get('num_users', 0),
            'num_uavs': data.get('num_uavs', 0)
        }

def plot_metrics(small_metrics: Dict, medium_metrics: Dict, large_metrics: Dict):
    """Plot comparison graphs for different network sizes."""
    plt.figure(figsize=(15, 12))
    
    # Plot training loss convergence
    plt.subplot(3, 2, 1)
    plt.plot(small_metrics['train_loss_history'], label=f'Small ({small_metrics["num_users"]} users, {small_metrics["num_uavs"]} UAVs)')
    plt.plot(medium_metrics['train_loss_history'], label=f'Medium ({medium_metrics["num_users"]} users, {medium_metrics["num_uavs"]} UAVs)')
    plt.plot(large_metrics['train_loss_history'], label=f'Large ({large_metrics["num_users"]} users, {large_metrics["num_uavs"]} UAVs)')
    plt.title('Training Loss Convergence')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test loss convergence
    plt.subplot(3, 2, 2)
    plt.plot(small_metrics['test_loss_history'], label=f'Small ({small_metrics["num_users"]} users, {small_metrics["num_uavs"]} UAVs)')
    plt.plot(medium_metrics['test_loss_history'], label=f'Medium ({medium_metrics["num_users"]} users, {medium_metrics["num_uavs"]} UAVs)')
    plt.plot(large_metrics['test_loss_history'], label=f'Large ({large_metrics["num_users"]} users, {large_metrics["num_uavs"]} UAVs)')
    plt.title('Test Loss Convergence')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(3, 2, 3)
    plt.plot(small_metrics['train_acc_history'], label=f'Small ({small_metrics["num_users"]} users, {small_metrics["num_uavs"]} UAVs)')
    plt.plot(medium_metrics['train_acc_history'], label=f'Medium ({medium_metrics["num_users"]} users, {medium_metrics["num_uavs"]} UAVs)')
    plt.plot(large_metrics['train_acc_history'], label=f'Large ({large_metrics["num_users"]} users, {large_metrics["num_uavs"]} UAVs)')
    plt.title('Training Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(3, 2, 4)
    plt.plot(small_metrics['test_acc_history'], label=f'Small ({small_metrics["num_users"]} users, {small_metrics["num_uavs"]} UAVs)')
    plt.plot(medium_metrics['test_acc_history'], label=f'Medium ({medium_metrics["num_users"]} users, {medium_metrics["num_uavs"]} UAVs)')
    plt.plot(large_metrics['test_acc_history'], label=f'Large ({large_metrics["num_users"]} users, {large_metrics["num_uavs"]} UAVs)')
    plt.title('Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot communication overhead
    plt.subplot(3, 2, 5)
    networks = ['Small', 'Medium', 'Large']
    overhead = [small_metrics['total_communication_overhead'],
                medium_metrics['total_communication_overhead'],
                large_metrics['total_communication_overhead']]
    plt.bar(networks, overhead)
    plt.title('Total Communication Overhead')
    plt.ylabel('MB')
    
    # Plot resource usage
    plt.subplot(3, 2, 6)
    memory = [small_metrics['memory_usage'],
              medium_metrics['memory_usage'],
              large_metrics['memory_usage']]
    times = [small_metrics['training_time'],
             medium_metrics['training_time'],
             large_metrics['training_time']]
    
    x = np.arange(len(networks))
    width = 0.35
    
    plt.bar(x - width/2, memory, width, label='Memory (MB)')
    plt.bar(x + width/2, times, width, label='Time (s)')
    plt.xticks(x, networks)
    plt.title('Resource Usage')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('network_comparison.png')

def main():
    network_types = ['small', 'medium', 'large']
    results = {}
    
    print("=== UAV Network Federation Analysis ===\n")
    
    for network_type in network_types:
        metrics = analyze_network_metrics(network_type)
        if metrics:
            results[network_type] = metrics
            print(f"\n=== {network_type.capitalize()} Network Configuration ===")
            print(f"Network Size: {metrics['num_users']} users, {metrics['num_uavs']} UAVs")
            print(f"Convergence Rate (Loss/round): {metrics['convergence_rate']:.6f}")
            print(f"Final Training Accuracy: {metrics['final_train_accuracy']:.4f}")
            print(f"Final Test Accuracy: {metrics['final_test_accuracy']:.4f}")
            print(f"Total Communication Overhead: {metrics['total_communication_overhead']:.2f} MB")
            print(f"Training Time: {metrics['training_time']:.3f}s")
            print(f"Memory Usage: {metrics['memory_usage']:.2f} MB")
            print("\nTraining Loss History:", [f"{loss:.6f}" for loss in metrics['train_loss_history'][:5]], "...")
            print("Test Loss History:", [f"{loss:.6f}" for loss in metrics['test_loss_history'][:5]], "...")
    
    if len(results) == 3:
        plot_metrics(results['small'], results['medium'], results['large'])
        print("\nMetrics visualization saved as 'network_comparison.png'")

if __name__ == "__main__":
    main()
