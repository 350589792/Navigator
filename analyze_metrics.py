import json
import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

def analyze_network_metrics(network_type: str) -> Dict:
    """Analyze metrics for a specific network configuration."""
    # Find the latest simulation directory
    sim_dirs = glob.glob('./logs/simulation_*')
    if not sim_dirs:
        return None
    
    latest_sim_dir = max(sim_dirs, key=os.path.getctime)
    metrics_file = os.path.join(latest_sim_dir, f'network_{network_type}', 'metrics.json')
    
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

def plot_scenario_comparison(metrics_by_scenario: Dict, save_dir: str):
    """Plot comparison graphs for different scenarios with Chinese labels."""
    plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('场景对比分析 (Scenario Comparison)', fontsize=16)
    
    # Plot training loss
    ax = axes[0, 0]
    for scenario, metrics in metrics_by_scenario.items():
        ax.plot(metrics['train_loss_history'], label=metrics['description'])
    ax.set_title('训练损失 (Training Loss)')
    ax.set_xlabel('轮次 (Round)')
    ax.set_ylabel('损失值 (Loss)')
    ax.legend()
    ax.grid(True)
    
    # Plot communication overhead
    ax = axes[0, 1]
    scenarios = list(metrics_by_scenario.keys())
    overhead = [metrics_by_scenario[s]['total_communication_overhead'] for s in scenarios]
    ax.bar(scenarios, overhead)
    ax.set_title('通信开销 (Communication Overhead)')
    ax.set_xlabel('场景 (Scenario)')
    ax.set_ylabel('开销 (MB)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True)
    
    # Plot training time
    ax = axes[1, 0]
    times = [metrics_by_scenario[s]['training_time'] for s in scenarios]
    ax.bar(scenarios, times)
    ax.set_title('训练时间 (Training Time)')
    ax.set_xlabel('场景 (Scenario)')
    ax.set_ylabel('时间 (秒)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True)
    
    # Plot final accuracy
    ax = axes[1, 1]
    accuracy = [metrics_by_scenario[s]['final_accuracy'] for s in scenarios]
    ax.bar(scenarios, accuracy)
    ax.set_title('最终准确率 (Final Accuracy)')
    ax.set_xlabel('场景 (Scenario)')
    ax.set_ylabel('准确率 (Accuracy)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scenario_comparison.png'))
    plt.close()

def plot_metrics(small_metrics: Dict, medium_metrics: Dict, large_metrics: Dict):
    """Plot detailed comparison graphs for different network sizes with Chinese labels."""
    # Configure font settings for Chinese characters
    plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set font properties
    title_font = {'family': 'WenQuanYi Zen Hei', 'size': 14, 'weight': 'bold'}
    label_font = {'family': 'WenQuanYi Zen Hei', 'size': 12}
    legend_font = {'family': 'WenQuanYi Zen Hei', 'size': 10}
    
    # Set up the main figure with 3x3 subplots
    plt.figure(figsize=(20, 20))
    
    # 1. Training Loss Convergence (训练损失收敛)
    plt.subplot(3, 3, 1)
    plt.plot(small_metrics['train_loss_history'], label=f'小型 ({small_metrics["num_users"]}用户, {small_metrics["num_uavs"]}UAV)')
    plt.plot(medium_metrics['train_loss_history'], label=f'中型 ({medium_metrics["num_users"]}用户, {medium_metrics["num_uavs"]}UAV)')
    plt.plot(large_metrics['train_loss_history'], label=f'大型 ({large_metrics["num_users"]}用户, {large_metrics["num_uavs"]}UAV)')
    plt.title('训练损失收敛曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 2. Test Loss Convergence (测试损失收敛)
    plt.subplot(3, 3, 2)
    plt.plot(small_metrics['test_loss_history'], label=f'小型 ({small_metrics["num_users"]}用户, {small_metrics["num_uavs"]}UAV)')
    plt.plot(medium_metrics['test_loss_history'], label=f'中型 ({medium_metrics["num_users"]}用户, {medium_metrics["num_uavs"]}UAV)')
    plt.plot(large_metrics['test_loss_history'], label=f'大型 ({large_metrics["num_users"]}用户, {large_metrics["num_uavs"]}UAV)')
    plt.title('测试损失收敛曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 3. Training Accuracy (训练准确率)
    plt.subplot(3, 3, 3)
    plt.plot(small_metrics['train_acc_history'], label=f'小型 ({small_metrics["num_users"]}用户, {small_metrics["num_uavs"]}UAV)')
    plt.plot(medium_metrics['train_acc_history'], label=f'中型 ({medium_metrics["num_users"]}用户, {medium_metrics["num_uavs"]}UAV)')
    plt.plot(large_metrics['train_acc_history'], label=f'大型 ({large_metrics["num_users"]}用户, {large_metrics["num_uavs"]}UAV)')
    plt.title('训练准确率进展')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 4. Test Accuracy (测试准确率)
    plt.subplot(3, 3, 4)
    plt.plot(small_metrics['test_acc_history'], label=f'小型 ({small_metrics["num_users"]}用户, {small_metrics["num_uavs"]}UAV)')
    plt.plot(medium_metrics['test_acc_history'], label=f'中型 ({medium_metrics["num_users"]}用户, {medium_metrics["num_uavs"]}UAV)')
    plt.plot(large_metrics['test_acc_history'], label=f'大型 ({large_metrics["num_users"]}用户, {large_metrics["num_uavs"]}UAV)')
    plt.title('测试准确率进展')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 5. Communication Overhead Over Time (通信开销随时间变化)
    plt.subplot(3, 3, 5)
    networks = ['小型网络', '中型网络', '大型网络']
    overhead = [small_metrics['total_communication_overhead'],
                medium_metrics['total_communication_overhead'],
                large_metrics['total_communication_overhead']]
    plt.bar(networks, overhead)
    plt.title('总通信开销')
    plt.ylabel('通信量 (MB)')
    plt.grid(True)
    
    # 6. Resource Usage (资源使用)
    plt.subplot(3, 3, 6)
    x = np.arange(len(networks))
    width = 0.35
    memory = [small_metrics['memory_usage'],
              medium_metrics['memory_usage'],
              large_metrics['memory_usage']]
    times = [small_metrics['training_time'],
             medium_metrics['training_time'],
             large_metrics['training_time']]
    
    plt.bar(x - width/2, memory, width, label='内存使用 (MB)')
    plt.bar(x + width/2, times, width, label='训练时间 (秒)')
    plt.xticks(x, networks)
    plt.title('资源使用情况')
    plt.legend()
    plt.grid(True)
    
    # 7. Convergence Rate Comparison (收敛速率对比)
    plt.subplot(3, 3, 7)
    conv_rates = [
        (small_metrics['train_loss_history'][0] - small_metrics['train_loss_history'][-1]) / len(small_metrics['train_loss_history']),
        (medium_metrics['train_loss_history'][0] - medium_metrics['train_loss_history'][-1]) / len(medium_metrics['train_loss_history']),
        (large_metrics['train_loss_history'][0] - large_metrics['train_loss_history'][-1]) / len(large_metrics['train_loss_history'])
    ]
    plt.bar(networks, conv_rates)
    plt.title('收敛速率对比')
    plt.ylabel('每轮损失下降')
    plt.grid(True)
    
    # 8. Final Performance Comparison (最终性能对比)
    plt.subplot(3, 3, 8)
    final_acc = [
        small_metrics['test_acc_history'][-1],
        medium_metrics['test_acc_history'][-1],
        large_metrics['test_acc_history'][-1]
    ]
    plt.bar(networks, final_acc)
    plt.title('最终测试准确率')
    plt.ylabel('准确率')
    plt.grid(True)
    
    # 9. Efficiency Metrics (效率指标)
    plt.subplot(3, 3, 9)
    efficiency = [
        small_metrics['test_acc_history'][-1] / small_metrics['training_time'],
        medium_metrics['test_acc_history'][-1] / medium_metrics['training_time'],
        large_metrics['test_acc_history'][-1] / large_metrics['training_time']
    ]
    plt.bar(networks, efficiency)
    plt.title('训练效率')
    plt.ylabel('准确率/训练时间')
    plt.grid(True)
    
    # Adjust layout and save
    plt.suptitle('联邦学习性能指标综合分析', fontsize=16, y=0.95, fontfamily='WenQuanYi Zen Hei', fontweight='bold')
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=300, bbox_inches='tight')

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
        print("\n详细的性能指标可视化已保存为 'network_comparison.png'")

if __name__ == "__main__":
    main()
