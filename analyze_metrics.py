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
        scenarios = json.load(f)
        
    metrics_by_scenario = {}
    for scenario, data in scenarios.items():
        # Extract metrics for each scenario
        train_losses = data.get('train_loss_history', [])
        train_accuracies = data.get('train_acc_history', [])
        test_losses = data.get('test_loss_history', [])
        test_accuracies = data.get('test_acc_history', [])
        
        # Calculate convergence rate (loss decrease per round)
        convergence_rate = (train_losses[0] - train_losses[-1]) / len(train_losses) if len(train_losses) > 1 else 0
        
        # Convert communication overhead from bytes to MB
        comm_overhead = data.get('total_communication_overhead', 0)
        
        metrics_by_scenario[scenario] = {
            'description': data.get('description', scenario),
            'convergence_rate': convergence_rate,
            'final_accuracy': data.get('final_accuracy', 0),
            'final_test_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'total_communication_overhead': comm_overhead,
            'training_time': data.get('training_time', 0),
            'train_loss_history': train_losses,
            'train_acc_history': train_accuracies,
            'test_loss_history': test_losses,
            'test_acc_history': test_accuracies,
            'memory_usage': data.get('memory_usage', 0),
            'num_users': data.get('num_users', 0),
            'num_uavs': data.get('num_uavs', 0),
            'use_relay': data.get('use_relay', False),
            'use_resource_opt': data.get('use_resource_opt', False)
        }
        
    return metrics_by_scenario

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
    
    scenarios = list(small_metrics.keys())
    
    # Create subplots for each metric type
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Training Loss Convergence (训练损失收敛)
    plt.subplot(3, 3, 1)
    for scenario in scenarios:
        s_metrics = small_metrics[scenario]
        m_metrics = medium_metrics[scenario]
        l_metrics = large_metrics[scenario]
        
        plt.plot(s_metrics['train_loss_history'], 
                label=f'{scenario}-小型({s_metrics["num_users"]}用户)')
        plt.plot(m_metrics['train_loss_history'], 
                label=f'{scenario}-中型({m_metrics["num_users"]}用户)')
        plt.plot(l_metrics['train_loss_history'], 
                label=f'{scenario}-大型({l_metrics["num_users"]}用户)')
    plt.title('训练损失收敛曲线', fontdict=title_font)
    plt.xlabel('训练轮次', fontdict=label_font)
    plt.ylabel('损失值', fontdict=label_font)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 2. Test Loss Convergence (测试损失收敛)
    plt.subplot(3, 3, 2)
    for scenario in scenarios:
        s_metrics = small_metrics[scenario]
        m_metrics = medium_metrics[scenario]
        l_metrics = large_metrics[scenario]
        
        plt.plot(s_metrics['test_loss_history'], 
                label=f'{scenario}-小型({s_metrics["num_users"]}用户)')
        plt.plot(m_metrics['test_loss_history'], 
                label=f'{scenario}-中型({m_metrics["num_users"]}用户)')
        plt.plot(l_metrics['test_loss_history'], 
                label=f'{scenario}-大型({l_metrics["num_users"]}用户)')
    plt.title('测试损失收敛曲线', fontdict=title_font)
    plt.xlabel('训练轮次', fontdict=label_font)
    plt.ylabel('损失值', fontdict=label_font)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 3. Training Accuracy (训练准确率)
    plt.subplot(3, 3, 3)
    for scenario in scenarios:
        s_metrics = small_metrics[scenario]
        m_metrics = medium_metrics[scenario]
        l_metrics = large_metrics[scenario]
        
        plt.plot(s_metrics['train_acc_history'], 
                label=f'{scenario}-小型({s_metrics["num_users"]}用户)')
        plt.plot(m_metrics['train_acc_history'], 
                label=f'{scenario}-中型({m_metrics["num_users"]}用户)')
        plt.plot(l_metrics['train_acc_history'], 
                label=f'{scenario}-大型({l_metrics["num_users"]}用户)')
    plt.title('训练准确率进展', fontdict=title_font)
    plt.xlabel('训练轮次', fontdict=label_font)
    plt.ylabel('准确率', fontdict=label_font)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 4. Communication Overhead by Scenario (通信开销对比)
    plt.subplot(3, 3, 4)
    x = np.arange(len(scenarios))
    width = 0.25
    
    small_overhead = [small_metrics[s]['total_communication_overhead'] for s in scenarios]
    medium_overhead = [medium_metrics[s]['total_communication_overhead'] for s in scenarios]
    large_overhead = [large_metrics[s]['total_communication_overhead'] for s in scenarios]
    
    plt.bar(x - width, small_overhead, width, label='小型网络')
    plt.bar(x, medium_overhead, width, label='中型网络')
    plt.bar(x + width, large_overhead, width, label='大型网络')
    
    plt.title('各场景通信开销对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('通信量 (MB)', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 5. Training Time by Scenario (训练时间对比)
    plt.subplot(3, 3, 5)
    small_time = [small_metrics[s]['training_time'] for s in scenarios]
    medium_time = [medium_metrics[s]['training_time'] for s in scenarios]
    large_time = [large_metrics[s]['training_time'] for s in scenarios]
    
    plt.bar(x - width, small_time, width, label='小型网络')
    plt.bar(x, medium_time, width, label='中型网络')
    plt.bar(x + width, large_time, width, label='大型网络')
    
    plt.title('各场景训练时间对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('时间 (秒)', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 6. Memory Usage by Scenario (内存使用对比)
    plt.subplot(3, 3, 6)
    small_mem = [small_metrics[s]['memory_usage'] for s in scenarios]
    medium_mem = [medium_metrics[s]['memory_usage'] for s in scenarios]
    large_mem = [large_metrics[s]['memory_usage'] for s in scenarios]
    
    plt.bar(x - width, small_mem, width, label='小型网络')
    plt.bar(x, medium_mem, width, label='中型网络')
    plt.bar(x + width, large_mem, width, label='大型网络')
    
    plt.title('各场景内存使用对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('内存使用 (MB)', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 7. Convergence Rate by Scenario (收敛速率对比)
    plt.subplot(3, 3, 7)
    small_conv = [small_metrics[s]['convergence_rate'] for s in scenarios]
    medium_conv = [medium_metrics[s]['convergence_rate'] for s in scenarios]
    large_conv = [large_metrics[s]['convergence_rate'] for s in scenarios]
    
    plt.bar(x - width, small_conv, width, label='小型网络')
    plt.bar(x, medium_conv, width, label='中型网络')
    plt.bar(x + width, large_conv, width, label='大型网络')
    
    plt.title('各场景收敛速率对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('每轮损失下降', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 8. Final Accuracy by Scenario (最终准确率对比)
    plt.subplot(3, 3, 8)
    small_acc = [small_metrics[s]['final_accuracy'] for s in scenarios]
    medium_acc = [medium_metrics[s]['final_accuracy'] for s in scenarios]
    large_acc = [large_metrics[s]['final_accuracy'] for s in scenarios]
    
    plt.bar(x - width, small_acc, width, label='小型网络')
    plt.bar(x, medium_acc, width, label='中型网络')
    plt.bar(x + width, large_acc, width, label='大型网络')
    
    plt.title('各场景最终准确率对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('准确率', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # 9. Training Efficiency by Scenario (训练效率对比)
    plt.subplot(3, 3, 9)
    small_eff = [small_metrics[s]['final_accuracy'] / small_metrics[s]['training_time'] 
                 for s in scenarios]
    medium_eff = [medium_metrics[s]['final_accuracy'] / medium_metrics[s]['training_time'] 
                  for s in scenarios]
    large_eff = [large_metrics[s]['final_accuracy'] / large_metrics[s]['training_time'] 
                 for s in scenarios]
    
    plt.bar(x - width, small_eff, width, label='小型网络')
    plt.bar(x, medium_eff, width, label='中型网络')
    plt.bar(x + width, large_eff, width, label='大型网络')
    
    plt.title('各场景训练效率对比', fontdict=title_font)
    plt.xlabel('场景', fontdict=label_font)
    plt.ylabel('准确率/训练时间', fontdict=label_font)
    plt.xticks(x, scenarios)
    plt.legend(prop=legend_font)
    plt.grid(True)
    
    # Adjust layout and save
    plt.suptitle('基于GNN的UAV联邦学习框架性能分析', fontsize=16, y=0.95, fontfamily='WenQuanYi Zen Hei', fontweight='bold')
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create additional plot for GNN-specific metrics
    plt.figure(figsize=(15, 10))
    
    # Plot relay usage comparison
    plt.subplot(2, 2, 1)
    relay_usage = {
        'small': sum(1 for s in scenarios if small_metrics[s]['use_relay']),
        'medium': sum(1 for s in scenarios if medium_metrics[s]['use_relay']),
        'large': sum(1 for s in scenarios if large_metrics[s]['use_relay'])
    }
    plt.bar(['小型网络', '中型网络', '大型网络'], 
            [relay_usage['small'], relay_usage['medium'], relay_usage['large']])
    plt.title('中继使用情况', fontdict=title_font)
    plt.ylabel('使用中继的场景数', fontdict=label_font)
    plt.grid(True)
    
    # Plot resource optimization usage
    plt.subplot(2, 2, 2)
    resource_opt = {
        'small': sum(1 for s in scenarios if small_metrics[s]['use_resource_opt']),
        'medium': sum(1 for s in scenarios if medium_metrics[s]['use_resource_opt']),
        'large': sum(1 for s in scenarios if large_metrics[s]['use_resource_opt'])
    }
    plt.bar(['小型网络', '中型网络', '大型网络'], 
            [resource_opt['small'], resource_opt['medium'], resource_opt['large']])
    plt.title('资源优化使用情况', fontdict=title_font)
    plt.ylabel('使用资源优化的场景数', fontdict=label_font)
    plt.grid(True)
    
    plt.suptitle('GNN特征分析', fontsize=16, y=0.95, fontfamily='WenQuanYi Zen Hei', fontweight='bold')
    plt.tight_layout()
    plt.savefig('gnn_analysis.png', dpi=300, bbox_inches='tight')

def main():
    network_types = ['small', 'medium', 'large']
    results = {}
    
    print("=== UAV网络联邦学习分析报告 ===\n")
    print("基于GNN的动态观测特征模型分析\n")
    
    for network_type in network_types:
        metrics_by_scenario = analyze_network_metrics(network_type)
        if metrics_by_scenario:
            results[network_type] = metrics_by_scenario
            print(f"\n=== {network_type.capitalize()} 网络配置 ===")
            
            for scenario, metrics in metrics_by_scenario.items():
                print(f"\n--- 场景: {metrics['description']} ---")
                print(f"网络规模: {metrics['num_users']} 用户, {metrics['num_uavs']} UAVs")
                print(f"收敛速率 (每轮损失下降): {metrics['convergence_rate']:.6f}")
                print(f"最终训练准确率: {metrics['final_accuracy']:.4f}")
                print(f"最终测试准确率: {metrics['final_test_accuracy']:.4f}")
                print(f"总通信开销: {metrics['total_communication_overhead']:.2f} MB")
                print(f"训练时间: {metrics['training_time']:.3f}秒")
                print(f"内存使用: {metrics['memory_usage']:.2f} MB")
                print(f"中继使用: {'是' if metrics['use_relay'] else '否'}")
                print(f"资源优化: {'是' if metrics['use_resource_opt'] else '否'}")
                print("\n训练损失历史:", [f"{loss:.6f}" for loss in metrics['train_loss_history'][:5]], "...")
                print("测试损失历史:", [f"{loss:.6f}" for loss in metrics['test_loss_history'][:5]], "...")
    
    if len(results) == 3:
        plot_metrics(results['small'], results['medium'], results['large'])
        print("\n详细的性能指标可视化已保存为 'network_comparison.png'")

if __name__ == "__main__":
    main()
