import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set Chinese font and style
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Use WenQuanYi Zen Hei font
plt.rcParams['axes.unicode_minus'] = False    # Fix minus sign display
plt.rcParams['font.family'] = 'sans-serif'    # Ensure we use the sans-serif font family
# Set basic style parameters
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3

def load_metrics():
    with open('sample_metrics.json', 'r') as f:
        return json.load(f)

def create_visualizations(metrics):
    # Set figure size and DPI for high quality
    fig_size = (12, 8)
    dpi = 300
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('联邦学习性能指标总览', fontsize=16, y=1.02)
    
    # Training Loss
    axes[0, 0].plot(metrics['training_loss'], 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('训练损失曲线', pad=10)
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(metrics['accuracy'], 'g-', linewidth=2, marker='o')
    axes[0, 1].set_title('模型准确率进展', pad=10)
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Communication Overhead
    axes[1, 0].plot(metrics['communication_overhead'], 'r-', linewidth=2, marker='o')
    axes[1, 0].set_title('通信开销分析', pad=10)
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('数据传输量 (MB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resource Usage
    l1, = axes[1, 1].plot(metrics['memory_usage'], 'c-', linewidth=2, marker='o', label='内存使用')
    l2, = axes[1, 1].plot(metrics['cpu_usage'], 'm-', linewidth=2, marker='o', label='CPU使用率')
    axes[1, 1].set_title('资源使用情况', pad=10)
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('使用率 (%)')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualizations/combined_metrics.png', dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    # Load metrics and create visualizations
    metrics = load_metrics()
    create_visualizations(metrics)

if __name__ == "__main__":
    main()
