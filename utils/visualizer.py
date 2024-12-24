import torch
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

# Set up font for Chinese characters
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'IPAGothic', 'Unifont', 'DejaVu Sans']  # Primary Chinese font with fallbacks
plt.rcParams['axes.unicode_minus'] = False  # For minus sign


def trajectory_visualizer(model, data_loader, device, save_path):
    r"""Visualize trajectories from model predictions
    
    Args:
        model: The trained model
        data_loader: DataLoader containing validation/test data
        device: Device to run predictions on
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Get first batch for visualization
    model.eval()
    with torch.no_grad():
        try:
            obs_traj, pred_traj_gt = next(iter(data_loader))
            # Ensure tensors are on CPU from the start
            obs_traj = obs_traj.float().cpu()
            pred_traj_gt = pred_traj_gt.float().cpu()
            
            # Get model predictions (model is already on CPU)
            pred_traj = model(obs_traj)
            
            if torch.isnan(pred_traj).any():
                warnings.warn("NaN values detected in model predictions")
                return
        except Exception as e:
            warnings.warn(f"Error during trajectory prediction: {str(e)}")
            return
    
    # Remove nodes dimension and convert to numpy
    obs_traj = obs_traj.squeeze(2).numpy()  # [batch, obs_len, 2]
    pred_traj_gt = pred_traj_gt.squeeze(2).numpy()  # [batch, pred_len, 2]
    pred_traj = pred_traj.squeeze(2).numpy()  # [batch, pred_len, 2]

    # Visualize trajectories
    linew = 3
    fig = plt.figure(figsize=(10, 8))

    # Plot each trajectory in batch
    for n in range(len(obs_traj)):
        # Plot observed trajectory
        plt.plot(obs_traj[n, :, 0], obs_traj[n, :, 1], 
                linestyle='-', color='darkorange', linewidth=linew, 
                label='观测轨迹' if n == 0 else None)
        
        # Plot ground truth future trajectory
        plt.plot(pred_traj_gt[n, :, 0], pred_traj_gt[n, :, 1], 
                linestyle='-', color='lime', linewidth=linew, 
                label='真实轨迹' if n == 0 else None)
        
        # Plot predicted trajectory
        plt.plot(pred_traj[n, :, 0], pred_traj[n, :, 1], 
                linestyle='-', color='yellow', linewidth=linew, 
                label='预测轨迹' if n == 0 else None)

    plt.tick_params(axis="y", direction="in", pad=-22)
    plt.tick_params(axis="x", direction="in", pad=-15)
    plt.xlim(-11, 18)
    plt.ylim(-12, 15)
    
    # Add Chinese labels
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.title('轨迹预测结果', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    try:
        # Save with higher DPI and Windows-compatible format
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    except Exception as e:
        warnings.warn(f"Error saving visualization to {save_path}: {str(e)}")
    finally:
        plt.close('all')


def controlpoint_visualizer(model, data_loader, device, save_path):
    r"""Visualize predicted trajectories as control points
    
    Args:
        model: The trained model
        data_loader: DataLoader containing validation/test data
        device: Device to run predictions on
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt

    # Get predictions from model
    model.eval()
    with torch.no_grad():
        try:
            obs_traj, _ = next(iter(data_loader))
            # Ensure tensor is on CPU from the start
            obs_traj = obs_traj.float().cpu()
            
            # Get predicted trajectory (model is already on CPU)
            pred_traj = model(obs_traj)
            
            if torch.isnan(pred_traj).any():
                warnings.warn("NaN values detected in model predictions")
                return
                
            # Convert to numpy
            pred_traj = pred_traj.squeeze(2).numpy()  # [batch, time, 2]
        except Exception as e:
            warnings.warn(f"Error during control point prediction: {str(e)}")
            return

    # Visualize predicted points as control points
    fig = plt.figure(figsize=(10, 8))

    # Plot each trajectory's predicted points
    for n in range(pred_traj.shape[0]):
        # Plot predicted points
        plt.scatter(pred_traj[n, :, 0], pred_traj[n, :, 1], 
                   c='red', marker='x', s=100, label='预测控制点' if n == 0 else None)
        
        # Connect predicted points with lines
        plt.plot(pred_traj[n, :, 0], pred_traj[n, :, 1], 
                'r--', alpha=0.5, label='预测路径' if n == 0 else None)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.title('轨迹控制点预测', fontsize=14)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.tight_layout()
    try:
        # Save with higher DPI and Windows-compatible format
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    except Exception as e:
        warnings.warn(f"Error saving visualization to {save_path}: {str(e)}")
    finally:
        plt.close('all')
