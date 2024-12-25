import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import Optional, Union, Dict
from pathlib import Path
import logging
from scipy import signal

logger = logging.getLogger(__name__)

def ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array if it's a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_waveforms(original: Union[np.ndarray, torch.Tensor],
                   noisy: Union[np.ndarray, torch.Tensor],
                   denoised: Union[np.ndarray, torch.Tensor],
                   metrics: Optional[dict] = None,
                   title: str = "Signal Comparison / 信号对比",
                   save_path: Optional[Union[str, Path]] = None):
    """Plot original, noisy, and denoised waveforms with metrics.
    
    Args:
        original: Original signal array
        noisy: Noisy signal array
        denoised: Denoised signal array
        metrics: Dictionary of metrics to display
        title: Plot title
        save_path: Path to save the plot
    """
    original = ensure_numpy(original)
    noisy = ensure_numpy(noisy)
    denoised = ensure_numpy(denoised)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Original signal
    plt.subplot(311)
    plt.plot(original, 'b-', label='Original')
    plt.title("Original Signal / 原始信号", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Amplitude / 振幅")
    plt.legend()
    
    # Noisy signal
    plt.subplot(312)
    plt.plot(noisy, 'r-', label='Noisy')
    plt.title("Noisy Signal / 加噪信号", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Amplitude / 振幅")
    plt.legend()
    
    # Denoised signal
    plt.subplot(313)
    plt.plot(denoised, 'g-', label='Denoised')
    plt.title("Denoised Signal / 去噪信号", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Sample / 采样点")
    plt.ylabel("Amplitude / 振幅")
    plt.legend()
    
    # Add metrics if provided
    if metrics:
        metrics_text = (
            f"SNR: {metrics.get('snr', 'N/A'):.2f} dB\n"
            f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB\n"
            f"MSE: {metrics.get('mse', 'N/A'):.6f}\n"
            f"Error Variance: {metrics.get('error_variance', 'N/A'):.6f}"
        )
        plt.figtext(1.02, 0.5, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved waveform plot to {save_path}")
    plt.close()

def plot_spectrograms(original: Union[np.ndarray, torch.Tensor],
                      noisy: Union[np.ndarray, torch.Tensor],
                      denoised: Union[np.ndarray, torch.Tensor],
                      metrics: Optional[dict] = None,
                      title: str = "Spectrogram Comparison / 谱图对比",
                      save_path: Optional[Union[str, Path]] = None,
                      sample_rate: int = 1000):
    """Plot spectrograms of original, noisy, and denoised signals with metrics.
    
    Args:
        original: Original signal array
        noisy: Noisy signal array
        denoised: Denoised signal array
        metrics: Dictionary of metrics to display
        title: Plot title
        save_path: Path to save the plot
        sample_rate: Sampling rate in Hz
    """
    original = ensure_numpy(original)
    noisy = ensure_numpy(noisy)
    denoised = ensure_numpy(denoised)
    
    fig = plt.figure(figsize=(15, 12))
    
    # Set spectrogram parameters
    nperseg = 256  # Window length
    noverlap = nperseg // 2  # Overlap between windows
    
    # Ensure signals are flattened
    original = original.flatten()
    noisy = noisy.flatten()
    denoised = denoised.flatten()
    
    # Plot spectrograms
    plt.subplot(311)
    f, t, Sxx = signal.spectrogram(original, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Power/dB / 功率/分贝')
    plt.title("Original Spectrogram / 原始谱图", fontsize=12)
    plt.ylabel("Frequency / 频率 [Hz]")
    
    plt.subplot(312)
    f, t, Sxx = signal.spectrogram(noisy, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Power/dB / 功率/分贝')
    plt.title("Noisy Spectrogram / 加噪谱图", fontsize=12)
    plt.ylabel("Frequency / 频率 [Hz]")
    
    plt.subplot(313)
    f, t, Sxx = signal.spectrogram(denoised, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Power/dB / 功率/分贝')
    plt.title("Denoised Spectrogram / 去噪谱图", fontsize=12)
    plt.ylabel("Frequency / 频率 [Hz]")
    plt.xlabel("Time / 时间 [s]")
    
    # Add metrics if provided
    if metrics:
        metrics_text = (
            f"Spectral MSE: {metrics.get('spectral_mse', 'N/A'):.6f}\n"
            f"Spectral SNR: {metrics.get('spectral_snr', 'N/A'):.2f} dB"
        )
        plt.figtext(1.02, 0.5, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectrogram plot to {save_path}")
    plt.close()

def plot_error_distribution(original: Union[np.ndarray, torch.Tensor],
                           denoised: Union[np.ndarray, torch.Tensor],
                           bins: int = 50,
                           title: str = "Error Distribution / 误差分布",
                           save_path: Optional[Union[str, Path]] = None):
    """Plot error distribution histogram.
    
    Args:
        original: Original signal array
        denoised: Denoised signal array
        bins: Number of histogram bins
        title: Plot title
        save_path: Path to save the plot
    """
    original = ensure_numpy(original)
    denoised = ensure_numpy(denoised)
    error = original - denoised
    
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=bins, density=True, alpha=0.75, color='b')
    plt.title(title, fontsize=12)
    plt.xlabel("Error / 误差")
    plt.ylabel("Density / 密度")
    plt.grid(True, alpha=0.3)
    
    # Add mean and std annotations
    mean_err = np.mean(error)
    std_err = np.std(error)
    plt.axvline(mean_err, color='r', linestyle='--', label=f'Mean / 均值: {mean_err:.2e}')
    plt.axvline(mean_err + std_err, color='g', linestyle=':', label=f'±1σ / 标准差: {std_err:.2e}')
    plt.axvline(mean_err - std_err, color='g', linestyle=':')
    plt.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")
    plt.close()

def plot_ablation_comparison(baseline_results: Dict, ablation_results: Dict,
                           save_path: Optional[str] = None):
    """Plot comprehensive ablation study comparison.
    
    Args:
        baseline_results: Results from baseline model
        ablation_results: Results from ablation variants
        save_path: Path to save the plot
    """
    metrics = ['mse', 'snr', 'psnr', 'error_variance', 'spectral_mse']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        models = []
        values = []
        
        # Add baseline result
        models.append('Full Model / 完整模型')
        values.append(baseline_results['proposed']['metrics'][metric])
        
        # Add ablation results
        for variant_name, result in ablation_results.items():
            models.append(f"{variant_name} / 消融变体")
            values.append(result['metrics'][metric])
        
        # Create bar plot
        bars = ax.bar(models, values)
        
        # Color bars (full model in blue, ablations in red)
        bars[0].set_color('blue')
        for bar in bars[1:]:
            bar.set_color('lightcoral')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
        
        # Customize plot
        ax.set_title(f'{metric.upper()} Comparison / {metric}对比', fontsize=12)
        ax.set_ylabel(metric.upper())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ablation Study Results / 消融实验结果', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ablation comparison plot to {save_path}")
    plt.close()
