import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.visualization import (
    plot_waveforms, plot_spectrograms, 
    plot_error_distribution, plot_ablation_comparison
)

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing visualization functions."""
    # Generate original signal (sine wave + noise)
    t = np.linspace(0, 10, n_samples)
    original = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.sin(2 * np.pi * 10.0 * t)
    
    # Generate noisy signal
    noise = np.random.normal(0, 0.2, n_samples)
    noisy = original + noise
    
    # Generate denoised signal (slightly better than noisy)
    denoised = original + 0.5 * noise
    
    return original, noisy, denoised

def generate_sample_metrics():
    """Generate sample metrics for visualization."""
    return {
        'snr': 15.5,
        'psnr': 25.8,
        'mse': 0.00123,
        'mae': 0.00234,
        'error_variance': 0.00345,
        'spectral_mse': 0.00456,
        'spectral_snr': 18.9
    }

def generate_ablation_results():
    """Generate sample ablation study results."""
    baseline_results = {
        'proposed': {
            'metrics': {
                'mse': 0.00123,
                'snr': 15.5,
                'psnr': 25.8,
                'error_variance': 0.00345,
                'spectral_mse': 0.00456
            }
        }
    }
    
    ablation_results = {
        'no_channel_attention': {
            'metrics': {
                'mse': 0.00145,
                'snr': 14.8,
                'psnr': 24.5,
                'error_variance': 0.00389,
                'spectral_mse': 0.00478
            }
        },
        'no_temporal_attention': {
            'metrics': {
                'mse': 0.00156,
                'snr': 14.2,
                'psnr': 24.1,
                'error_variance': 0.00412,
                'spectral_mse': 0.00489
            }
        },
        'no_adaptive_weights': {
            'metrics': {
                'mse': 0.00167,
                'snr': 13.9,
                'psnr': 23.8,
                'error_variance': 0.00434,
                'spectral_mse': 0.00501
            }
        }
    }
    
    return baseline_results, ablation_results

def main():
    """Test visualization functions with sample data."""
    # Create output directory
    output_dir = Path('visualization_test_results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    original, noisy, denoised = generate_sample_data()
    metrics = generate_sample_metrics()
    baseline_results, ablation_results = generate_ablation_results()
    
    # Test waveform plotting
    print("Generating waveform plot...")
    plot_waveforms(
        original, noisy, denoised,
        metrics=metrics,
        title="Signal Comparison / 信号对比",
        save_path=output_dir / 'waveforms.png'
    )
    
    # Test spectrogram plotting
    print("Generating spectrogram plot...")
    plot_spectrograms(
        original, noisy, denoised,
        metrics=metrics,
        title="Spectrogram Comparison / 谱图对比",
        save_path=output_dir / 'spectrograms.png',
        sample_rate=100
    )
    
    # Test error distribution plotting
    print("Generating error distribution plot...")
    plot_error_distribution(
        original, denoised,
        metrics=metrics,
        title="Error Distribution / 误差分布",
        save_path=output_dir / 'error_distribution.png'
    )
    
    # Test ablation comparison plotting
    print("Generating ablation comparison plot...")
    plot_ablation_comparison(
        baseline_results,
        ablation_results,
        save_path=output_dir / 'ablation_comparison.png'
    )
    
    print(f"All plots saved to {output_dir}")

if __name__ == '__main__':
    main()
