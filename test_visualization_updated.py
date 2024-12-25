import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.visualization import (
    plot_waveforms, plot_spectrograms, plot_error_distribution, plot_ablation_comparison
)

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing visualization functions."""
    t = np.linspace(0, 10, n_samples)
    # Original signal: sum of sinusoids with different frequencies
    original = (np.sin(2*np.pi*0.5*t) + 
               0.5*np.sin(2*np.pi*1.5*t) + 
               0.3*np.sin(2*np.pi*3.0*t))
    
    # Add different types of noise
    noise1 = np.random.normal(0, 0.2, n_samples)  # Gaussian noise
    noise2 = 0.1 * np.sin(2*np.pi*10*t)  # High-frequency sinusoidal noise
    noise = noise1 + noise2
    
    noisy = original + noise
    # Simulate denoised signal (better than noisy but not perfect)
    residual_noise = noise * 0.3  # Simulate 70% noise reduction
    denoised = original + residual_noise
    
    return original, noisy, denoised

def calculate_metrics(original, noisy, denoised):
    """Calculate various metrics for signal comparison."""
    # Calculate SNR
    noise_power = np.mean((original - noisy) ** 2)
    signal_power = np.mean(original ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    
    # Calculate PSNR
    max_signal = np.max(np.abs(original))
    mse = np.mean((original - denoised) ** 2)
    psnr = 20 * np.log10(max_signal / np.sqrt(mse))
    
    # Calculate Error Variance
    error = original - denoised
    error_variance = np.var(error)
    
    # Calculate Spectral MSE
    from scipy import signal
    f_orig, t_orig, Sxx_orig = signal.spectrogram(original)
    f_den, t_den, Sxx_den = signal.spectrogram(denoised)
    spectral_mse = np.mean((Sxx_orig - Sxx_den) ** 2)
    
    # Calculate Spectral SNR
    noise_spec = Sxx_orig - Sxx_den
    spectral_snr = 10 * np.log10(np.mean(Sxx_orig ** 2) / np.mean(noise_spec ** 2))
    
    return {
        'snr': snr,
        'psnr': psnr,
        'mse': mse,
        'error_variance': error_variance,
        'spectral_mse': spectral_mse,
        'spectral_snr': spectral_snr
    }

def generate_ablation_results():
    """Generate sample ablation study results."""
    baseline_results = {
        'proposed': {
            'metrics': {
                'mse': 0.00123,
                'snr': 15.5,
                'psnr': 25.8,
                'error_variance': 0.00456,
                'spectral_mse': 0.00789
            }
        }
    }
    
    ablation_results = {
        'No Channel Attention': {
            'metrics': {
                'mse': 0.00145,
                'snr': 14.8,
                'psnr': 24.5,
                'error_variance': 0.00523,
                'spectral_mse': 0.00856
            }
        },
        'No Temporal Attention': {
            'metrics': {
                'mse': 0.00156,
                'snr': 14.2,
                'psnr': 23.9,
                'error_variance': 0.00567,
                'spectral_mse': 0.00912
            }
        },
        'No Multi-Scale': {
            'metrics': {
                'mse': 0.00178,
                'snr': 13.5,
                'psnr': 22.8,
                'error_variance': 0.00612,
                'spectral_mse': 0.00978
            }
        }
    }
    return baseline_results, ablation_results

def main():
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    original, noisy, denoised = generate_sample_data()
    
    # Calculate metrics
    metrics = calculate_metrics(original, noisy, denoised)
    
    # Generate ablation results
    baseline_results, ablation_results = generate_ablation_results()
    
    # Plot waveforms with metrics
    plot_waveforms(original, noisy, denoised, metrics=metrics,
                  save_path=output_dir / "waveforms.png")
    
    # Plot spectrograms with metrics
    plot_spectrograms(original, noisy, denoised, metrics=metrics,
                     save_path=output_dir / "spectrograms.png")
    
    # Plot error distribution
    plot_error_distribution(original, denoised,
                          save_path=output_dir / "error_distribution.png")
    
    # Plot ablation comparison
    plot_ablation_comparison(baseline_results, ablation_results,
                           save_path=output_dir / "ablation_comparison.png")
    
    print("Generated visualization plots in visualization_output directory:")
    print("1. waveforms.png - Original, noisy, and denoised signal comparison")
    print("2. spectrograms.png - Spectrogram analysis")
    print("3. error_distribution.png - Error distribution analysis")
    print("4. ablation_comparison.png - Ablation study results")

if __name__ == "__main__":
    main()
