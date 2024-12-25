import numpy as np
import torch
from typing import Union, Dict
from skimage.metrics import peak_signal_noise_ratio

def ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array if it's a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def compute_mse(y_true: Union[np.ndarray, torch.Tensor], 
                y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Mean Squared Error."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def compute_mae(y_true: Union[np.ndarray, torch.Tensor],
                y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Mean Absolute Error."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def compute_snr(y_true: Union[np.ndarray, torch.Tensor],
                y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    
    # Calculate noise as difference between true and predicted
    noise = y_true - y_pred
    
    # Compute signal and noise power
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:  # Avoid division by zero
        return 100.0  # Cap SNR at 100dB for numerical stability
        
    return 10 * np.log10(signal_power / noise_power)

def compute_psnr(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    # Ensure consistent shapes
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape(y_true.shape)
    return peak_signal_noise_ratio(y_true, y_pred, data_range=y_true.max() - y_true.min())

def compute_variance_metrics(y_true: Union[np.ndarray, torch.Tensor],
                           y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute variance-based metrics."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    error = y_true - y_pred
    
    return {
        'error_variance': float(np.var(error)),
        'prediction_variance': float(np.var(y_pred)),
        'target_variance': float(np.var(y_true)),
        'variance_ratio': float(np.var(y_pred) / np.var(y_true))
    }

def compute_spectral_metrics(y_true: Union[np.ndarray, torch.Tensor],
                           y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute spectral domain metrics."""
    y_true, y_pred = ensure_numpy(y_true), ensure_numpy(y_pred)
    
    # Compute FFT
    y_true_fft = np.abs(np.fft.fft(y_true))
    y_pred_fft = np.abs(np.fft.fft(y_pred))
    
    return {
        'spectral_mse': float(np.mean((y_true_fft - y_pred_fft) ** 2)),
        'spectral_mae': float(np.mean(np.abs(y_true_fft - y_pred_fft))),
        'spectral_snr': float(10 * np.log10(np.mean(y_true_fft ** 2) / 
                                           np.mean((y_true_fft - y_pred_fft) ** 2)))
    }

def compute_metrics(y_true: Union[np.ndarray, torch.Tensor],
                   y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth signal (numpy array or torch tensor)
        y_pred: Predicted/denoised signal (numpy array or torch tensor)
        
    Returns:
        Dictionary containing all metrics:
        - MSE (Mean Squared Error)
        - MAE (Mean Absolute Error)
        - SNR (Signal-to-Noise Ratio in dB)
        - PSNR (Peak Signal-to-Noise Ratio in dB)
        - Variance metrics (error, prediction, target variances)
        - Spectral metrics (FFT-based MSE, MAE, SNR)
    """
    # Basic metrics
    metrics = {
        'mse': compute_mse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'snr': compute_snr(y_true, y_pred),
        'psnr': compute_psnr(y_true, y_pred)
    }
    
    # Add variance metrics
    metrics.update(compute_variance_metrics(y_true, y_pred))
    
    # Add spectral metrics
    metrics.update(compute_spectral_metrics(y_true, y_pred))
    
    return metrics
