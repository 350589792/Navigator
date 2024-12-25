import numpy as np
from scipy import signal

def analyze_signal(data):
    """Analyze signal characteristics for adaptive noise generation.
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary of signal characteristics
    """
    # Calculate basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(np.abs(data))
    
    # Calculate signal energy
    energy = np.sum(np.square(data))
    
    # Calculate frequency characteristics
    freqs, psd = signal.welch(data)
    dominant_freq_idx = np.argmax(psd)
    dominant_freq = freqs[dominant_freq_idx]
    
    return {
        'mean': mean_val,
        'std': std_val,
        'max_amplitude': max_val,
        'energy': energy,
        'dominant_freq': dominant_freq,
        'freq_power': psd[dominant_freq_idx]
    }

def adaptive_noise_ratio(signal_features):
    """Calculate adaptive noise ratio based on signal features.
    
    Args:
        signal_features: Dictionary of signal characteristics
        
    Returns:
        Adapted noise ratio
    """
    # Base the noise ratio on signal characteristics
    energy_factor = np.clip(np.log10(signal_features['energy']) / 10, 0.1, 1.0)
    amplitude_factor = np.clip(signal_features['max_amplitude'] / 100, 0.1, 1.0)
    
    # Combine factors (can be adjusted based on requirements)
    adaptive_ratio = 0.1 * (energy_factor + amplitude_factor) / 2
    return adaptive_ratio

def add_gaussian_noise(data, noise_ratio=0.1, adaptive=True):
    """Add Gaussian noise to the input data with adaptive noise level.
    
    Args:
        data: Input data array
        noise_ratio: Base standard deviation ratio for noise generation
        adaptive: Whether to use adaptive noise ratio
    
    Returns:
        Noisy data array
    """
    if adaptive:
        features = analyze_signal(data)
        noise_ratio = adaptive_noise_ratio(features)
    
    std_dev = noise_ratio * np.std(data)
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def add_poisson_noise(data, scale=1.0, adaptive=True):
    """Add Poisson noise to the input data with adaptive scaling.
    
    Args:
        data: Input data array
        scale: Base scale factor for noise
        adaptive: Whether to use adaptive scaling
    
    Returns:
        Noisy data array
    """
    if adaptive:
        features = analyze_signal(data)
        scale = adaptive_noise_ratio(features)
    
    # Normalize and scale data for Poisson noise
    normalized_data = (data - data.min()) * scale
    noise = np.random.poisson(normalized_data)
    
    # Scale noise back to original range
    noise = noise / scale
    return data + noise

def add_uniform_noise(data, noise_ratio=0.1, adaptive=True):
    """Add uniform noise to the input data with adaptive range.
    
    Args:
        data: Input data array
        noise_ratio: Base range ratio for noise generation
        adaptive: Whether to use adaptive noise ratio
    
    Returns:
        Noisy data array
    """
    if adaptive:
        features = analyze_signal(data)
        noise_ratio = adaptive_noise_ratio(features)
    
    range_val = noise_ratio * (np.max(data) - np.min(data))
    noise = np.random.uniform(-range_val, range_val, data.shape)
    return data + noise

def add_salt_and_pepper_noise(data, prob=0.01, adaptive=True):
    """Add salt and pepper noise to the input data with adaptive probability.
    
    Args:
        data: Input data array
        prob: Base probability of noise occurrence
        adaptive: Whether to use adaptive probability
    
    Returns:
        Noisy data array
    """
    if adaptive:
        features = analyze_signal(data)
        prob = adaptive_noise_ratio(features) * 0.1  # Scale down ratio for reasonable probability
    
    noisy_data = data.copy()
    total_pixels = data.size
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)
    
    # Add salt noise (set to max value)
    salt_coords = np.random.choice(total_pixels, num_salt, replace=False)
    noisy_data.ravel()[salt_coords] = np.max(data)
    
    # Add pepper noise (set to min value)
    pepper_coords = np.random.choice(total_pixels, num_pepper, replace=False)
    noisy_data.ravel()[pepper_coords] = np.min(data)
    
    return noisy_data
