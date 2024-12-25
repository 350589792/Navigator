import torch
import numpy as np
from pathlib import Path
from src.experiments.comparison import ModelComparison
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(config):
    """Create dummy test data for testing the comparison pipeline."""
    # Get parameters from config
    batch_size = config['model']['params']['batch_size']
    sequence_length = 1000  # Fixed sequence length
    input_channels = config['model']['input_channels']
    
    # Generate multiple synthetic traces
    t = np.linspace(0, 10, sequence_length)
    original_data = []
    noisy_data = []
    
    for _ in range(batch_size):
        # Create synthetic trace with multiple frequencies
        trace = (np.sin(2 * np.pi * 0.5 * t) + 
                0.5 * np.sin(2 * np.pi * 1.5 * t) + 
                0.3 * np.sin(2 * np.pi * 2.5 * t))
        
        # Add random phase shift for variety
        phase_shift = np.random.uniform(0, 2 * np.pi)
        trace = np.sin(2 * np.pi * t + phase_shift)
        
        # Add noise
        noise = (np.random.normal(0, 0.2, sequence_length) + 
                np.random.uniform(-0.1, 0.1, sequence_length))
        noisy_trace = trace + noise
        
        original_data.append(trace)
        noisy_data.append(noisy_trace)
    
    # Convert to numpy arrays with shape (batch_size, sequence_length)
    original_data = np.array(original_data)
    noisy_data = np.array(noisy_data)
    
    # Add channel dimension: (batch_size, sequence_length) -> (batch_size, channels, sequence_length)
    original_data = original_data[:, np.newaxis, :]
    noisy_data = noisy_data[:, np.newaxis, :]
    
    # Create output directory
    output_dir = Path(config['experiment']['report_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    save_path = output_dir / 'test_data.npz'
    np.savez(
        save_path,
        original_data=original_data,
        noisy_data=noisy_data
    )
    return save_path

def create_dummy_models(config):
    """Create and save dummy model weights for testing."""
    model_paths = {}
    models_dir = Path('test_models')
    models_dir.mkdir(exist_ok=True)
    
    # Fixed model parameters to match implementations
    hidden_channels = 64  # Fixed size for all models
    lstm_hidden = 64     # Fixed size for BiLSTM models
    num_layers = 2       # Fixed number of layers
    
    # Create dummy weights for each model type
    model_types = ['cnn', 'bilstm', 'cnn_bilstm', 'msc', 'msc_bilstm', 'msc_transformer_bilstm']
    
    for model_type in model_types:
        # Create appropriate state dict based on model type
        if model_type == 'cnn':
            state_dict = {
                'network.0.weight': torch.randn(hidden_channels, 1, 3),
                'network.0.bias': torch.randn(hidden_channels),
                'network.2.weight': torch.randn(hidden_channels, hidden_channels, 3),
                'network.2.bias': torch.randn(hidden_channels),
                'network.4.weight': torch.randn(hidden_channels, hidden_channels, 3),
                'network.4.bias': torch.randn(hidden_channels),
                'network.6.weight': torch.randn(1, hidden_channels, 3),
                'network.6.bias': torch.randn(1)
            }
        elif model_type == 'bilstm':
            # Create state dict for multi-layer BiLSTM
            state_dict = {}
            input_size = 1
            
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else lstm_hidden * 2
                
                # Forward direction
                state_dict[f'bilstm.weight_ih_l{layer}'] = torch.randn(4 * lstm_hidden, layer_input_size)
                state_dict[f'bilstm.weight_hh_l{layer}'] = torch.randn(4 * lstm_hidden, lstm_hidden)
                state_dict[f'bilstm.bias_ih_l{layer}'] = torch.randn(4 * lstm_hidden)
                state_dict[f'bilstm.bias_hh_l{layer}'] = torch.randn(4 * lstm_hidden)
                
                # Backward direction
                state_dict[f'bilstm.weight_ih_l{layer}_reverse'] = torch.randn(4 * lstm_hidden, layer_input_size)
                state_dict[f'bilstm.weight_hh_l{layer}_reverse'] = torch.randn(4 * lstm_hidden, lstm_hidden)
                state_dict[f'bilstm.bias_ih_l{layer}_reverse'] = torch.randn(4 * lstm_hidden)
                state_dict[f'bilstm.bias_hh_l{layer}_reverse'] = torch.randn(4 * lstm_hidden)
            
            # Output layer
            state_dict['fc.weight'] = torch.randn(1, lstm_hidden * 2)
            state_dict['fc.bias'] = torch.randn(1)
        elif model_type == 'cnn_bilstm':
            state_dict = {
                'cnn.0.weight': torch.randn(hidden_channels, 1, 3),
                'cnn.0.bias': torch.randn(hidden_channels),
                'cnn.2.weight': torch.randn(hidden_channels, hidden_channels, 3),
                'cnn.2.bias': torch.randn(hidden_channels),
                'cnn.4.weight': torch.randn(hidden_channels, hidden_channels, 3),
                'cnn.4.bias': torch.randn(hidden_channels),
                # First LSTM layer
                'bilstm.weight_ih_l0': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.weight_hh_l0': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.bias_ih_l0': torch.randn(4 * hidden_channels),
                'bilstm.bias_hh_l0': torch.randn(4 * hidden_channels),
                'bilstm.weight_ih_l0_reverse': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.weight_hh_l0_reverse': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.bias_ih_l0_reverse': torch.randn(4 * hidden_channels),
                'bilstm.bias_hh_l0_reverse': torch.randn(4 * hidden_channels),
                # Second LSTM layer
                'bilstm.weight_ih_l1': torch.randn(4 * hidden_channels, hidden_channels * 2),
                'bilstm.weight_hh_l1': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.bias_ih_l1': torch.randn(4 * hidden_channels),
                'bilstm.bias_hh_l1': torch.randn(4 * hidden_channels),
                'bilstm.weight_ih_l1_reverse': torch.randn(4 * hidden_channels, hidden_channels * 2),
                'bilstm.weight_hh_l1_reverse': torch.randn(4 * hidden_channels, hidden_channels),
                'bilstm.bias_ih_l1_reverse': torch.randn(4 * hidden_channels),
                'bilstm.bias_hh_l1_reverse': torch.randn(4 * hidden_channels),
                'fc.weight': torch.randn(1, hidden_channels * 2),
                'fc.bias': torch.randn(1)
            }
        elif model_type == 'msc':
            state_dict = {
                'conv1.weight': torch.randn(hidden_channels, 1, 3),
                'conv1.bias': torch.randn(hidden_channels),
                'conv2.weight': torch.randn(hidden_channels, 1, 5),
                'conv2.bias': torch.randn(hidden_channels),
                'conv3.weight': torch.randn(hidden_channels, 1, 7),
                'conv3.bias': torch.randn(hidden_channels),
                'conv_out.weight': torch.randn(1, hidden_channels * 3, 3),
                'conv_out.bias': torch.randn(1)
            }
        elif model_type == 'msc_bilstm':
            state_dict = {
                'conv1.weight': torch.randn(hidden_channels, 1, 3),
                'conv1.bias': torch.randn(hidden_channels),
                'conv2.weight': torch.randn(hidden_channels, 1, 5),
                'conv2.bias': torch.randn(hidden_channels),
                'conv3.weight': torch.randn(hidden_channels, 1, 7),
                'conv3.bias': torch.randn(hidden_channels),
                # First LSTM layer
                'bilstm.weight_ih_l0': torch.randn(4 * lstm_hidden, hidden_channels * 3),
                'bilstm.weight_hh_l0': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l0': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l0': torch.randn(4 * lstm_hidden),
                'bilstm.weight_ih_l0_reverse': torch.randn(4 * lstm_hidden, hidden_channels * 3),
                'bilstm.weight_hh_l0_reverse': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l0_reverse': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l0_reverse': torch.randn(4 * lstm_hidden),
                # Second LSTM layer
                'bilstm.weight_ih_l1': torch.randn(4 * lstm_hidden, lstm_hidden * 2),
                'bilstm.weight_hh_l1': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l1': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l1': torch.randn(4 * lstm_hidden),
                'bilstm.weight_ih_l1_reverse': torch.randn(4 * lstm_hidden, lstm_hidden * 2),
                'bilstm.weight_hh_l1_reverse': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l1_reverse': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l1_reverse': torch.randn(4 * lstm_hidden),
                'fc.weight': torch.randn(1, lstm_hidden * 2),
                'fc.bias': torch.randn(1)
            }
        else:  # msc_transformer_bilstm
            state_dict = {
                'conv1.weight': torch.randn(hidden_channels, 1, 3),
                'conv1.bias': torch.randn(hidden_channels),
                'conv2.weight': torch.randn(hidden_channels, 1, 5),
                'conv2.bias': torch.randn(hidden_channels),
                'conv3.weight': torch.randn(hidden_channels, 1, 7),
                'conv3.bias': torch.randn(hidden_channels),
                'transformer.self_attn.in_proj_weight': torch.randn(3 * hidden_channels * 3, hidden_channels * 3),
                'transformer.self_attn.in_proj_bias': torch.randn(3 * hidden_channels * 3),
                'transformer.self_attn.out_proj.weight': torch.randn(hidden_channels * 3, hidden_channels * 3),
                'transformer.self_attn.out_proj.bias': torch.randn(hidden_channels * 3),
                'transformer.norm1.weight': torch.randn(hidden_channels * 3),
                'transformer.norm1.bias': torch.randn(hidden_channels * 3),
                'transformer.ff.0.weight': torch.randn(128, hidden_channels * 3),  # Updated to match model architecture
                'transformer.ff.0.bias': torch.randn(128),  # Updated to match model architecture
                'transformer.ff.2.weight': torch.randn(hidden_channels * 3, 128),  # Updated to match model architecture
                'transformer.ff.2.bias': torch.randn(hidden_channels * 3),
                'transformer.norm2.weight': torch.randn(hidden_channels * 3),
                'transformer.norm2.bias': torch.randn(hidden_channels * 3),
                # First LSTM layer
                'bilstm.weight_ih_l0': torch.randn(4 * lstm_hidden, hidden_channels * 3),
                'bilstm.weight_hh_l0': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l0': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l0': torch.randn(4 * lstm_hidden),
                'bilstm.weight_ih_l0_reverse': torch.randn(4 * lstm_hidden, hidden_channels * 3),
                'bilstm.weight_hh_l0_reverse': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l0_reverse': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l0_reverse': torch.randn(4 * lstm_hidden),
                # Second LSTM layer
                'bilstm.weight_ih_l1': torch.randn(4 * lstm_hidden, lstm_hidden * 2),
                'bilstm.weight_hh_l1': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l1': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l1': torch.randn(4 * lstm_hidden),
                'bilstm.weight_ih_l1_reverse': torch.randn(4 * lstm_hidden, lstm_hidden * 2),
                'bilstm.weight_hh_l1_reverse': torch.randn(4 * lstm_hidden, lstm_hidden),
                'bilstm.bias_ih_l1_reverse': torch.randn(4 * lstm_hidden),
                'bilstm.bias_hh_l1_reverse': torch.randn(4 * lstm_hidden),
                'fc.weight': torch.randn(1, lstm_hidden * 2),
                'fc.bias': torch.randn(1)
            }
        
        # Save weights
        save_path = models_dir / f'{model_type}.pth'
        torch.save(state_dict, save_path)
        model_paths[model_type] = str(save_path)
    
    return model_paths

def main():
    """Test the model comparison pipeline with ablation studies."""
    # Load config
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    report_dir = Path(config['experiment']['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = report_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Create test data and model weights
    logger.info("Creating test data...")
    test_data_path = create_dummy_data(config)
    
    logger.info("Creating dummy model weights...")
    model_paths = create_dummy_models(config)
    
    # Initialize comparison
    logger.info("Initializing model comparison...")
    comparison = ModelComparison(config)
    
    # Run baseline comparison
    logger.info("Running baseline model comparison...")
    baseline_results = comparison.run_comparison(
        test_data_path=test_data_path,
        model_paths=model_paths,
        save_dir=str(report_dir)
    )
    
    # Run ablation studies
    logger.info("\nRunning ablation studies...")
    ablation_variants = {
        'no_channel_attention': {'disable_channel_attention': True},
        'no_temporal_attention': {'disable_temporal_attention': True},
        'no_adaptive_weights': {'disable_adaptive_weights': True},
        'no_perceptual_loss': {'disable_perceptual_loss': True},
        'no_attention': {
            'disable_channel_attention': True,
            'disable_temporal_attention': True
        }
    }
    
    ablation_results = {}
    for variant_name, variant_config in ablation_variants.items():
        logger.info(f"\nTesting ablation variant: {variant_name}")
        variant_comparison = ModelComparison(config, ablation_config=variant_config)
        results = variant_comparison.run_comparison(
            test_data_path=test_data_path,
            model_paths={'proposed': model_paths['msc_transformer_bilstm']},
            save_dir=str(report_dir / variant_name)
        )
        ablation_results[variant_name] = results['proposed']
    
    # Verify and compare results
    logger.info("\nVerifying results...")
    
    # Baseline models results
    logger.info("\nBaseline Models Results:")
    for model_name, model_results in baseline_results.items():
        logger.info(f"\nResults for {model_name}:")
        metrics = model_results['metrics']
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"SNR: {metrics['snr']:.2f} dB")
        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"Error Variance: {metrics['error_variance']:.6f}")
        logger.info(f"Spectral MSE: {metrics['spectral_mse']:.6f}")
    
    # Ablation study results
    logger.info("\nAblation Study Results:")
    for variant_name, variant_results in ablation_results.items():
        logger.info(f"\nResults for {variant_name}:")
        metrics = variant_results['metrics']
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"SNR: {metrics['snr']:.2f} dB")
        logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"Error Variance: {metrics['error_variance']:.6f}")
        logger.info(f"Spectral MSE: {metrics['spectral_mse']:.6f}")
    
    # Generate comparison plots
    logger.info("\nGenerating comparison plots...")
    plot_metrics = ['mse', 'snr', 'psnr', 'error_variance']
    for metric in plot_metrics:
        comparison.plot_metric_comparison(
            baseline_results,
            ablation_results,
            metric,
            save_path=str(report_dir / f'{metric}_comparison.png')
        )
    
    logger.info(f"\nResults and plots saved to {report_dir}")
    logger.info("Test completed successfully")

if __name__ == '__main__':
    main()
