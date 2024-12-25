import os
import torch
import logging
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
import numpy as np

from src.data.dataset import SeismicDataset
from src.experiments.comparison import ModelComparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from yaml file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    """Main pipeline for model comparison."""
    # Load configuration
    config = load_config()
    
    # Set random seeds
    setup_seed(config['training']['seed'])
    
    # Set device
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    # Initialize dataset
    test_dataset = SeismicDataset(
        data_path=config['data']['test_clean_path'],
        noisy_data_path=config['data']['test_noisy_path']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['params']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model comparison
    comparison = ModelComparison(config, device=device)
    
    # Load baseline models
    baseline_paths = {
        'cnn': config['models']['cnn_path'],
        'bilstm': config['models']['bilstm_path'],
        'cnn_bilstm': config['models']['cnn_bilstm_path'],
        'msc': config['models']['msc_path'],
        'msc_bilstm': config['models']['msc_bilstm_path'],
        'msc_transformer_bilstm': config['models']['msc_transformer_bilstm_path']
    }
    
    try:
        logger.info("Loading baseline models...")
        comparison.load_baseline_models(baseline_paths)
    except Exception as e:
        logger.error(f"Error loading baseline models: {str(e)}")
        raise
    
    # Load proposed model
    try:
        logger.info("Loading proposed model...")
        comparison.load_proposed_model(config['models']['proposed_path'])
    except Exception as e:
        logger.error(f"Error loading proposed model: {str(e)}")
        raise
    
    # Evaluate all models
    logger.info("Starting model evaluation...")
    comparison.evaluate_models(test_loader)
    
    # Generate comprehensive report
    report_dir = Path(config['experiment']['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating comparison report...")
    comparison.generate_report(report_dir)
    
    logger.info(f"Comparison complete. Report generated in {report_dir}")

if __name__ == "__main__":
    main()
