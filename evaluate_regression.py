import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.irrigation_model import IrrigationModel
from models.water_saving_model import WaterSavingModel
from utils.create_dataloaders import create_dataloaders

def check_model_exists(model_path):
    """Check if model file exists and print status."""
    exists = os.path.exists(model_path)
    print(f"\nChecking model path: {model_path}")
    print(f"Model file exists: {exists}")
    return exists

def evaluate_regression_model(model_path, model_type='water_saving'):
    """Evaluate regression model performance."""
    logging.info(f"\nEvaluating {model_type} regression model...")
    
    if not check_model_exists(model_path):
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=32,
            task=model_type
        )
        logging.info(f"Created dataloaders successfully. Test set size: {len(test_loader.dataset)}")
    except Exception as e:
        logging.error(f"Error creating dataloaders: {str(e)}")
        logging.error(traceback.format_exc())
        return
    
    # Load model
    logging.info("Loading model...")
    try:
        if model_type == 'water_saving':
            model = WaterSavingModel()
        else:
            model = IrrigationModel()
        
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        return
    
    # Evaluate model
    logging.info("\nStarting evaluation...")
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for batch_idx, (images, labels, texture_features) in enumerate(val_loader):
                if batch_idx % 10 == 0:
                    logging.info(f"Processing batch {batch_idx}/{len(val_loader)}")
                
                images = images.to(device)
                texture_features = texture_features.to(device)
                outputs = model(images, texture_features)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_labels, all_preds)
        
        # Plot predictions vs true values
        plt.figure(figsize=(10, 6))
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{model_type.replace("_", " ").title()} Predictions vs True Values')
        plt.savefig(f'{model_type}_predictions.png')
        plt.close()
        
        logging.info(f"\n{model_type.replace('_', ' ').title()} Regression Results:")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"R² Score: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        logging.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    logging.info("\nStarting regression model evaluation...")
    
    # Define model paths
    models_dir = os.path.dirname(os.path.abspath(__file__))
    water_saving_path = os.path.join(models_dir, 'best_water_saving_model.pth')
    irrigation_path = os.path.join(models_dir, 'best_irrigation_model.pth')
    
    try:
        # Evaluate both models
        logging.info("\nStarting water saving model evaluation...")
        water_results = evaluate_regression_model(water_saving_path, 'water_saving')
        
        logging.info("\nStarting irrigation model evaluation...")
        irrigation_results = evaluate_regression_model(irrigation_path, 'irrigation')
        
        if water_results and irrigation_results:
            logging.info("\nEvaluation completed successfully for both models")
            
            # Save results to file
            results = {
                'water_saving': {
                    'rmse': float(water_results['rmse']),
                    'r2': float(water_results['r2'])
                },
                'irrigation': {
                    'rmse': float(irrigation_results['rmse']),
                    'r2': float(irrigation_results['r2'])
                }
            }
            
            import json
            with open('regression_results.json', 'w') as f:
                json.dump(results, f, indent=4)
            logging.info("Results saved to regression_results.json")
        else:
            logging.error("\nEvaluation failed for one or both models")
            
    except Exception as e:
        logging.error(f"\nUnexpected error during evaluation: {str(e)}")
        logging.error(traceback.format_exc())
