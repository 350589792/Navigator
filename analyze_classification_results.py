import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_confusion_matrix(cm, classes, title, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_classification_results(task, predictions, true_labels, class_names):
    """Analyze classification results for a specific task."""
    logging.info(f"\nAnalyzing {task} classification results...")
    
    # Convert predictions to class indices
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_classes)
    
    # Generate classification report
    report = classification_report(true_labels, pred_classes, target_names=class_names, output_dict=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, f'{task} Confusion Matrix', f'{task}_confusion_matrix.png')
    
    # Save metrics
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    with open(f'{task}_classification_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Log summary metrics
    logging.info(f"\n{task} Classification Results:")
    
    # Handle the dictionary structure from classification_report
    if isinstance(report, dict):
        accuracy = report.get('accuracy', 0.0)
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nPer-class metrics:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                logging.info(f"{class_name}:")
                logging.info(f"  Precision: {metrics['precision']:.4f}")
                logging.info(f"  Recall: {metrics['recall']:.4f}")
                logging.info(f"  F1-score: {metrics['f1-score']:.4f}")
            else:
                logging.warning(f"No metrics found for class {class_name}")
    else:
        logging.error("Classification report is not in expected dictionary format")

def main():
    """Main function to analyze both classification tasks."""
    logging.info("Starting classification analysis...")
    
    # Load results
    try:
        with open('classification_results.json', 'r') as f:
            results = json.load(f)
        
        # Analyze water saving results
        water_saving_classes = [f'Class_{i}' for i in range(5)]  # Update with actual class names
        analyze_classification_results(
            'water_saving',
            np.array(results['water_saving']['predictions']),
            np.array(results['water_saving']['true_labels']),
            water_saving_classes
        )
        
        # Analyze irrigation results
        irrigation_classes = [f'Class_{i}' for i in range(5)]  # Update with actual class names
        analyze_classification_results(
            'irrigation',
            np.array(results['irrigation']['predictions']),
            np.array(results['irrigation']['true_labels']),
            irrigation_classes
        )
        
        logging.info("\nAnalysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
