import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_confusion_matrix():
    print("=== Training Results Analysis ===")
    
    # Confusion matrix from training results
    cm = np.array([
        [ 0,  1,  2,  1, 25],
        [ 0,  8,  1,  0, 22],
        [ 0,  4,  1,  0, 26],
        [ 0,  7,  0,  2, 22],
        [ 0,  4,  1,  0, 90]
    ])
    
    # Calculate metrics
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    acc_per_class = correct_per_class / total_per_class
    
    print("\nAccuracy by Class:")
    for i, acc in enumerate(acc_per_class):
        print(f'Class {i}: {acc:.2%}')
    
    print('\nClass Distribution:')
    for i, total in enumerate(total_per_class):
        print(f'Class {i}: {total} samples ({total/cm.sum():.1%})')
    
    print('\nPrediction Distribution:')
    pred_per_class = cm.sum(axis=0)
    for i, pred in enumerate(pred_per_class):
        print(f'Class {i}: {pred} predictions ({pred/cm.sum():.1%})')
    
    print("\nModel Analysis:")
    print("- Class 4 (majority): High recall (95%) but potential overfitting")
    print("- Class 0: Complete failure (0% accuracy)")
    print("- Classes 1-3: Low accuracy but some correct predictions")
    print(f"- Overall accuracy: {correct_per_class.sum()/cm.sum():.2%}")
    
    print("\nRecommended improvements:")
    print("1. Increase training epochs (early stopping at 13 epochs is too soon)")
    print("2. Add more aggressive data augmentation for minority classes")
    print("3. Further adjust class weights to better handle imbalance")
    print("4. Lower learning rate (0.00002) and longer warmup period")
    print("5. Consider focal loss to focus on hard examples")
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs/analysis', exist_ok=True)
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/analysis/normalized_confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    analyze_confusion_matrix()
