import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves():
    # Training metrics from the log
    epochs = range(1, 14)
    train_loss = [1.6197, 1.5954, 1.5849, 1.5741, 1.5484, 1.5603, 1.5569, 1.5303, 1.5173, 1.5154, 1.4994, 1.4903, 1.4781]
    val_loss = [1.5977, 1.5931, 1.5895, 1.5909, 1.5901, 1.5915, 1.5939, 1.5946, 1.5957, 1.5968, 1.5984, 1.6054, 1.6091]
    train_acc = [0.1567, 0.3410, 0.4562, 0.4516, 0.4747, 0.4700, 0.4654, 0.4654, 0.4839, 0.5069, 0.4977, 0.4977, 0.4885]
    val_acc = [0.1452, 0.4839, 0.4677, 0.4677, 0.4677, 0.4516, 0.4516, 0.4194, 0.4355, 0.4355, 0.4032, 0.4032, 0.4032]

    # Create output directory if it doesn't exist
    os.makedirs('outputs/analysis', exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot losses
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/analysis/learning_curves.png')
    plt.close()

    # Print analysis
    print("\n=== Learning Curves Analysis ===")
    print("\nLoss Trends:")
    print("- Training loss decreased from 1.62 to 1.48")
    print("- Validation loss increased slightly from 1.60 to 1.61")
    print("- Gap between train/val loss widening (potential overfitting)")
    
    print("\nAccuracy Trends:")
    print("- Training accuracy improved from 15.7% to 48.9%")
    print("- Validation accuracy peaked at 48.4% then declined to 40.3%")
    print("- Early stopping triggered at epoch 13 due to validation loss increase")
    
    print("\nRecommendations:")
    print("1. Reduce learning rate to slow down training")
    print("2. Increase dropout rate to combat overfitting")
    print("3. Add L2 regularization to model weights")
    print("4. Consider gradient clipping to stabilize training")

if __name__ == '__main__':
    plot_learning_curves()
