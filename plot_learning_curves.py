import matplotlib.pyplot as plt
import numpy as np
import re
import sys

def extract_losses(log_file):
    """Extract training and validation losses from log file."""
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if 'Training Loss:' in line and 'Validation Loss:' in line:
            # For classification models
            train_match = re.search(r'Training Loss: (\d+\.\d+)', line)
            val_match = re.search(r'Validation Loss: (\d+\.\d+)', line)
            if train_match and val_match:
                train_losses.append(float(train_match.group(1)))
                val_losses.append(float(val_match.group(1)))
        elif 'Train Loss:' in line and 'Val Loss:' in line:
            # For regression models
            train_match = re.search(r'Train Loss: (\d+\.\d+)', line)
            val_match = re.search(r'Val Loss: (\d+\.\d+)', line)
            if train_match and val_match:
                train_losses.append(float(train_match.group(1)))
                val_losses.append(float(val_match.group(1)))
    
    return np.array(train_losses), np.array(val_losses)

def plot_learning_curves(log_file, title):
    """Plot learning curves from training log."""
    train_losses, val_losses = extract_losses(log_file)
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{title} Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}_learning_curves.png')
    plt.close()
    
    print(f"\n{title} Training Progress:")
    print(f"Initial Train Loss: {train_losses[0]:.4f}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Best Train Loss: {min(train_losses):.4f}")
    print(f"Initial Val Loss: {val_losses[0]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    print(f"Best Val Loss: {min(val_losses):.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python plot_learning_curves.py <log_file> <title>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    title = sys.argv[2]
    plot_learning_curves(log_file, title)
