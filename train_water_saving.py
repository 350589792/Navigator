import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from models.classification_model import SingleTaskClassificationModel
from preprocess_images_v2 import prepare_dataset
from utils.binning import ValueBinner

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    output_dir: str
) -> tuple[list[float], list[float], nn.Module]:
    """Train the water saving classification model and save metrics.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Classification model
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        output_dir: Directory to save metrics and plots
        
    Returns:
        tuple[list[float], list[float], nn.Module]: 
            Training losses, validation losses, criterion
    """
    # Calculate class weights from training data
    labels = []
    for _, _, water_label, _ in train_loader:
        labels.extend(water_label.numpy())
    
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    
    # Set up weighted loss function
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, texture_features, labels, _ in train_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images, texture_features)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, texture_features, labels, _ in val_loader:
                images = images.to(device)
                texture_features = texture_features.to(device)
                labels = labels.to(device)
                
                logits = model(images, texture_features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, 'best_model.pth'),
                weights_only=True  # For security
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print('Early stopping triggered')
                break
    
    return train_losses, val_losses, criterion

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    criterion: nn.Module
):
    """Evaluate the model and save metrics and confusion matrices."""
    model.eval()
    predictions = []
    true_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texture_features, labels, _ in test_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            labels = labels.to(device)
            
            logits = model(images, texture_features)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    # Generate classification report as string for logging
    report_str = classification_report(true_labels, predictions, zero_division=0)
    
    # Calculate metrics using numpy for type safety
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate per-class metrics
    class_metrics = precision_recall_fscore_support(
        true_labels,
        predictions,
        average=None,
        zero_division=0
    )
    
    # Calculate macro averages
    macro_metrics = precision_recall_fscore_support(
        true_labels,
        predictions,
        average='macro',
        zero_division=0
    )
    
    # Save metrics
    metrics = {
        'test_loss': float(avg_test_loss),
        'test_accuracy': float(test_acc),
        'class_metrics': {},
        'macro_avg': {}
    }
    
    # Convert tuple outputs to numpy arrays
    precisions = np.array(class_metrics[0])
    recalls = np.array(class_metrics[1])
    f1_scores = np.array(class_metrics[2])
    supports = np.array(class_metrics[3])
    
    # Extract per-class metrics safely
    for i in range(5):
        if i < len(precisions):
            metrics['class_metrics'][f'class_{i}'] = {
                'precision': float(precisions[i].item()),
                'recall': float(recalls[i].item()),
                'f1_score': float(f1_scores[i].item()),
                'support': int(supports[i].item())
            }
    
    # Add macro averages (macro_metrics is already a tuple of floats)
    metrics['macro_avg'] = {
        'precision': float(macro_metrics[0]),
        'recall': float(macro_metrics[1]),
        'f1_score': float(macro_metrics[2]),
        'support': int(supports.sum().item())
    }
    
    import json
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Water Saving Classification Report:\n")
        f.write(report_str)
    
    # Generate confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Class {i}' for i in range(5)],
        yticklabels=[f'Class {i}' for i in range(5)]
    )
    plt.title('Water Saving Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    # Set up device and output directory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/water_saving_classification_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and move to device
    model = SingleTaskClassificationModel(num_classes=5).to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders with classification labels
    train_loader, val_loader, test_loader = prepare_dataset(
        excel_path='/home/ubuntu/attachments/11.xlsx',
        img_dir='/home/ubuntu/attachments/img_xin11',
        batch_size=32,
        classification=True
    )
    
    # Train model
    train_losses, val_losses, criterion = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        output_dir=output_dir
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    
    # Load best model and evaluate
    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, 'best_model.pth'),
            weights_only=True  # For security
        )
    )
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        criterion=criterion
    )

if __name__ == '__main__':
    main()
