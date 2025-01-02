import os
import logging
from datetime import datetime

# Set up logging configuration at the very start
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'outputs/classification_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create handlers
file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
console_handler = logging.StreamHandler()

# Set handler levels
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set specific logger levels
logging.getLogger('preprocess_images_v2').setLevel(logging.INFO)

# After logging setup, import other modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.classification_model import SingleTaskClassificationModel
from utils.binning import ValueBinner
from utils.create_dataloaders import create_dataloaders

def train_classification_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    output_dir: str
) -> tuple[list[float], list[float], nn.Module]:
    """Train the classification model with proper metric tracking.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Classification model
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        output_dir: Directory to save metrics
        
    Returns:
        tuple[list[float], list[float], nn.Module]: Training losses, validation losses, criterion
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing training...")
    
    # Calculate class weights from training data
    labels = []
    for _, label_batch, _ in train_loader:
        batch_labels = label_batch.numpy().flatten()
        labels.extend(batch_labels.astype(int).tolist())
    
    from sklearn.utils.class_weight import compute_class_weight
    
    # Force all 5 classes (0-4)
    all_classes = np.arange(5)
    
    # Ensure labels are in valid range and convert to integers
    labels = np.clip(labels, 0, 4).astype(int)
    
    # Compute weights for all 5 classes silently
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=all_classes,
        y=labels
    )
    
    # Set up weighted loss function
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    # Calculate class weights and set up loss function
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    no_improve = 0
    
    logger.info("Starting training...")
    logger.info(f"Training on device: {device}")
    logger.info(f"Number of batches per epoch: {len(train_loader)}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Total training samples: {len(train_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        error_count = 0
        max_errors = 5  # Maximum number of errors before stopping epoch
        
        for batch_idx, (images, labels, texture_features) in enumerate(train_loader):
            try:
                images = images.to(device)
                labels = labels.to(device)
                texture_features = texture_features.to(device)
                
                # Log shapes once at the start of training
                if epoch == 0 and batch_idx == 0:
                    logger.info("Starting training with input shapes:")
                    logger.info(f"Images: {images.shape}")
                    logger.info(f"Labels: {labels.shape}")
                    logger.info(f"Texture features: {texture_features.shape}")
                
                optimizer.zero_grad()
                logits = model(images, texture_features)
            except Exception as e:
                error_count += 1
                logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
                if error_count >= max_errors: 
                    logger.error(f"Stopping epoch {epoch+1} due to too many errors")
                    break
                continue
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            error_count = 0
            max_errors = 5  # Maximum number of errors before stopping validation
            
            for batch_idx, (images, labels, texture_features) in enumerate(val_loader):
                try:
                    images = images.to(device)
                    labels = labels.to(device)
                    texture_features = texture_features.to(device)
                    
                    # Validation in progress
                    logits = model(images, texture_features)
                except Exception as e:
                    error_count += 1
                    print(f"\nError in validation batch {batch_idx+1}:")
                    print(f"  {str(e)}")
                    if error_count >= max_errors:
                        print(f"\nStopping validation due to too many errors")
                        break
                    continue
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model and check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            no_improve = 0
        else:
            no_improve += 1
            
        # Calculate batch training accuracy and metrics
        with torch.no_grad():
            _, train_preds = torch.max(logits.data, 1)
            train_correct = (train_preds == labels).sum().item()
            train_total = labels.shape[0]
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            # Calculate validation accuracy and metrics
            val_correct = 0
            val_total = 0
            val_preds = []
            val_trues = []
            model.eval()
            for val_images, val_labels, val_features in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_features = val_features.to(device)
                val_outputs = model(val_images, val_features)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.shape[0]
                val_correct += (val_predicted == val_labels).sum().item()
                val_preds.extend(val_predicted.cpu().numpy())
                val_trues.extend(val_labels.cpu().numpy())
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # Log epoch metrics
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
            logger.info(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')
            
            # Early stopping check
            if no_improve >= patience:
                logger.info(f'Early stopping after {epoch+1} epochs without improvement')
                break
            model.train()
    
    return train_losses, val_losses, criterion

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    criterion: nn.Module
):
    """Evaluate the model and save detailed metrics and visualizations.
    
    Args:
        model: Classification model
        test_loader: Test data loader
        device: Device to evaluate on
        output_dir: Directory to save metrics and plots
        criterion: Loss criterion
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels, texture_features in test_loader:  # Correct order: images, labels, texture_features
            images = images.to(device)
            labels = labels.to(device)
            texture_features = texture_features.to(device)
            
            logits = model(images, texture_features)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report))
    
    # Generate confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting classification training...")
    
    # Explicitly set level for all loggers
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('preprocess_images_v2').setLevel(logging.INFO)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and move to device
    model = SingleTaskClassificationModel(num_classes=5, num_texture_features=31).to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders with classification labels
    train_loader, val_loader = create_dataloaders(
        task='water_saving_class',
        batch_size=32,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1
    )
    
    # Create test loader separately
    test_loader, _ = create_dataloaders(
        task='water_saving_class',
        batch_size=32,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        return_test=True
    )
    
    # Train model
    train_losses, val_losses, criterion = train_classification_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        num_epochs=50,
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
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        criterion=criterion
    )

if __name__ == '__main__':
    main()
