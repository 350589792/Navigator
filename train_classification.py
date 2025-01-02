import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from models.classification_model import RGBClassificationModel
from utils.binning import ValueBinner
from preprocess_images_v2 import prepare_dataset

def train_classification_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    output_dir: str
) -> tuple[list[float], list[float], nn.Module, nn.Module]:
    
    # Calculate class weights from training data
    water_labels = []
    irr_labels = []
    for _, _, water_label, irr_label in train_loader:
        water_labels.extend(water_label.numpy())
        irr_labels.extend(irr_label.numpy())
    
    from sklearn.utils.class_weight import compute_class_weight
    water_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(water_labels),
        y=water_labels
    )
    irr_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(irr_labels),
        y=irr_labels
    )
    
    # Set up weighted loss functions
    water_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(water_weights, dtype=torch.float32).to(device)
    )
    irr_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(irr_weights, dtype=torch.float32).to(device)
    )
    """Train the classification model and save metrics.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Classification model
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        output_dir: Directory to save metrics and plots
        
    Returns:
        tuple[list[float], list[float], nn.Module, nn.Module]: 
            Training losses, validation losses, water criterion, irrigation criterion
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for images, texture_features, water_labels, irr_labels in train_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            water_labels = water_labels.to(device)
            irr_labels = irr_labels.to(device)
            
            optimizer.zero_grad()
            water_logits, irr_logits = model(images, texture_features)
            
            loss = water_criterion(water_logits, water_labels) + irr_criterion(irr_logits, irr_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texture_features, water_labels, irr_labels in val_loader:
                images = images.to(device)
                texture_features = texture_features.to(device)
                water_labels = water_labels.to(device)
                irr_labels = irr_labels.to(device)
                
                water_logits, irr_logits = model(images, texture_features)
                loss = water_criterion(water_logits, water_labels) + irr_criterion(irr_logits, irr_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses, water_criterion, irr_criterion

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    water_criterion: nn.Module,
    irr_criterion: nn.Module
):
    """Evaluate the model and save metrics and confusion matrices.
    
    Args:
        model: Classification model
        test_loader: Test data loader
        device: Device to evaluate on
        output_dir: Directory to save metrics and plots
    """
    model.eval()
    water_preds = []
    water_labels = []
    irr_preds = []
    irr_labels = []
    
    with torch.no_grad():
        for images, texture_features, water_true, irr_true in test_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            
            water_logits, irr_logits = model(images, texture_features)
            
            water_pred = torch.argmax(water_logits, dim=1).cpu().numpy()
            irr_pred = torch.argmax(irr_logits, dim=1).cpu().numpy()
            
            water_preds.extend(water_pred)
            water_labels.extend(water_true.numpy())
            irr_preds.extend(irr_pred)
            irr_labels.extend(irr_true.numpy())
    
    # Generate classification reports
    water_report = classification_report(water_labels, water_preds, zero_division=0)
    irr_report = classification_report(irr_labels, irr_preds, zero_division=0)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Water Saving Classification Report:\n")
        f.write(str(water_report))  # Convert to string explicitly
        f.write("\n\nIrrigation Classification Report:\n")
        f.write(str(irr_report))  # Convert to string explicitly
    
    # Generate confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    water_cm = confusion_matrix(water_labels, water_preds)
    sns.heatmap(water_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Water Saving Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    irr_cm = confusion_matrix(irr_labels, irr_preds)
    sns.heatmap(irr_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Irrigation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

def main():
    # Set up device and output directory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/classification_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and move to device
    model = RGBClassificationModel().to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders with classification labels
    train_loader, val_loader, test_loader = prepare_dataset(
        excel_path='/home/ubuntu/attachments/11.xlsx',
        img_dir='/home/ubuntu/attachments/img_xin11',
        batch_size=32,
        classification=True  # Enable classification mode
    )
    
    # Train model
    train_losses, val_losses, water_criterion, irr_criterion = train_classification_model(
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
        water_criterion=water_criterion,
        irr_criterion=irr_criterion
    )

if __name__ == '__main__':
    main()
