import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from models.irrigation_model import IrrigationModel, IrrigationLoss
from utils.create_dataloaders import create_dataloaders
import pandas as pd

def train_model(excel_path, image_dir, num_epochs=100, batch_size=32, learning_rate=0.0001):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders for irrigation regression
    train_loader, val_loader = create_dataloaders(
        task='irrigation',
        batch_size=batch_size,
        train_split=0.7,
        val_split=0.2
    )
    # Create test loader separately
    _, test_loader = create_dataloaders(
        task='irrigation',
        batch_size=batch_size,
        train_split=0.0,
        val_split=1.0
    )
    
    # Use original data ranges for loss calculation
    irr_min, irr_max = 1459.0, 1800.0
    
    print(f"\nUsing original data range for loss calculation:")
    print(f"Irrigation range: {irr_min:.1f} to {irr_max:.1f}\n")
    
    # Initialize model
    model = IrrigationModel(pretrained=True).to(device)
    criterion = IrrigationLoss(irr_min, irr_max)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Print shapes for first batch in first epoch
        for images, texture_features, _, irrigation_labels in train_loader:
            if epoch == 0:
                print("\nVerifying feature dimensions:")
                print(f"Images shape: {images.shape}")
                print(f"Texture features shape: {texture_features.shape}")
                print(f"Irrigation labels shape: {irrigation_labels.shape}\n")
            break
        
        for images, texture_features, _, irrigation_labels in train_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            irrigation_labels = irrigation_labels.to(device)
            
            optimizer.zero_grad()
            irrigation_pred = model(images, texture_features)
            loss = criterion(irrigation_pred, irrigation_labels.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, texture_features, _, irrigation_labels in val_loader:
                images = images.to(device)
                texture_features = texture_features.to(device)
                irrigation_labels = irrigation_labels.to(device)
                
                irrigation_pred = model(images, texture_features)
                loss = criterion(irrigation_pred, irrigation_labels.squeeze())
                
                val_losses.append(loss.item())
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_irrigation_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Generate training curves
    plt.figure(figsize=(12, 8))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Irrigation Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('irrigation_训练曲线.png')
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_irrigation_model.pth'))
    model.eval()
    
    # Evaluate on test set
    test_predictions = []
    test_true = []
    
    with torch.no_grad():
        for images, texture_features, _, irrigation_labels in test_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            irrigation_pred = model(images, texture_features)
            
            test_predictions.extend(irrigation_pred.cpu().numpy())
            test_true.extend(irrigation_labels.numpy())
    
    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(test_true, test_predictions, alpha=0.5)
    plt.plot([min(test_true), max(test_true)],
             [min(test_true), max(test_true)], 'r--')
    plt.title('Irrigation: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig('irrigation_预测值vs真实值散点图.png')
    plt.close()
    
    # Generate residual analysis
    residuals = np.array(test_predictions) - np.array(test_true)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(test_predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Irrigation: Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('irrigation_残差分析.png')
    plt.close()

if __name__ == '__main__':
    excel_path = '/home/ubuntu/attachments/11.xlsx'
    image_dir = '/home/ubuntu/attachments/img_xin'
    train_model(excel_path, image_dir, num_epochs=50, batch_size=32, learning_rate=0.001)
