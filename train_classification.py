import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from classification_model_v2 import RGBClassificationModel, ClassificationLoss, get_class_boundaries, assign_bin_labels
from preprocess_images_v2 import ImagePreprocessor, prepare_dataset
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, water_labels, irrigation_labels, water_bins, irrigation_bins, preprocessor, augment=False):
        self.image_paths = image_paths
        self.water_labels = water_labels
        self.irrigation_labels = irrigation_labels
        self.preprocessor = preprocessor
        self.augment = augment
        
        # Values are already in original scale from prepare_dataset
        # Convert continuous values to bin labels using the provided bins
        self.water_classes = assign_bin_labels(water_labels, water_bins)
        self.irrigation_classes = assign_bin_labels(irrigation_labels, irrigation_bins)
        
        # Print class distributions during initialization
        print("\nClass distributions in dataset:")
        print("Water classes:", np.bincount(self.water_classes, minlength=5))
        print("Irrigation classes:", np.bincount(self.irrigation_classes, minlength=5))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image with texture features
        image, texture_features = self.preprocessor.preprocess_image(self.image_paths[idx], augment=self.augment)
        image = torch.FloatTensor(image.transpose(2, 0, 1))
        texture_features = torch.FloatTensor(texture_features)
        
        # Get class labels
        water_class = torch.LongTensor([self.water_classes[idx]])
        irrigation_class = torch.LongTensor([self.irrigation_classes[idx]])
        
        return image, texture_features, water_class, irrigation_class

def train_classification_model(excel_path, image_dir, num_epochs=100, batch_size=32, learning_rate=0.001, num_classes=5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Prepare dataset
    dataset = prepare_dataset(excel_path, image_dir, preprocessor)
    
    # Get percentile-based class boundaries
    water_bins, irrigation_bins = get_class_boundaries()
    
    print("\nWater saving bins (percentile-based):", water_bins)
    print("Irrigation bins (percentile-based):", irrigation_bins)
    
    # Print class distributions
    water_classes = assign_bin_labels(dataset['train'][1], water_bins)
    irr_classes = assign_bin_labels(dataset['train'][2], irrigation_bins)
    
    print("\nInitial class distributions:")
    print("Water saving classes:", np.bincount(water_classes, minlength=num_classes))
    print("Irrigation classes:", np.bincount(irr_classes, minlength=num_classes))
    
    # Print value ranges before binning
    print("\nValue ranges before binning:")
    print(f"Water saving: {min(dataset['train'][1])} to {max(dataset['train'][1])}")
    print(f"Irrigation: {min(dataset['train'][2])} to {max(dataset['train'][2])}")
    
    # Create data loaders with all valid pairs (filtering done in prepare_dataset)
    
    # Create data loaders
    train_dataset = ImageClassificationDataset(
        dataset['train'][0], dataset['train'][1], dataset['train'][2],
        water_bins, irrigation_bins, preprocessor, augment=True
    )
    val_dataset = ImageClassificationDataset(
        dataset['val'][0], dataset['val'][1], dataset['val'][2],
        water_bins, irrigation_bins, preprocessor, augment=False
    )
    test_dataset = ImageClassificationDataset(
        dataset['test'][0], dataset['test'][1], dataset['test'][2],
        water_bins, irrigation_bins, preprocessor, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = RGBClassificationModel(num_classes, num_classes, texture_dim=31, pretrained=True).to(device)
    criterion = ClassificationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_water_acc': [], 'val_water_acc': [],
        'train_irr_acc': [], 'val_irr_acc': [],
        'train_water_f1': [], 'val_water_f1': [],
        'train_irr_f1': [], 'val_irr_f1': []
    }
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_water_correct = 0
        train_irr_correct = 0
        train_total = 0
        
        for batch_idx, (images, texture_features, water_labels, irr_labels) in enumerate(train_loader):
            images = images.to(device)
            texture_features = texture_features.to(device)
            water_labels = water_labels.to(device).squeeze()
            irr_labels = irr_labels.to(device).squeeze()
            
            optimizer.zero_grad()
            water_pred, irr_pred = model(images, texture_features)
            loss, water_loss, irr_loss = criterion(
                water_pred, water_labels,
                irr_pred, irr_labels
            )
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Print class distributions periodically
            if batch_idx % 10 == 0:
                print(f"\nBatch {batch_idx} class distributions:")
                print("Water classes:", torch.bincount(water_labels, minlength=5).tolist())
                print("Irrigation classes:", torch.bincount(irr_labels, minlength=5).tolist())
            
            # Calculate accuracy
            _, water_predicted = torch.max(water_pred.data, 1)
            _, irr_predicted = torch.max(irr_pred.data, 1)
            train_total += water_labels.size(0)
            train_water_correct += (water_predicted == water_labels).sum().item()
            train_irr_correct += (irr_predicted == irr_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_losses = []
        val_water_correct = 0
        val_irr_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, texture_features, water_labels, irr_labels in val_loader:
                images = images.to(device)
                texture_features = texture_features.to(device)
                water_labels = water_labels.to(device).squeeze()
                irr_labels = irr_labels.to(device).squeeze()
                
                water_pred, irr_pred = model(images, texture_features)
                loss, water_loss, irr_loss = criterion(
                    water_pred, water_labels,
                    irr_pred, irr_labels
                )
                
                val_losses.append(loss.item())
                
                # Calculate accuracy
                _, water_predicted = torch.max(water_pred.data, 1)
                _, irr_predicted = torch.max(irr_pred.data, 1)
                val_total += water_labels.size(0)
                val_water_correct += (water_predicted == water_labels).sum().item()
                val_irr_correct += (irr_predicted == irr_labels).sum().item()
        
        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_water_acc = 100 * train_water_correct / train_total
        train_irr_acc = 100 * train_irr_correct / train_total
        val_water_acc = 100 * val_water_correct / val_total
        val_irr_acc = 100 * val_irr_correct / val_total
        
        # Calculate F1 scores
        train_water_f1 = f1_score(water_labels.cpu(), water_predicted.cpu(), average='macro')
        train_irr_f1 = f1_score(irr_labels.cpu(), irr_predicted.cpu(), average='macro')
        val_water_f1 = f1_score(water_labels.cpu(), water_predicted.cpu(), average='macro')
        val_irr_f1 = f1_score(irr_labels.cpu(), irr_predicted.cpu(), average='macro')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_water_acc'].append(train_water_acc)
        history['val_water_acc'].append(val_water_acc)
        history['train_irr_acc'].append(train_irr_acc)
        history['val_irr_acc'].append(val_irr_acc)
        history['train_water_f1'].append(train_water_f1)
        history['val_water_f1'].append(val_water_f1)
        history['train_irr_f1'].append(train_irr_f1)
        history['val_irr_f1'].append(val_irr_f1)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc - Water: {train_water_acc:.2f}%, F1: {train_water_f1:.3f}, Irrigation: {train_irr_acc:.2f}%, F1: {train_irr_f1:.3f}")
        print(f"Val Acc - Water: {val_water_acc:.2f}%, F1: {val_water_f1:.3f}, Irrigation: {val_irr_acc:.2f}%, F1: {val_irr_f1:.3f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_classification_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Generate training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_water_acc'], label='Train Water Acc')
    plt.plot(history['val_water_acc'], label='Val Water Acc')
    plt.plot(history['train_irr_acc'], label='Train Irrigation Acc')
    plt.plot(history['val_irr_acc'], label='Val Irrigation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('分类训练曲线.png')
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_classification_model.pth'))
    model.eval()
    
    # Evaluate on test set
    test_water_preds = []
    test_water_true = []
    test_irr_preds = []
    test_irr_true = []
    
    with torch.no_grad():
        for images, texture_features, water_labels, irr_labels in test_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            water_pred, irr_pred = model(images, texture_features)
            
            _, water_predicted = torch.max(water_pred.data, 1)
            _, irr_predicted = torch.max(irr_pred.data, 1)
            
            test_water_preds.extend(water_predicted.cpu().numpy())
            test_water_true.extend(water_labels.numpy().squeeze())
            test_irr_preds.extend(irr_predicted.cpu().numpy())
            test_irr_true.extend(irr_labels.numpy().squeeze())
    
    # Generate confusion matrices
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    cm_water = confusion_matrix(test_water_true, test_water_preds)
    sns.heatmap(cm_water, annot=True, fmt='d', cmap='Blues')
    plt.title('Water Saving Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    cm_irr = confusion_matrix(test_irr_true, test_irr_preds)
    sns.heatmap(cm_irr, annot=True, fmt='d', cmap='Blues')
    plt.title('Irrigation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('分类混淆矩阵.png')
    plt.close()
    
    # Print classification reports
    print("\nWater Saving Classification Report:")
    print(classification_report(test_water_true, test_water_preds))
    
    print("\nIrrigation Classification Report:")
    print(classification_report(test_irr_true, test_irr_preds))

if __name__ == '__main__':
    excel_path = '11.xlsx'
    image_dir = '所有数据/小区图像数据'
    train_classification_model(excel_path, image_dir)
