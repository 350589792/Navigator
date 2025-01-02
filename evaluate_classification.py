import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.classification_model import RGBClassificationModel
from utils.create_dataloaders import create_dataloaders

def evaluate_model(model_path, task='water_saving'):
    """Evaluate classification model performance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBClassificationModel(num_texture_features=33).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, val_loader = create_dataloaders(task=task, batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, texture_features in val_loader:
            images = images.to(device)
            texture_features = texture_features.to(device)
            outputs = model(images, texture_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{task.replace("_", " ").title()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{task}_confusion_matrix.png')
    plt.close()
    
    print(f"\n{task.replace('_', ' ').title()} Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == '__main__':
    # Evaluate both models
    evaluate_model('best_water_saving_classification.pth', 'water_saving')
    evaluate_model('best_irrigation_classification.pth', 'irrigation')
