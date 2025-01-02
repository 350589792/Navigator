import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd

class RGBClassificationModel(nn.Module):
    def __init__(self, num_classes_water=5, num_classes_irrigation=5, texture_dim=31, pretrained=True):
        super(RGBClassificationModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add texture processing branch
        self.texture_branch = nn.Sequential(
            nn.Linear(texture_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Combined feature dimension
        combined_dim = in_features + 32
        
        # Add classification heads for water saving and irrigation
        self.water_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_water)
        )
        
        self.irrigation_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_irrigation)
        )
        
        # Initialize the new layers
        for module in [self.texture_branch, self.water_head, self.irrigation_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, texture_features):
        # Extract features using ResNet backbone
        image_features = self.backbone(x)
        
        # Process texture features
        texture_output = self.texture_branch(texture_features)
        
        # Combine features
        combined_features = torch.cat([image_features, texture_output], dim=1)
        
        # Get predictions from classification heads
        water_pred = self.water_head(combined_features)
        irrigation_pred = self.irrigation_head(combined_features)
        
        return water_pred, irrigation_pred

class ClassificationLoss(nn.Module):
    def __init__(self, water_weight=0.5):
        super(ClassificationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.water_weight = water_weight
    
    def forward(self, water_pred, water_true, irr_pred, irr_true):
        water_loss = self.ce(water_pred, water_true)
        irr_loss = self.ce(irr_pred, irr_true)
        
        # Weighted sum of losses
        total_loss = self.water_weight * water_loss + (1 - self.water_weight) * irr_loss
        
        return total_loss, water_loss, irr_loss

def get_class_boundaries():
    """Get the percentile-based class boundaries for water saving and irrigation"""
    # Read data to calculate percentile-based boundaries
    df = pd.read_excel('11.xlsx')
    valid_df = df.dropna(subset=['节水', '灌溉'])
    
    # Calculate percentile-based bins (20th, 40th, 60th, 80th percentiles)
    water_bins = np.percentile(valid_df['节水'], [0, 20, 40, 60, 80, 100])
    irr_bins = np.percentile(valid_df['灌溉'], [0, 20, 40, 60, 80, 100])
    
    return water_bins, irr_bins

def create_value_bins(values=None, num_bins=5, type='water'):
    """Return pre-defined bins for water saving or irrigation values
    
    Args:
        values: Optional array of values (kept for backwards compatibility)
        num_bins: Number of bins (kept for backwards compatibility)
        type: Either 'water' or 'irrigation' to specify which bins to return
    
    Returns:
        numpy array of bin boundaries
    """
    water_bins, irr_bins = get_class_boundaries()
    if type == 'water':
        return water_bins
    elif type == 'irrigation':
        return irr_bins
    else:
        raise ValueError(f"Invalid type '{type}'. Must be 'water' or 'irrigation'.")

def assign_bin_labels(values, bins=None, type='water'):
    """Assign bin labels (0 to num_bins-1) to continuous values
    
    Args:
        values: Array of continuous values to bin
        bins: Optional pre-computed bin boundaries
        type: Either 'water' or 'irrigation' to specify which bins to use if bins=None
    
    Returns:
        numpy array of bin labels (0-4)
    """
    if bins is None:
        bins = create_value_bins(type=type)
    
    # Ensure values are within valid ranges
    values = np.clip(values, bins[0], bins[-1])
    
    # Assign to bins (0-4)
    labels = np.zeros_like(values, dtype=int)
    for i in range(len(bins)-1):
        mask = (values >= bins[i]) & (values <= bins[i+1])
        labels[mask] = i
        
    return labels

def get_class_names():
    """Get human-readable class names for both water saving and irrigation"""
    class_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    return class_names
