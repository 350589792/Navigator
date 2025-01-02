import torch
import torch.nn as nn
import torchvision.models as models

class WaterSavingModel(nn.Module):
    def __init__(self, texture_dim=33, pretrained=True):
        super(WaterSavingModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the last fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add texture feature processing branch
        self.texture_branch = nn.Sequential(
            nn.Linear(texture_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Combined features dimension
        combined_dim = in_features + 32  # CNN features + processed texture features
        
        # Add regression head for water saving only
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Single output for water saving
        )
        
        # Initialize the new layers
        for module in [self.texture_branch, self.regression_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, texture_features):
        # Extract features using ResNet backbone
        cnn_features = self.backbone(x)
        
        # Process texture features
        processed_texture = self.texture_branch(texture_features)
        
        # Concatenate CNN and texture features
        combined_features = torch.cat([cnn_features, processed_texture], dim=1)
        
        # Get prediction from regression head
        water_saving = self.regression_head(combined_features)
        
        return water_saving.squeeze()

class WaterSavingLoss(nn.Module):
    def __init__(self, water_min, water_max):
        super(WaterSavingLoss, self).__init__()
        self.mse = nn.MSELoss()
        
        # Store normalization ranges
        self.water_min = water_min
        self.water_max = water_max
    
    def normalize(self, water_val):
        return (water_val - self.water_min) / (self.water_max - self.water_min)
    
    def denormalize(self, water_norm):
        return water_norm * (self.water_max - self.water_min) + self.water_min
    
    def forward(self, water_pred, water_true):
        # Normalize both predictions and true values
        water_pred_norm = self.normalize(water_pred)
        water_true_norm = self.normalize(water_true)
        
        # Calculate loss using normalized values
        water_loss = self.mse(water_pred_norm, water_true_norm)
        
        return water_loss
