import torch
import torch.nn as nn
import torchvision.models as models

class RGBRegressionModel(nn.Module):
    def __init__(self, texture_dim=31, pretrained=True):  # Actual texture feature dimension from preprocessor
        super(RGBRegressionModel, self).__init__()
        
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
        
        # Add regression heads for water saving and irrigation
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 outputs: water saving and irrigation
        )  # Removed Sigmoid to allow unconstrained regression
        
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
        
        # Get predictions from regression head
        predictions = self.regression_head(combined_features)
        
        # Split predictions into water saving and irrigation
        water_saving = predictions[:, 0]
        irrigation = predictions[:, 1]
        
        return water_saving, irrigation

class RegressionLoss(nn.Module):
    def __init__(self, water_min, water_max, irr_min, irr_max, water_weight=0.5):
        super(RegressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.water_weight = water_weight
        
        # Store normalization ranges
        self.water_min = water_min
        self.water_max = water_max
        self.irr_min = irr_min
        self.irr_max = irr_max
    
    def normalize(self, water_val, irr_val):
        water_norm = (water_val - self.water_min) / (self.water_max - self.water_min)
        irr_norm = (irr_val - self.irr_min) / (self.irr_max - self.irr_min)
        return water_norm, irr_norm
    
    def denormalize(self, water_norm, irr_norm):
        water_val = water_norm * (self.water_max - self.water_min) + self.water_min
        irr_val = irr_norm * (self.irr_max - self.irr_min) + self.irr_min
        return water_val, irr_val
    
    def forward(self, water_pred, water_true, irr_pred, irr_true):
        # Normalize predictions (since model outputs raw values)
        water_pred_norm, irr_pred_norm = self.normalize(water_pred, irr_pred)
        
        # Calculate losses using normalized values
        water_loss = self.mse(water_pred_norm, water_true)
        irr_loss = self.mse(irr_pred_norm, irr_true)
        
        # Weighted sum of losses
        total_loss = self.water_weight * water_loss + (1 - self.water_weight) * irr_loss
        
        return total_loss, water_loss, irr_loss
