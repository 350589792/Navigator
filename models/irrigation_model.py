import torch
import torch.nn as nn
import torchvision.models as models

class IrrigationModel(nn.Module):
    def __init__(self, texture_dim=31, pretrained=True):
        super(IrrigationModel, self).__init__()
        
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
        
        # Add regression head for irrigation only
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Single output for irrigation
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
        irrigation = self.regression_head(combined_features)
        
        return irrigation.squeeze()

class IrrigationLoss(nn.Module):
    def __init__(self, irr_min, irr_max):
        super(IrrigationLoss, self).__init__()
        self.mse = nn.MSELoss()
        
        # Store normalization ranges
        self.irr_min = irr_min
        self.irr_max = irr_max
    
    def normalize(self, irr_val):
        return (irr_val - self.irr_min) / (self.irr_max - self.irr_min)
    
    def denormalize(self, irr_norm):
        return irr_norm * (self.irr_max - self.irr_min) + self.irr_min
    
    def forward(self, irr_pred, irr_true):
        # Normalize predictions (since model outputs raw values)
        irr_pred_norm = self.normalize(irr_pred)
        
        # Calculate loss using normalized values
        irr_loss = self.mse(irr_pred_norm, irr_true)
        
        return irr_loss
