import torch
import torch.nn as nn
import torchvision.models as models

class SingleTaskClassificationModel(nn.Module):
    """Single-task RGB Classification model based on ResNet18 with texture feature processing.
    
    The model uses a pre-trained ResNet18 backbone and adds texture feature
    processing layers followed by a single classification head.
    """
    def __init__(self, num_classes: int = 5, num_texture_features: int = 31):
        super().__init__()
        
        # Load pre-trained ResNet18 with weights_only=True for security
        resnet = models.resnet18(pretrained=True, weights_only=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Texture feature processing
        self.texture_processor = nn.Sequential(
            nn.Linear(num_texture_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined feature dimension
        combined_features = 512 + 32  # ResNet18 features (512) + processed texture features (32)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images: torch.Tensor, texture_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            images: Batch of RGB images [batch_size, 3, height, width]
            texture_features: Batch of texture features [batch_size, num_texture_features]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Extract image features
        x = self.features(images)
        x = torch.flatten(x, 1)
        
        # Process texture features
        texture_processed = self.texture_processor(texture_features)
        
        # Combine features
        combined = torch.cat([x, texture_processed], dim=1)
        
        # Get class logits
        logits = self.classifier(combined)
        
        return logits

class RGBClassificationModel(nn.Module):
    """RGB Classification model based on ResNet18 with texture feature processing.
    
    The model uses a pre-trained ResNet18 backbone and adds texture feature
    processing layers followed by classification heads for water saving and
    irrigation prediction.
    """
    def __init__(self, num_texture_features: int = 31):
        super().__init__()
        
        # Load pre-trained ResNet18 with weights_only=True for security
        resnet = models.resnet18(pretrained=True, weights_only=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Texture feature processing
        self.texture_processor = nn.Sequential(
            nn.Linear(num_texture_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined feature dimension
        combined_features = 512 + 32  # ResNet18 features (512) + processed texture features (32)
        
        # Classification heads (5 classes each)
        self.water_classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # 5 classes for water saving
        )
        
        self.irrigation_classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # 5 classes for irrigation
        )
    
    def forward(self, images: torch.Tensor, texture_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            images: Batch of RGB images [batch_size, 3, height, width]
            texture_features: Batch of texture features [batch_size, num_texture_features]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Water saving and irrigation class logits
        """
        # Extract image features
        x = self.features(images)
        x = torch.flatten(x, 1)
        
        # Process texture features
        texture_processed = self.texture_processor(texture_features)
        
        # Combine features
        combined = torch.cat([x, texture_processed], dim=1)
        
        # Get class logits
        water_logits = self.water_classifier(combined)
        irrigation_logits = self.irrigation_classifier(combined)
        
        return water_logits, irrigation_logits
