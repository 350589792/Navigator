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
        
        self.num_texture_features = num_texture_features
        
        # Load pre-trained ResNet34 for increased capacity
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Enhanced feature extraction with regularization
        modules = list(resnet.children())[:-2]  # Remove final pooling and fc
        modules.append(nn.BatchNorm2d(512))  # Normalize features
        modules.append(nn.Dropout2d(0.4))    # Spatial dropout
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*modules)
        
        # Simplified texture feature processing with batch normalization
        self.texture_processor = nn.Sequential(
            nn.Linear(num_texture_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_features = 512 + 32  # ResNet18 features (512) + processed texture features (32)
        
        # Enhanced classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
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
        
        # Process texture features (ensure float32 dtype and proper shape)
        texture_features = texture_features.to(dtype=torch.float32)
        
        # Handle single sample case (no batch dimension)
        if texture_features.dim() == 1:
            texture_features = texture_features.unsqueeze(0)
        
        # Handle case where batch dimension is second
        if texture_features.dim() == 2 and texture_features.shape[1] != self.num_texture_features:
            texture_features = texture_features.t()
        
        # Process features through texture network
        texture_processed = self.texture_processor(texture_features)
        
        # Expand to match batch size if needed
        if texture_processed.size(0) == 1 and x.size(0) > 1:
            texture_processed = texture_processed.expand(x.size(0), -1)
        
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
        
        self.num_texture_features = num_texture_features
        
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Texture feature processing (match input dimension of 31)
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
        
        # Process texture features (ensure float32 dtype and proper shape)
        print(f"Initial texture_features shape: {texture_features.shape}")
        print(f"Initial texture_features: {texture_features}")
        
        # Convert to float32
        texture_features = texture_features.to(dtype=torch.float32)
        
        # Handle single sample case (no batch dimension)
        if texture_features.dim() == 1:
            texture_features = texture_features.unsqueeze(0)
        
        # Handle case where batch dimension is second
        if texture_features.dim() == 2 and texture_features.shape[1] != self.num_texture_features:
            texture_features = texture_features.t()
        
        print(f"Reshaped texture_features shape: {texture_features.shape}")
        
        # Process features through texture network
        texture_processed = self.texture_processor(texture_features)
        print(f"After texture_processor shape: {texture_processed.shape}")
        
        # Expand to match batch size if needed
        if texture_processed.size(0) == 1 and x.size(0) > 1:
            texture_processed = texture_processed.expand(x.size(0), -1)
        print(f"Final texture_processed shape: {texture_processed.shape}")
        
        # Combine features
        combined = torch.cat([x, texture_processed], dim=1)
        
        # Get class logits
        water_logits = self.water_classifier(combined)
        irrigation_logits = self.irrigation_classifier(combined)
        
        return water_logits, irrigation_logits
