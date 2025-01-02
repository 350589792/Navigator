import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, Optional
from utils.binning import ValueBinner

class RGBDataset(Dataset):
    """Dataset for RGB images with water saving and irrigation values."""
    
    def __init__(
        self,
        excel_path: str,
        img_dir: str,
        transform: Optional[A.Compose] = None,
        classification: bool = False
    ):
        """Initialize dataset.
        
        Args:
            excel_path: Path to Excel file with labels
            img_dir: Directory containing images
            transform: Albumentations transforms
            classification: If True, return class labels instead of continuous values
        """
        self.img_dir = img_dir
        self.transform = transform
        self.classification = classification
        
        # Load Excel data
        df = pd.read_excel(excel_path)
        self.data = []
        
        # Initialize binner if in classification mode
        self.binner = ValueBinner() if classification else None
        
        # Process each row
        for _, row in df.iterrows():
            water_saving = row['节水']
            irrigation = row['灌溉']
            
            # Skip if either value is NaN
            if pd.isna(water_saving) or pd.isna(irrigation):
                continue
            
            # Convert to class labels if in classification mode
            if classification and self.binner is not None:
                water_class = self.binner.transform_water(np.array([water_saving]))[0]
                irr_class = self.binner.transform_irrigation(np.array([irrigation]))[0]
                self.data.append({
                    'img_name': f"{int(row['Unnamed: 0'])}.jpg",
                    'water_saving': water_class,
                    'irrigation': irr_class
                })
            else:
                self.data.append({
                    'img_name': f"{int(row['Unnamed: 0'])}.jpg",
                    'water_saving': water_saving,
                    'irrigation': irrigation
                })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['img_name'])
        
        # Read and process image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract texture features (31 features)
        texture_features = extract_texture_features(image)
        
        # Log texture feature shape for verification
        if idx == 0:  # Only log for first item to avoid spam
            print(f"Texture features shape: {texture_features.shape}")
            print(f"Number of texture features: {len(texture_features)}")
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert labels to tensors
        water_saving = torch.tensor(item['water_saving'])
        irrigation = torch.tensor(item['irrigation'])
        
        if self.classification:
            water_saving = water_saving.long()  # Convert to long for CrossEntropyLoss
            irrigation = irrigation.long()
        else:
            water_saving = water_saving.float()
            irrigation = irrigation.float()
        
        texture_features = torch.tensor(texture_features, dtype=torch.float32)
        
        return image, texture_features, water_saving, irrigation

def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """Extract texture features from image.
    
    Args:
        image: RGB image array
        
    Returns:
        Array of texture features
    """
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM features
    glcm = cv2.resize(gray, (32, 32))  # Resize for consistent feature size
    glcm = glcm.astype(np.float32) / 255.0  # Normalize
    
    # Extract various texture features (31 total)
    features = []
    
    # Basic statistical features (5)
    features.extend([
        np.mean(glcm),
        np.std(glcm),
        np.var(glcm),
        np.max(glcm) - np.min(glcm),  # Range
        np.median(glcm)
    ])
    
    # Gradient features (8)
    gx = cv2.Sobel(glcm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(glcm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    features.extend([
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.mean(np.abs(gx)),
        np.mean(np.abs(gy)),
        np.std(gx),
        np.std(gy),
        np.sum(mag > 0.1) / mag.size  # Edge density
    ])
    
    # Local binary pattern features (8)
    from skimage.feature import local_binary_pattern
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(glcm, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features.extend(hist)
    
    # Haralick features (10)
    from skimage.feature import graycomatrix, graycoprops
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_matrix = graycomatrix(
        (glcm * 255).astype(np.uint8),
        distances,
        angles,
        symmetric=True,
        normed=True
    )
    from typing import Literal
    
    def get_props(matrix, prop: Literal['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']):
        return graycoprops(matrix, prop).ravel()[:2]
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        features.extend(get_props(glcm_matrix, prop))
    
    return np.array(features, dtype=np.float32)

def prepare_dataset(
    excel_path: str,
    img_dir: str,
    batch_size: int = 32,
    classification: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train, validation, and test data loaders.
    
    Args:
        excel_path: Path to Excel file with labels
        img_dir: Directory containing images
        batch_size: Batch size for data loaders
        classification: If True, return class labels instead of continuous values
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transforms with enhanced color augmentation
    train_transform = A.Compose([
        # Geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        
        # Color and intensity transforms
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.2
        ),
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=0.2
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.2
        ),
        
        # Normalization (always applied)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Create datasets
    full_dataset = RGBDataset(
        excel_path=excel_path,
        img_dir=img_dir,
        transform=train_transform,
        classification=classification
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override transforms for validation and test sets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
