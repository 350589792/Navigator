import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, Optional, Literal
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
        
        # Initialize counters as instance variables
        self.total_rows = 0
        self.skipped_rows = 0
        self.found_files = 0
        
        # Process each row
        print("\nProcessing Excel rows for dataset creation...")
        for idx, row in df.iterrows():
            self.total_rows += 1
            water_saving = row['节水']
            irrigation = row['灌溉']
            
            # Skip if either value is NaN
            if pd.isna(water_saving) or pd.isna(irrigation):
                self.skipped_rows += 1
                print(f"Skipping row {idx}: NaN values found")
                continue
            
            # Debug print for first few rows
            idx_num = int(idx) if isinstance(idx, (int, np.integer)) else 0
            if idx_num < 5:
                print(f"\nProcessing row {idx}:")
                print(f"Excel value: {row['Unnamed: 0']}")
                print(f"Water saving: {water_saving}")
                print(f"Irrigation: {irrigation}")
            
            # Convert to class labels if in classification mode
            if classification and self.binner is not None:
                water_class = self.binner.transform_water(np.array([water_saving]))[0]
                irr_class = self.binner.transform_irrigation(np.array([irrigation]))[0]
                # Try matching against both water saving and irrigation values
                img_name = None
                idx_num = int(idx) if isinstance(idx, (int, np.integer)) else 0
                
                # Values to try matching against
                values_to_try = [water_saving, irrigation]
                if idx_num < 5:  # Debug print for first few rows
                    print(f"Looking for image files with water_saving={water_saving}, irrigation={irrigation}")
                
                for val in values_to_try:
                    for name_format in [f"{int(val)}的副本.jpg", f"{val:.2f}的副本.jpg"]:
                        if idx_num < 5:  # Debug print for first few rows
                            print(f"Trying filename: {name_format}")
                        
                        for class_dir in ['class1', 'class2', 'class3', 'class4', 'class5']:
                            full_path = os.path.join('/home/ubuntu/attachments/img_xin', class_dir, name_format)
                            if idx_num < 5:  # Debug print for first few rows
                                print(f"Checking path: {full_path}")
                            
                            if os.path.exists(full_path):
                                img_name = name_format
                                self.found_files += 1
                                if idx_num < 5:  # Debug print for first few rows
                                    print(f"Found image at: {full_path}")
                                break
                        if img_name:
                            break
                    if img_name:
                        break
                if img_name is None:
                    if idx_num < 5:  # Debug print for first few rows
                        print(f"No matching image found for water_saving={water_saving}, irrigation={irrigation}")
                    self.skipped_rows += 1
                    continue  # Skip if no matching file found
                
                self.data.append({
                    'img_name': img_name,
                    'water_saving': water_class,
                    'irrigation': irr_class
                })
            else:
                # Try matching against both water saving and irrigation values
                img_name = None
                # Values to try matching against
                values_to_try = [water_saving, irrigation]
                
                for val in values_to_try:
                    for name_format in [f"{int(val)}的副本.jpg", f"{val:.2f}的副本.jpg"]:
                        for class_dir in ['class1', 'class2', 'class3', 'class4', 'class5']:
                            if os.path.exists(os.path.join('/home/ubuntu/attachments/img_xin', class_dir, name_format)):
                                img_name = name_format
                                break
                        if img_name:
                            break
                    if img_name:
                        break
                if img_name is None:
                    continue  # Skip if no matching file found
                
                self.data.append({
                    'img_name': img_name,
                    'water_saving': water_saving,
                    'irrigation': irrigation
                })
    
    def __len__(self) -> int:
        print(f"\nDataset statistics:")
        print(f"Total rows processed: {self.total_rows}")
        print(f"Rows skipped: {self.skipped_rows}")
        print(f"Files found: {self.found_files}")
        print(f"Final dataset size: {len(self.data)}")
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        # Images are organized in class subdirectories
        for class_dir in ['class1', 'class2', 'class3', 'class4', 'class5']:
            img_path = os.path.join('/home/ubuntu/attachments/img_xin', class_dir, item['img_name'])
            if os.path.exists(img_path):
                break
        
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
    
    def get_props(matrix, prop: Literal['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']) -> np.ndarray:
        return graycoprops(matrix, prop).ravel()[:2]
    
    props: list[Literal['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']] = [
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
    ]
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
