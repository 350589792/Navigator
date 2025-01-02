import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocess_images_v2 import ImagePreprocessor, extract_texture_features
from utils.binning import create_class_bins

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

class RGBDataset(Dataset):
    def __init__(self, image_paths, labels, preprocessor):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        processed_image, texture_features = self.preprocessor.process_image(image)
        
        return processed_image, torch.tensor(label, dtype=torch.float32), torch.tensor(texture_features, dtype=torch.float32)

class RGBClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, preprocessor, num_classes=5):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)  # Convert to numpy array
        
        # Extract texture features
        texture_features = extract_texture_features(image)
        
        # Process image for model input
        if self.preprocessor and self.preprocessor.transform:
            transformed = self.preprocessor.transform(image=image)
            image = transformed['image']
        else:
            # Default normalization if no transform
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = transform(image=image)
            image = transformed['image']
        
        # Convert label to class index
        class_idx = torch.tensor(label, dtype=torch.long)
        
        # Log debug info for first few items
        if idx < 5:
            logger.debug(f"Processing item {idx}:")
            logger.debug(f"  Image path: {img_path}")
            logger.debug(f"  Label: {label}")
            logger.debug(f"  Texture features shape: {texture_features.shape}")
            logger.debug(f"  Number of texture features: {len(texture_features)}")
        
        return image, class_idx, torch.tensor(texture_features, dtype=torch.float32)

def create_dataloaders(task='water_saving', batch_size=32, train_split=0.7, val_split=0.2, test_split=0.1, return_test=False):
    """Create train and validation dataloaders for the specified task.
    
    Args:
        task: Task name ('water_saving', 'irrigation', 'water_saving_class', 'irrigation_class')
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        return_test: If True, returns test loader instead of train loader
        
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader) or (test_loader, val_loader)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Load and preprocess data
    excel_path = '/home/ubuntu/attachments/11.xlsx'
    logger.info(f"Loading Excel file from: {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"Initial DataFrame shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Clean data
    # Map task names to column names
    task_to_column = {
        'water_saving': '节水',
        'irrigation': '灌溉',
        'water_saving_class': '节水',
        'irrigation_class': '灌溉'
    }
    column = task_to_column.get(task)
    if not column:
        raise ValueError(f"Unknown task: {task}")
    df = df.dropna(subset=[column])
    
    # Get image paths and labels
    image_dir = '/home/ubuntu/attachments/img_xin'
    print(f"\nSearching for images in: {image_dir}")
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Find all image files recursively
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # Map filenames to full paths
    image_name_to_path = {}
    print("\nMapping image files:")
    
    # Function to extract numeric part from filename
    def extract_number(filename):
        # Remove extension and '的副本' suffix
        name = os.path.splitext(filename)[0].replace('的副本', '')
        try:
            return float(name)
        except ValueError:
            return None
            
    def find_closest_image(target_num, image_name_to_path, max_diff=5):
        """Find the closest available image number within max_diff."""
        target_str = str(float(target_num))
        if target_str in image_name_to_path:
            return target_str
        
        # Try numbers within ±max_diff
        for diff in range(1, max_diff + 1):
            above = str(float(target_num + diff))
            below = str(float(target_num - diff))
            if above in image_name_to_path:
                return above
            if below in image_name_to_path:
                return below
        return None

    def convert_excel_id_to_image_numbers(excel_id, image_name_to_path):
        """Convert Excel ID to possible image numbers using known multiplication factors."""
        # Core multiplication factors that worked well
        core_factors = [12.8, 13.4, 14.0, 14.6, 15.2, 15.8]
        
        # Try each core factor and find closest match
        for factor in core_factors:
            img_num = round(excel_id * factor)
            closest = find_closest_image(img_num, image_name_to_path)
            if closest is not None:
                return float(closest)  # Return the actual matched image number
                
        # If no match found with core factors, try a wider range
        min_factor, max_factor = 11.0, 17.0
        step = 0.2
        for factor in [min_factor + i * step for i in range(int((max_factor - min_factor) / step) + 1)]:
            img_num = round(excel_id * factor)
            closest = find_closest_image(img_num, image_name_to_path)
            if closest is not None:
                return float(closest)
                
        return None  # No match found
    
    # Process all image files
    for f in image_files:
        basename = os.path.basename(f)
        number = extract_number(basename)
        if number is not None:
            image_name_to_path[str(number)] = f
            
    # Print first 5 mappings for debugging
    print("First 5 image mappings:")
    for k in list(image_name_to_path.keys())[:5]:
        print(f"ID: {k} -> File: {os.path.basename(image_name_to_path[k])}")
    
    print(f"\nFound {len(image_files)} image files")
    print("First 5 Excel identifiers:", df['Unnamed: 0'].head().tolist())
    
    # Get image paths and labels
    valid_data = []
    
    # Convert column values to numeric, dropping any non-numeric values
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Filter rows with valid values
    valid_mask = df[column].notna()
    valid_df = df[valid_mask].copy()
    print(f"\nValid rows for column '{column}': {len(valid_df)}")
    print(f"Value range: {valid_df[column].min():.2f} - {valid_df[column].max():.2f}")
    
    # Process each valid row
    print("\nProcessing rows:")
    valid_count = 0
    debug_count = 0
    for _, row in valid_df.iterrows():
        excel_id = float(row['Unnamed: 0'])
        matched_img_num = convert_excel_id_to_image_numbers(excel_id, image_name_to_path)
        
        if matched_img_num is not None:
            img_name = str(float(matched_img_num))
            if debug_count < 5:  # Log first 5 rows
                logger.debug(f"Row {debug_count}:")
                logger.debug(f"  Excel identifier: {excel_id}")
                logger.debug(f"  Matched image number: {matched_img_num}")
                logger.debug(f"  Using factor: {matched_img_num/excel_id:.2f}")
            debug_count += 1
            valid_count += 1
        
        if matched_img_num is None and debug_count < 5:
            logger.warning(f"Row {debug_count}:")
            logger.warning(f"  Excel identifier: {excel_id}")
            logger.warning(f"  No match found")
            logger.warning(f"  Available keys (first 5): {list(image_name_to_path.keys())[:5]}")
            debug_count += 1
    
    logger.info(f"Total valid matches found: {valid_count} out of {len(valid_df)} rows")
    
    # Log details of unmatched rows
    if valid_count < len(valid_df):
        logger.warning("Unmatched Excel IDs:")
        unmatched_count = 0
        for _, row in valid_df.iterrows():
            excel_id = float(row['Unnamed: 0'])
            matched_img_num = convert_excel_id_to_image_numbers(excel_id, image_name_to_path)
            if matched_img_num is None and unmatched_count < 5:
                logger.warning(f"Excel ID: {excel_id}")
                # Try core factors to show attempted matches
                core_factors = [12.8, 13.4, 14.0, 14.6, 15.2, 15.8]
                tried_numbers = [round(excel_id * factor) for factor in core_factors]
                logger.warning(f"Tried numbers with core factors: {tried_numbers}")
                # Show available numbers in a similar range
                min_num = min(tried_numbers)
                max_num = max(tried_numbers)
                nearby_available = []
                for num in range(min_num - 10, max_num + 11):
                    if str(float(num)) in image_name_to_path:
                        nearby_available.append(num)
                logger.warning(f"Available numbers in range: {sorted(nearby_available)[:10]}")
                unmatched_count += 1
    
    # Reset loop to collect data
    for _, row in valid_df.iterrows():
        excel_id = float(row['Unnamed: 0'])
        matched_img_num = convert_excel_id_to_image_numbers(excel_id, image_name_to_path)
        
        if matched_img_num is not None: 
            img_name = str(float(matched_img_num))
            label_value = float(row[column])
            
            # Convert to class index for classification tasks
            if task in ['water_saving_class', 'irrigation_class']:
                if task == 'water_saving_class':
                    # Binning for water saving (559-900)
                    bins = [0, 600, 650, 700, 750, float('inf')]  # Adjusted based on actual range
                else:
                    # Binning for irrigation (1459-1800)
                    bins = [0, 1550, 1600, 1650, 1700, float('inf')]
                if len(valid_data) < 5:  # Debug print for first few samples
                    logger.debug(f"Using bins for {task}: {bins}")
                    logger.debug(f"Value {label_value:.2f} -> Class {np.digitize(label_value, bins) - 1}")
                final_label = np.digitize(label_value, bins) - 1  # 0-based class index
            else:
                final_label = label_value
            
            valid_data.append({
                'path': image_name_to_path[img_name],
                'label': final_label
            })
    
    if not valid_data:
        raise ValueError(f"No valid data found for task {task}")
    
    logger.info(f"Collected {len(valid_data)} valid samples")
    logger.info("First 5 samples:")
    for i, data in enumerate(valid_data[:5]):
        logger.info(f"Sample {i+1}: Label = {data['label']}, Path = {data['path']}")
    
    # Convert to numpy arrays
    image_paths = [d['path'] for d in valid_data]
    if task in ['water_saving_class', 'irrigation_class']:
        labels = np.array([d['label'] for d in valid_data], dtype=np.int64)
    else:
        labels = np.array([d['label'] for d in valid_data], dtype=np.float32)
    
    # Create indices for train/val/test split
    indices = list(range(len(valid_data)))
    np.random.shuffle(indices)
    
    train_size = int(train_split * len(indices))
    val_size = int(val_split * len(indices))
    test_size = len(indices) - train_size - val_size
    
    # Always create all splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # If return_test is True, use test set for evaluation
    if return_test:
        # Keep original training/validation split, just enable test set
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        logger.info("Test set enabled with original train/val split maintained")
    
    logger.info("Split sizes:")
    logger.info(f"Total samples: {len(indices)}")
    logger.info(f"Training samples: {len(train_indices)}")
    logger.info(f"Validation samples: {len(val_indices)}")
    logger.info(f"Test samples: {len(test_indices)}")
    
    # Analyze class distribution
    all_labels = [d['label'] for d in valid_data]
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    logger.info("Class distribution in full dataset:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"Class {label}: {count} samples ({count/len(all_labels)*100:.1f}%)")
    
    # Define transforms
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
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
    
    # Create preprocessors with appropriate transforms
    train_preprocessor = ImagePreprocessor(transform=train_transform)
    val_preprocessor = ImagePreprocessor(transform=val_transform)
    
    # Create datasets
    if task in ['water_saving_class', 'irrigation_class']:
        # For classification tasks
        train_dataset = RGBClassificationDataset(
            [image_paths[i] for i in train_indices],
            [labels[i] for i in train_indices],
            train_preprocessor
        )
        val_dataset = RGBClassificationDataset(
            [image_paths[i] for i in val_indices],
            [labels[i] for i in val_indices],
            val_preprocessor
        )
    else:
        # For regression tasks
        train_dataset = RGBDataset(
            [image_paths[i] for i in train_indices],
            [labels[i] for i in train_indices],
            train_preprocessor
        )
        val_dataset = RGBDataset(
            [image_paths[i] for i in val_indices],
            [labels[i] for i in val_indices],
            val_preprocessor
        )
    
    logger.info("Dataset sizes:")
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
