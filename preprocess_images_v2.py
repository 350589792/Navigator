import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class ImagePreprocessor:
    def __init__(self):
        self.transform = A.Compose([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.Resize(224, 224, always_apply=True),
            A.Normalize(always_apply=True),
        ])
        
        self.augmentation = A.Compose([
            # Existing spatial transforms
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            
            # Color transformations (removed RandomGamma)
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        ])

    def enhance_image(self, image):
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            image = clahe.apply(image)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        
        return image

    def remove_black_borders(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Add small padding
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            return cropped
        return image

    def compute_texture_features(self, image):
        """Compute texture features using LBP and GLCM"""
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure proper scaling for GLCM computation
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Compute LBP features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        
        try:
            # Compute GLCM features with reduced levels for efficiency
            distances = [1, 2]  # Reduced distances
            angles = [0, np.pi/2]  # Reduced angles
            levels = 32  # Reduced quantization levels
            
            # Rescale image to fewer gray levels
            gray_scaled = (gray / 255 * (levels - 1)).astype(np.uint8)
            glcm = graycomatrix(gray_scaled, distances, angles, levels, symmetric=True, normed=True)
            
            # Calculate GLCM properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
        except Exception as e:
            print(f"Warning: GLCM computation failed, using zeros. Error: {e}")
            contrast = dissimilarity = homogeneity = energy = correlation = 0.0
        
        # Combine all texture features
        glcm_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
        
        return np.concatenate([lbp_hist, glcm_features])

    def preprocess_image(self, image_path, augment=False):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhancement
        image = self.enhance_image(image)
        
        # Remove black borders
        image = self.remove_black_borders(image)
        
        # Compute texture features before normalization
        texture_features = self.compute_texture_features(image)
        
        # Apply base transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Apply augmentation if requested
        if augment:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        return image, texture_features

def prepare_dataset(excel_path, image_dir, preprocessor, test_size=0.1, val_size=0.2):
    """Prepare dataset with train/val/test split"""
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Filter out rows with NaN values and create a copy
    valid_df = df.dropna(subset=['节水', '灌溉']).copy()
    print(f"\nFound {len(valid_df)} valid rows in Excel file")
    print(f"Water saving range: {valid_df['节水'].min():.1f} to {valid_df['节水'].max():.1f}")
    print(f"Irrigation range: {valid_df['灌溉'].min():.1f} to {valid_df['灌溉'].max():.1f}")
    
    # Store original values for binning
    water_min, water_max = valid_df['节水'].min(), valid_df['节水'].max()
    irr_min, irr_max = valid_df['灌溉'].min(), valid_df['灌溉'].max()
    
    # Keep original values for binning
    valid_df.loc[:, '节水_orig'] = valid_df['节水'].copy()
    valid_df.loc[:, '灌溉_orig'] = valid_df['灌溉'].copy()
    
    # Normalize target values for model training
    valid_df.loc[:, '节水_norm'] = (valid_df['节水'] - water_min) / (water_max - water_min)
    valid_df.loc[:, '灌溉_norm'] = (valid_df['灌溉'] - irr_min) / (irr_max - irr_min)
    
    # Verify normalization
    assert valid_df['节水_norm'].min() >= 0.0 and valid_df['节水_norm'].max() <= 1.0
    assert valid_df['灌溉_norm'].min() >= 0.0 and valid_df['灌溉_norm'].max() <= 1.0
    
    print("\nValue ranges:")
    print(f"Water saving original: {valid_df['节水_orig'].min():.1f} to {valid_df['节水_orig'].max():.1f}")
    print(f"Irrigation original: {valid_df['灌溉_orig'].min():.1f} to {valid_df['灌溉_orig'].max():.1f}")
    print(f"Water saving normalized: {valid_df['节水_norm'].min():.3f} to {valid_df['节水_norm'].max():.3f}")
    print(f"Irrigation normalized: {valid_df['灌溉_norm'].min():.3f} to {valid_df['灌溉_norm'].max():.3f}")
    
    # Get image paths and labels
    image_paths = []
    labels_water = []
    labels_irrigation = []
    
    # Store normalization parameters
    global_params = {
        'water_min': water_min,
        'water_max': water_max,
        'irr_min': irr_min,
        'irr_max': irr_max
    }
    
    # Use the provided image directory path directly
    image_dir = Path(image_dir)
    print(f"\nLooking for images in: {image_dir}")
    
    # Create a mapping of water saving values to irrigation values using original values
    water_to_irrigation = dict(zip(valid_df['节水_orig'], valid_df['灌溉_orig']))
    
    # Track duplicates
    seen_pairs = set()
    processed_images = {}  # Track processed images and their values
    
    # Initialize counters
    matched_count = 0
    skipped_count = 0
    
    for img_path in sorted(Path(image_dir).rglob('*.jpg')):  # Sort for deterministic processing
        # Skip hidden files and thumbnails
        if img_path.name.startswith('.') or img_path.name == 'Thumbs.db':
            continue
            
        try:
            # Extract ID from filename (which matches water saving value)
            filename = img_path.stem
            if '的副本' in filename:
                img_id = float(filename.split('的')[0])
            else:
                img_id = float(filename)
            
            # Find matching water saving value with exact match
            matched = False
            for water_val, irr_val in water_to_irrigation.items():
                if abs(img_id - water_val) < 0.1:  # Require more precise matching
                    # Check if values are within valid ranges
                    if not (559.0 <= water_val <= 900.0):
                        print(f"Warning: Water saving value {water_val} out of valid range [559, 900] for image {img_path.name}")
                        break
                        
                    if not (1459.0 <= irr_val <= 1800.0):
                        print(f"Warning: Irrigation value {irr_val} out of valid range [1459, 1800] for image {img_path.name}")
                        break
                    
                    # Check for duplicates
                    pair_key = (water_val, irr_val)
                    if pair_key in seen_pairs:
                        print(f"Warning: Duplicate value pair found for image {img_path.name} (water: {water_val}, irrigation: {irr_val})")
                        # If we already have this pair, only keep the image if it's significantly different
                        if pair_key in processed_images:
                            print(f"Skipping duplicate image {img_path.name} as values already exist")
                            break
                    
                    # Store both original and normalized values
                    water_norm = (float(water_val) - global_params['water_min']) / (global_params['water_max'] - global_params['water_min'])
                    irr_norm = (float(irr_val) - global_params['irr_min']) / (global_params['irr_max'] - global_params['irr_min'])
                    
                    image_paths.append(str(img_path))
                    # Store original values for binning
                    labels_water.append(water_val)  # Original value for binning
                    labels_irrigation.append(irr_val)  # Original value for binning
                    print(f"Matched image {img_path.name} (ID: {img_id}) with water saving value: {water_val:.1f} (norm: {water_norm:.3f})")
                    
                    # Record the processed pair and image
                    seen_pairs.add(pair_key)
                    processed_images[pair_key] = str(img_path)
                    matched_count += 1
                    matched = True
                    break
            
            if not matched:
                print(f"Warning: No matching water saving value found for image {img_path.name} (ID: {img_id})")
                skipped_count += 1
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process filename {img_path.name}: {e}")
            skipped_count += 1
            
    print(f"\nProcessing Summary:")
    print(f"Total images processed: {matched_count}")
    print(f"Images skipped: {skipped_count}")
    print(f"Unique value pairs: {len(seen_pairs)}")
    print(f"\nFound {len(image_paths)} valid images with matching labels")
    
    # Print value ranges for verification
    water_vals = np.array(labels_water)
    irr_vals = np.array(labels_irrigation)
    print(f"\nActual value ranges:")
    print(f"Water saving: {water_vals.min():.1f} to {water_vals.max():.1f}")
    print(f"Irrigation: {irr_vals.min():.1f} to {irr_vals.max():.1f}")
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found with matching labels. Please check the data.")
    
    # Convert to numpy arrays
    X = np.array(image_paths)
    y_water = np.array(labels_water)
    y_irrigation = np.array(labels_irrigation)
    
    # First split into temp and test
    X_temp, X_test, y_water_temp, y_water_test, y_irr_temp, y_irr_test = train_test_split(
        X, y_water, y_irrigation, test_size=test_size, random_state=42
    )
    
    # Then split temp into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_water_train, y_water_val, y_irr_train, y_irr_val = train_test_split(
        X_temp, y_water_temp, y_irr_temp, test_size=val_ratio, random_state=42
    )
    
    return {
        'train': (X_train, y_water_train, y_irr_train),
        'val': (X_val, y_water_val, y_irr_val),
        'test': (X_test, y_water_test, y_irr_test)
    }
