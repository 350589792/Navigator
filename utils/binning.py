import numpy as np
from typing import List, Tuple, Dict

def create_class_bins(values: np.ndarray, task: str = 'water_saving') -> Dict[str, np.ndarray]:
    """Create classification bins for the given task.
    
    Args:
        values: Array of values to create bins for
        task: Either 'water_saving' or 'irrigation'
        
    Returns:
        Dict containing:
            'bins': Array of bin boundaries
            'labels': Array of class labels (0 to num_classes-1)
    """
    binner = ValueBinner(method='equal_width')
    
    if task == 'water_saving':
        labels = binner.transform_water(values)
        bins = binner.water_bins
    else:
        labels = binner.transform_irrigation(values)
        bins = binner.irr_bins
        
    return {
        'bins': bins,
        'labels': labels
    }

def create_equal_width_bins(min_val: float, max_val: float, num_bins: int = 5) -> np.ndarray:
    """Create equal-width bins between min and max values."""
    return np.linspace(min_val, max_val, num_bins + 1)

def create_percentile_bins(values: np.ndarray, num_bins: int = 5) -> np.ndarray:
    """Create bins based on percentiles of the data."""
    return np.percentile(values, np.linspace(0, 100, num_bins + 1))

def assign_bin_labels(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Assign bin labels (0 to num_bins-1) to values, ensuring valid class assignments.
    
    Args:
        values: Array of values to bin
        bins: Array of bin boundaries
        
    Returns:
        np.ndarray: Array of bin labels (0 to num_classes-1)
    """
    num_classes = len(bins) - 1
    values_clamped = np.clip(values, bins[0], bins[-1])
    
    # Initialize labels array
    labels = np.zeros_like(values_clamped, dtype=int)
    
    # Assign labels based on bin boundaries
    for i in range(num_classes):
        if i == num_classes - 1:
            # For the last bin, include the right boundary
            mask = (values_clamped >= bins[i]) & (values_clamped <= bins[i + 1])
        else:
            # For other bins, exclude the right boundary
            mask = (values_clamped >= bins[i]) & (values_clamped < bins[i + 1])
        labels[mask] = i
    
    return labels

class ValueBinner:
    def __init__(self, method: str = 'equal_width'):
        """Initialize binning for water saving and irrigation values.
        
        Based on data analysis, the following pre-computed bin boundaries are used:
        Water Saving Bins (559.0 to 900.0):
        - Class 0: 559.0 to 627.2 (Very Low)
        - Class 1: 627.2 to 695.4 (Low)
        - Class 2: 695.4 to 763.6 (Medium)
        - Class 3: 763.6 to 831.8 (High)
        - Class 4: 831.8 to 900.0 (Very High)
        
        Irrigation Bins (1459.0 to 1800.0):
        - Class 0: 1459.0 to 1527.2 (Very Low)
        - Class 1: 1527.2 to 1595.4 (Low)
        - Class 2: 1595.4 to 1663.6 (Medium)
        - Class 3: 1663.6 to 1731.8 (High)
        - Class 4: 1731.8 to 1800.0 (Very High)
        
        Args:
            method: Binning method ('equal_width' or 'percentile')
                   Note: For consistent results, 'equal_width' is recommended
                   as it uses pre-computed optimal bin boundaries.
        """
        self.method = method
        
        # Pre-computed optimal bin boundaries from data analysis
        self.water_bins = np.array([559.0, 627.2, 695.4, 763.6, 831.8, 900.0])
        self.irr_bins = np.array([1459.0, 1527.2, 1595.4, 1663.6, 1731.8, 1800.0])
        
        # Equal-width bins are pre-computed, only need fitting for percentile
        self.is_fitted = method == 'equal_width'
    
    def fit(self, water_values: np.ndarray | None = None, irr_values: np.ndarray | None = None):
        """Compute bin edges based on data if using percentile method.
        
        Args:
            water_values: Array of water saving values
            irr_values: Array of irrigation values
            
        Note:
            For consistent results across runs, it's recommended to use
            'equal_width' method which uses pre-computed optimal bin boundaries.
            The percentile method is provided for experimental purposes.
        """
        if self.method == 'percentile':
            if water_values is not None:
                self.water_bins = create_percentile_bins(water_values, num_bins=5)
            if irr_values is not None:
                self.irr_bins = create_percentile_bins(irr_values, num_bins=5)
            self.is_fitted = True
    
    def transform_water(self, values: np.ndarray) -> np.ndarray:
        """Transform water saving values to bin labels."""
        if not self.is_fitted:
            raise ValueError("Binner must be fitted first if using percentile method")
        return assign_bin_labels(values, self.water_bins)
    
    def transform_irrigation(self, values: np.ndarray) -> np.ndarray:
        """Transform irrigation values to bin labels."""
        if not self.is_fitted:
            raise ValueError("Binner must be fitted first if using percentile method")
        return assign_bin_labels(values, self.irr_bins)
    
    def get_bin_ranges(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get the range for each bin for both water saving and irrigation."""
        if not self.is_fitted:
            raise ValueError("Binner must be fitted first if using percentile method")
        
        water_ranges = [(self.water_bins[i], self.water_bins[i+1]) 
                       for i in range(len(self.water_bins)-1)]
        irr_ranges = [(self.irr_bins[i], self.irr_bins[i+1]) 
                     for i in range(len(self.irr_bins)-1)]
        
        return water_ranges, irr_ranges
    
    def print_bin_ranges(self):
        """Print the range for each bin."""
        water_ranges, irr_ranges = self.get_bin_ranges()
        
        print("Water Saving Bins:")
        for i, (start, end) in enumerate(water_ranges):
            print(f"Class {i}: {start:.1f} to {end:.1f}")
        
        print("\nIrrigation Bins:")
        for i, (start, end) in enumerate(irr_ranges):
            print(f"Class {i}: {start:.1f} to {end:.1f}")
