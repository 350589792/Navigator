import numpy as np
from binning import ValueBinner

def test_equal_width_binning():
    """Test equal width binning strategy."""
    binner = ValueBinner(method='equal_width')
    
    # Test water saving binning
    water_values = np.array([560, 650, 750, 850, 890])
    water_labels = binner.transform_water(water_values)
    print("\nEqual Width Binning Test:")
    print("Water Saving Values:", water_values)
    print("Assigned Labels:", water_labels)
    
    # Test irrigation binning
    irr_values = np.array([1460, 1550, 1650, 1750, 1790])
    irr_labels = binner.transform_irrigation(irr_values)
    print("\nIrrigation Values:", irr_values)
    print("Assigned Labels:", irr_labels)
    
    # Print bin ranges
    print("\nBin Ranges:")
    binner.print_bin_ranges()

def test_percentile_binning():
    """Test percentile-based binning strategy."""
    # Generate sample data
    np.random.seed(42)
    water_values = np.random.uniform(559, 900, 100)
    irr_values = np.random.uniform(1459, 1800, 100)
    
    binner = ValueBinner(method='percentile')
    binner.fit(water_values, irr_values)
    
    # Test binning
    water_labels = binner.transform_water(water_values)
    irr_labels = binner.transform_irrigation(irr_values)
    
    print("\nPercentile Binning Test:")
    print("Water Saving Label Distribution:")
    for i in range(5):
        count = np.sum(water_labels == i)
        print(f"Class {i}: {count} samples")
    
    print("\nIrrigation Label Distribution:")
    for i in range(5):
        count = np.sum(irr_labels == i)
        print(f"Class {i}: {count} samples")
    
    # Print bin ranges
    print("\nBin Ranges:")
    binner.print_bin_ranges()

if __name__ == '__main__':
    print("Testing Equal Width Binning...")
    test_equal_width_binning()
    
    print("\nTesting Percentile Binning...")
    test_percentile_binning()
