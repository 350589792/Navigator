import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binning import ValueBinner
import seaborn as sns
import os
import sys

def analyze_binning_distribution(excel_path: str):
    """Analyze the distribution of values in bins for both binning methods."""
    # Load data
    df = pd.read_excel(excel_path)
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Extract values, dropping any NaN
    water_values = np.asarray(df['节水'].dropna().values)
    irr_values = np.asarray(df['灌溉'].dropna().values)
    
    print(f"\nTotal samples after dropping NaN: {len(water_values)} water saving, {len(irr_values)} irrigation")
    
    # Create binners
    equal_width_binner = ValueBinner(method='equal_width')
    percentile_binner = ValueBinner(method='percentile')
    percentile_binner.fit(water_values, irr_values)
    
    # Convert to numpy arrays for analysis
    water_values_np = water_values.reshape(-1)
    irr_values_np = irr_values.reshape(-1)
    
    # Get labels for both methods
    water_labels_eq = equal_width_binner.transform_water(water_values)
    irr_labels_eq = equal_width_binner.transform_irrigation(irr_values)
    
    water_labels_pct = percentile_binner.transform_water(water_values)
    irr_labels_pct = percentile_binner.transform_irrigation(irr_values)
    
    # Print bin ranges
    print("Equal Width Binning:")
    equal_width_binner.print_bin_ranges()
    
    print("\nPercentile Binning:")
    percentile_binner.print_bin_ranges()
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Equal width distributions
    sns.histplot(data=pd.DataFrame({'Water Saving': water_values_np}), x='Water Saving', 
                bins=equal_width_binner.water_bins, ax=axes[0,0])
    axes[0,0].set_title('Water Saving Distribution (Equal Width)')
    axes[0,0].set_xlabel('Water Saving Value')
    
    sns.histplot(data=pd.DataFrame({'Irrigation': irr_values_np}), x='Irrigation',
                bins=equal_width_binner.irr_bins, ax=axes[0,1])
    axes[0,1].set_title('Irrigation Distribution (Equal Width)')
    axes[0,1].set_xlabel('Irrigation Value')
    
    # Percentile distributions
    sns.histplot(data=pd.DataFrame({'Water Saving': water_values_np}), x='Water Saving',
                bins=percentile_binner.water_bins, ax=axes[1,0])
    axes[1,0].set_title('Water Saving Distribution (Percentile)')
    axes[1,0].set_xlabel('Water Saving Value')
    
    sns.histplot(data=pd.DataFrame({'Irrigation': irr_values_np}), x='Irrigation',
                bins=percentile_binner.irr_bins, ax=axes[1,1])
    axes[1,1].set_title('Irrigation Distribution (Percentile)')
    axes[1,1].set_xlabel('Irrigation Value')
    
    plt.tight_layout()
    # Save plot with proper permissions
    import os
    output_path = '/home/ubuntu/attachments/binning_distributions.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved distribution plot to: {output_path}")
    plt.close()
    
    # Print class distributions
    print("\nClass Distribution Analysis:")
    print("\nEqual Width Binning:")
    print("Water Saving Classes:")
    for i in range(5):
        count = np.sum(water_labels_eq == i)
        percent = (count / len(water_labels_eq)) * 100
        print(f"Class {i}: {count} samples ({percent:.1f}%)")
    
    print("\nIrrigation Classes:")
    for i in range(5):
        count = np.sum(irr_labels_eq == i)
        percent = (count / len(irr_labels_eq)) * 100
        print(f"Class {i}: {count} samples ({percent:.1f}%)")
    
    print("\nPercentile Binning:")
    print("Water Saving Classes:")
    for i in range(5):
        count = np.sum(water_labels_pct == i)
        percent = (count / len(water_labels_pct)) * 100
        print(f"Class {i}: {count} samples ({percent:.1f}%)")
    
    print("\nIrrigation Classes:")
    for i in range(5):
        count = np.sum(irr_labels_pct == i)
        percent = (count / len(irr_labels_pct)) * 100
        print(f"Class {i}: {count} samples ({percent:.1f}%)")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_binning.py <path_to_excel_file>")
        sys.exit(1)
    excel_path = sys.argv[1]
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        sys.exit(1)
    analyze_binning_distribution(excel_path)
