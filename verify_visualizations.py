import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from pathlib import Path

def verify_visualization_files():
    """Verify the generated visualization files."""
    output_dir = Path("visualization_output")
    expected_files = [
        "waveforms.png",
        "spectrograms.png",
        "error_distribution.png",
        "ablation_comparison.png"
    ]
    
    print("=== Visualization Verification Report ===\n")
    
    # Check if output directory exists
    if not output_dir.exists():
        print("ERROR: visualization_output directory not found!")
        return False
        
    # Check for expected files
    print("Checking for required files:")
    all_files_present = True
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists():
            img = mpimg.imread(str(file_path))
            size_kb = file_path.stat().st_size / 1024
            print(f"\n{filename}:")
            print(f"  - Status: Found")
            print(f"  - Dimensions: {img.shape}")
            print(f"  - Size: {size_kb:.2f} KB")
            
            # Additional quality checks
            if size_kb < 10:
                print("  - WARNING: File size seems too small!")
            if img.shape[0] < 300 or img.shape[1] < 300:
                print("  - WARNING: Image resolution might be too low!")
        else:
            print(f"\n{filename}:")
            print(f"  - Status: MISSING")
            all_files_present = False
    
    # Summary
    print("\n=== Summary ===")
    if all_files_present:
        print("✓ All required visualization files are present")
    else:
        print("✗ Some visualization files are missing!")
    
    return all_files_present

if __name__ == "__main__":
    success = verify_visualization_files()
    exit(0 if success else 1)
