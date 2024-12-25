#!/bin/bash
set -e

# Create output directory if it doesn't exist
mkdir -p visualization_output

# Make sure the script is executable
chmod +x test_visualization_updated.py

# Run the visualization test script
python test_visualization_updated.py

# Display success message
echo "Visualization plots have been generated in the visualization_output directory"
