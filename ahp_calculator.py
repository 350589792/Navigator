"""
AHP (Analytic Hierarchy Process) Calculator for Tailgating Detection

This module implements the AHP method for computing weights of different
threat factors in tailgating detection. It uses the Saaty scale for
pairwise comparisons and includes consistency checking.

Saaty Scale:
1 - Equal importance (同等重要)
3 - Slightly important (稍微重要)
5 - Clearly important (明显重要)
7 - Extremely important (极其重要)
9 - Strongly important (强烈重要)
2,4,6,8 - Intermediate values
"""

import numpy as np
from typing import Tuple, List

class AHPCalculator:
    # Random Index values for consistency checking (from Saaty)
    RI_VALUES = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }

    def __init__(self):
        # Initialize with default comparison matrix for distance, speed, angle
        # Based on requirements: distance is most important, then speed, then angle
        self.comparison_matrix = np.array([
            [1.0, 3.0, 5.0],  # distance compared to others
            [1/3, 1.0, 3.0],  # speed compared to others
            [1/5, 1/3, 1.0]   # angle compared to others
        ])

    def compute_weights(self) -> Tuple[float, float, float]:
        """
        Compute the AHP weights using the comparison matrix.
        Returns weights for (distance, speed, angle).
        """
        # 1. Calculate geometric mean of each row
        n = len(self.comparison_matrix)
        row_products = np.prod(self.comparison_matrix, axis=1)
        geometric_means = np.power(row_products, 1/n)
        
        # 2. Normalize the geometric means to get weights
        weights = geometric_means / np.sum(geometric_means)
        
        # 3. Check consistency
        if not self._check_consistency():
            print("Warning: Comparison matrix is not consistent!")
        
        return tuple(weights)  # returns (distance_w, speed_w, angle_w)

    def _check_consistency(self) -> bool:
        """
        Check the consistency of the comparison matrix.
        Returns True if consistent (CR < 0.1), False otherwise.
        """
        n = len(self.comparison_matrix)
        
        # Calculate principal eigenvalue
        eigenvalues = np.linalg.eigvals(self.comparison_matrix)
        lambda_max = max(eigenvalues.real)
        
        # Calculate Consistency Index (CI)
        ci = (lambda_max - n) / (n - 1)
        
        # Get Random Index (RI)
        ri = self.RI_VALUES[n]
        
        # Calculate Consistency Ratio (CR)
        # For n=3, CR should be < 0.05
        # For n>3, CR should be < 0.1
        cr = ci / ri if ri != 0 else 0
        
        return cr < 0.1

    def update_comparison(self, matrix: List[List[float]]) -> None:
        """
        Update the comparison matrix with new values.
        
        Args:
            matrix: 3x3 matrix of pairwise comparisons using Saaty scale
        """
        if not isinstance(matrix, (list, np.ndarray)) or len(matrix) != 3:
            raise ValueError("Matrix must be 3x3")
        self.comparison_matrix = np.array(matrix)

def compute_ahp_weights(distance_importance: float = 5.0,
                       speed_importance: float = 3.0,
                       angle_importance: float = 1.0) -> Tuple[float, float, float]:
    """
    Compute AHP weights for tailgating threat factors.
    
    Args:
        distance_importance: Importance of distance (1-9 Saaty scale)
        speed_importance: Importance of speed (1-9 Saaty scale)
        angle_importance: Importance of angle (1-9 Saaty scale)
    
    Returns:
        Tuple[float, float, float]: Normalized weights for (distance, speed, angle)
    """
    calculator = AHPCalculator()
    
    # Create comparison matrix based on provided importance values
    matrix = [
        [1.0, distance_importance/speed_importance, distance_importance/angle_importance],
        [speed_importance/distance_importance, 1.0, speed_importance/angle_importance],
        [angle_importance/distance_importance, angle_importance/speed_importance, 1.0]
    ]
    
    calculator.update_comparison(matrix)
    return calculator.compute_weights()
