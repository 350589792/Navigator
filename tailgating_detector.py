import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List

class TailgatingType(Enum):
    DIRECT = "正后尾随行为"  # Direct rear following
    LATERAL = "侧向尾随行为"  # Side following
    HORIZONTAL = "横线尾随行为"  # Horizontal following

@dataclass
class Person:
    x: float  # pixel x coordinate
    y: float  # pixel y coordinate
    frame_num: int
    speed: float = 0.0
    
    def calculate_speed(self, prev_position: 'Person', fps: float = 25.0) -> float:
        """Calculate speed in pixels per second"""
        if not prev_position:
            return 0.0
        dx = self.x - prev_position.x
        dy = self.y - prev_position.y
        dt = (self.frame_num - prev_position.frame_num) / fps
        if dt == 0:
            return 0.0
        self.speed = np.sqrt(dx*dx + dy*dy) / dt
        return self.speed

    def calculate_angle(self, other: 'Person') -> float:
        """Calculate angle between two people in degrees"""
        dx = other.x - self.x
        dy = other.y - self.y
        return np.degrees(np.arctan2(dy, dx))

    def calculate_distance(self, other: 'Person') -> float:
        """Calculate Euclidean distance between two people in pixels"""
        dx = other.x - self.x
        dy = other.y - self.y
        return np.sqrt(dx*dx + dy*dy)

class ThreatCalculator:
    def __init__(self, distance_threshold: float = 200.0, 
                 speed_threshold: float = 2.0,
                 angle_threshold: float = 170.0):
        self.distance_threshold = distance_threshold
        self.speed_threshold = speed_threshold
        self.angle_threshold = angle_threshold

    def calculate_distance_threat(self, distance: float) -> float:
        """
        Calculate distance threat level (0-1)
        Threat increases as distance decreases
        """
        if distance >= self.distance_threshold:
            return 0.0
        # Exponential decay function
        return np.exp(-distance / self.distance_threshold)

    def calculate_speed_threat(self, speed: float) -> float:
        """
        Calculate speed threat level (0-1)
        S-curve function for speed threat
        """
        if speed <= 0:
            return 0.001  # Small non-zero value to prevent entropy calculation issues
        # Sigmoid function
        c = self.speed_threshold / 2  # Inflection point
        return 1 / (1 + np.exp(-2 * (speed - c)))

    def calculate_angle_threat(self, angle: float) -> float:
        """
        Calculate angle threat level (0-1)
        Threat increases as angle approaches 180 degrees
        """
        normalized_angle = abs(angle)
        if normalized_angle > 180:
            normalized_angle = 360 - normalized_angle
        # Using arctan to create smooth transition
        return np.arctan(max(0, normalized_angle - 90) / 45) / (np.pi/2)

class EntropyWeightCalculator:
    def calculate_weights(self, threat_values: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Calculate weights using entropy weight method
        Input: List of (speed_threat, distance_threat, angle_threat) tuples
        Output: (speed_weight, distance_weight, angle_weight)
        """
        if not threat_values:
            return (0.33, 0.33, 0.34)  # Default equal weights
            
        # Convert to numpy array for easier calculation
        data = np.array(threat_values)
        
        # Normalize the data
        row_sums = data.sum(axis=1)
        normalized = data / row_sums[:, np.newaxis]
        
        # Calculate entropy for each factor
        n = len(threat_values)
        entropies = np.zeros(3)
        for j in range(3):
            valid_vals = normalized[:, j][normalized[:, j] > 0]
            if len(valid_vals) > 0:
                entropies[j] = -np.sum(valid_vals * np.log(valid_vals)) / np.log(n)
        
        # Calculate weights
        diff = 1 - entropies
        if np.sum(diff) == 0:
            return (0.33, 0.33, 0.34)
        weights = diff / np.sum(diff)
        
        return tuple(weights)

class TailgatingDetector:
    def __init__(self, threat_calculator: ThreatCalculator,
                 entropy_calculator: EntropyWeightCalculator,
                 threat_threshold: float = 2.0):
        self.threat_calculator = threat_calculator
        self.entropy_calculator = entropy_calculator
        self.threat_threshold = threat_threshold
        self.contour_area_threshold = 1000  # Increased contour area threshold
        self.motion_detection_threshold = 35  # Adjusted motion detection threshold
        self.history: List[Tuple[Person, Person]] = []  # Store (target, follower) pairs


    def detect_tailgating(self, target: Person, follower: Person) -> Tuple[bool, TailgatingType, float]:
        """
        Detect if tailgating is occurring and determine the type
        Returns: (is_tailgating, tailgating_type, threat_score)
        """
        # Calculate basic metrics
        distance = target.calculate_distance(follower)
        angle = target.calculate_angle(follower)
        
        # Calculate threat levels
        speed_threat = self.threat_calculator.calculate_speed_threat(follower.speed)
        distance_threat = self.threat_calculator.calculate_distance_threat(distance)
        angle_threat = self.threat_calculator.calculate_angle_threat(angle)
        
        # Store threats for entropy calculation
        self.history.append((target, follower))
        threat_values = [(speed_threat, distance_threat, angle_threat)]
        
        # Calculate weights using entropy method
        weights = self.entropy_calculator.calculate_weights(threat_values)
        
        # Calculate final threat score
        threat_score = (
            weights[0] * speed_threat +
            weights[1] * distance_threat +
            weights[2] * angle_threat
        )
        
        # Determine tailgating type based on angle
        if abs(angle) > 140:  # Adjusted threshold for direct rear
            tailgating_type = TailgatingType.DIRECT
        elif abs(angle) < 50:  # Adjusted threshold for horizontal movement
            tailgating_type = TailgatingType.HORIZONTAL
        else:
            tailgating_type = TailgatingType.LATERAL
            
        is_tailgating = threat_score >= self.threat_threshold
        
        return is_tailgating, tailgating_type, threat_score
