import numpy as np
from enum import Enum
from typing import Tuple, Optional

class TailgatingType(Enum):
    DIRECT = "direct"         # Following directly behind
    LATERAL = "lateral"       # Following from the side
    HORIZONTAL = "horizontal" # Following across

class Person:
    def __init__(self, x: float, y: float, frame_num: int):
        self.x = x
        self.y = y
        self.frame_num = frame_num
        self.speed = 0.0
        
    def calculate_speed(self, prev_person: 'Person', fps: float):
        """Calculate speed based on previous position"""
        if prev_person and fps > 0:
            dx = self.x - prev_person.x
            dy = self.y - prev_person.y
            dt = (self.frame_num - prev_person.frame_num) / fps
            if dt > 0:
                self.speed = np.sqrt(dx*dx + dy*dy) / dt

class ThreatCalculator:
    def __init__(self):
        self.distance_threshold = 100  # pixels
        self.speed_threshold = 50      # pixels per frame
        self.angle_thresholds = {
            TailgatingType.DIRECT: 140,     # degrees
            TailgatingType.HORIZONTAL: 50    # degrees
        }
    
    def calculate_distance_threat(self, distance: float) -> float:
        """Calculate threat level based on distance"""
        if distance <= 0:
            return 1.0
        return min(1.0, self.distance_threshold / distance)
    
    def calculate_speed_threat(self, speed: float) -> float:
        """Calculate threat level based on speed"""
        if speed <= 0:
            return 0.0
        return min(1.0, speed / self.speed_threshold)
    
    def calculate_angle_threat(self, angle: float) -> float:
        """Calculate threat level based on angle"""
        return min(1.0, abs(180 - angle) / 180)

class EntropyWeightCalculator:
    def calculate_weights(self, distance_threats: list, speed_threats: list, angle_threats: list) -> Tuple[float, float, float]:
        """Calculate weights using entropy method"""
        # Combine all threats
        all_threats = [
            distance_threats,
            speed_threats,
            angle_threats
        ]
        
        # Calculate entropy for each factor
        entropies = []
        for threats in all_threats:
            if not threats:
                entropies.append(0)
                continue
            
            # Normalize threats
            total = sum(threats)
            if total == 0:
                entropies.append(0)
                continue
                
            proportions = [t/total for t in threats]
            
            # Calculate entropy
            entropy = 0
            for p in proportions:
                if p > 0:
                    entropy -= p * np.log(p)
            entropies.append(entropy)
        
        # Calculate weights
        total_entropy = sum(entropies)
        if total_entropy == 0:
            return (1/3, 1/3, 1/3)
        
        weights = [1 - (e/total_entropy) for e in entropies]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return (1/3, 1/3, 1/3)
        
        return tuple(w/total_weight for w in weights)

class TailgatingDetector:
    def __init__(self, threat_calculator: ThreatCalculator, entropy_calculator: EntropyWeightCalculator):
        self.threat_calculator = threat_calculator
        self.entropy_calculator = entropy_calculator
        self.contour_area_threshold = 500  # Adjusted for 720p resolution
        self.motion_detection_threshold = 35  # Good for frame difference detection
        
    def detect_tailgating(self, target: Person, follower: Person) -> Tuple[bool, Optional[TailgatingType], float]:
        """Detect if follower is tailgating target"""
        # Calculate distance
        dx = follower.x - target.x
        dy = follower.y - target.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate angle
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Calculate threats
        distance_threat = self.threat_calculator.calculate_distance_threat(distance)
        speed_threat = self.threat_calculator.calculate_speed_threat(follower.speed)
        angle_threat = self.threat_calculator.calculate_angle_threat(angle)
        
        # Calculate weights
        weights = self.entropy_calculator.calculate_weights(
            [distance_threat], [speed_threat], [angle_threat]
        )
        
        # Calculate final threat score
        threat_score = (
            weights[0] * distance_threat +
            weights[1] * speed_threat +
            weights[2] * angle_threat
        )
        
        # Determine tailgating type based on angle
        tailgating_type = None
        if threat_score > 0.5:  # Threshold for considering it tailgating
            if abs(180 - angle) <= self.threat_calculator.angle_thresholds[TailgatingType.DIRECT]:
                tailgating_type = TailgatingType.DIRECT
            elif abs(90 - angle) <= self.threat_calculator.angle_thresholds[TailgatingType.HORIZONTAL]:
                tailgating_type = TailgatingType.HORIZONTAL
            else:
                tailgating_type = TailgatingType.LATERAL
                
        return threat_score > 0.5, tailgating_type, threat_score
