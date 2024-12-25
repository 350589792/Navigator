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
        """Calculate speed based on previous position and normalize to pixels per second"""
        if prev_person and fps > 0:
            dx = self.x - prev_person.x
            dy = self.y - prev_person.y
            dt = max(1/fps, (self.frame_num - prev_person.frame_num) / fps)  # Ensure minimum dt of one frame
            
            # Calculate speed in pixels per second
            speed_px_per_sec = np.sqrt(dx*dx + dy*dy) / dt
            
            # Clamp speed to reasonable values (max 500 pixels per second)
            MAX_SPEED = 500  # roughly 1/3 of screen width per second
            self.speed = min(speed_px_per_sec, MAX_SPEED)

class ThreatCalculator:
    def __init__(self):
        self.distance_threshold = 100  # pixels
        self.speed_threshold = 200     # pixels per second
        self.angle_thresholds = {
            TailgatingType.DIRECT: 40,      # degrees (stricter for direct: 180° ±40°)
            TailgatingType.LATERAL: 45,      # degrees (for angles near 0° or 360°)
            TailgatingType.HORIZONTAL: 70    # degrees (more lenient for 90° or 270° ±20°)
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

class TailgatingPair:
    def __init__(self, target: Person, follower: Person):
        self.target = target
        self.follower = follower
        self.frames_detected = 1
        self.consistent_angle = True
        self.last_angle = None
        self.angle_tolerance = 20  # degrees

    def update(self, target: Person, follower: Person, angle: float) -> bool:
        if self.last_angle is not None:
            angle_diff = abs(angle - self.last_angle)
            self.consistent_angle = angle_diff <= self.angle_tolerance
        self.last_angle = angle
        self.frames_detected += 1
        self.target = target
        self.follower = follower
        return self.consistent_angle and self.frames_detected >= 3

class TailgatingDetector:
    def __init__(self, threat_calculator: ThreatCalculator, entropy_calculator: EntropyWeightCalculator):
        self.threat_calculator = threat_calculator
        self.entropy_calculator = entropy_calculator
        self.contour_area_threshold = 600   # Reduced threshold to detect smaller human shapes (~24x24px)
        self.motion_detection_threshold = 30  # Reduced to improve detection of smaller movements
        self.min_speed_threshold = 1.0  # Minimum speed (pixels/frame) for tailgating detection
        self.tracked_pairs = {}  # Dictionary to track potential tailgating pairs
        
    def detect_tailgating(self, target: Person, follower: Person) -> Tuple[bool, Optional[TailgatingType], float]:
        """Detect if follower is tailgating target"""
        print(f"Checking tailgating: Target({target.x:.1f}, {target.y:.1f}) -> Follower({follower.x:.1f}, {follower.y:.1f})")
        # Calculate distance
        dx = follower.x - target.x
        dy = follower.y - target.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate angle
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Check minimum speed threshold
        if follower.speed < self.min_speed_threshold:
            return False, None, 0.0
            
        # Create or update pair tracking
        pair_key = (target.x, target.y, follower.x, follower.y)
        if pair_key not in self.tracked_pairs:
            self.tracked_pairs[pair_key] = TailgatingPair(target, follower)
        
        # Update tracking and check if pair is consistent
        if not self.tracked_pairs[pair_key].update(target, follower, angle):
            return False, None, 0.0
            
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
        if threat_score > 0.5:  # Lowered threshold to allow more varied tailgating detection
            # Check for horizontal tailgating first (around 90° or 270°)
            if min(abs(90 - angle), abs(270 - angle)) <= self.threat_calculator.angle_thresholds[TailgatingType.HORIZONTAL]:
                tailgating_type = TailgatingType.HORIZONTAL
            # Then check for lateral tailgating (around 0° or 360°)
            elif min(angle, abs(360 - angle)) <= self.threat_calculator.angle_thresholds[TailgatingType.LATERAL]:
                tailgating_type = TailgatingType.LATERAL
            # Finally check for direct tailgating (around 180°)
            elif abs(180 - angle) <= self.threat_calculator.angle_thresholds[TailgatingType.DIRECT]:
                tailgating_type = TailgatingType.DIRECT
            # No tailgating type if none of the patterns match
            else:
                tailgating_type = None
                
        return threat_score > 0.5, tailgating_type, threat_score
