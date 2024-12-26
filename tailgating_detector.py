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
        self.is_authorized = False  # Track authorization status
        self.entry_zone = None  # Track which zone they entered through
        print(f"Created person at ({x:.1f}, {y:.1f}) frame={frame_num} auth={self.is_authorized}")
        
    def calculate_speed(self, prev_person: 'Person', fps: float):
        """Calculate speed based on previous position and normalize to pixels per second"""
        if prev_person and fps > 0:
            dx = self.x - prev_person.x
            dy = self.y - prev_person.y
            dt = max(1/fps, (self.frame_num - prev_person.frame_num) / fps)  # Ensure minimum dt of one frame
            
            # Calculate speed in pixels per frame first
            distance = np.sqrt(dx*dx + dy*dy)
            speed_px_per_frame = distance / max(1, self.frame_num - prev_person.frame_num)
            
            # Convert to pixels per second
            speed_px_per_sec = speed_px_per_frame * fps
            
            # Clamp speed to reasonable values (max 200 pixels per second)
            MAX_SPEED = 200  # roughly 1/6 of screen width per second
            self.speed = min(speed_px_per_sec, MAX_SPEED)
            print(f"Speed calculation: dx={dx:.1f}, dy={dy:.1f}, dt={dt:.3f}, speed={self.speed:.2f} px/s")

class ThreatCalculator:
    def __init__(self):
        self.distance_threshold = 150  # pixels (increased for better detection)
        self.speed_threshold = 50      # pixels per second (lowered for better sensitivity)
        self.angle_thresholds = {
            TailgatingType.DIRECT: 60,      # degrees (more lenient: 180° ±60°)
            TailgatingType.LATERAL: 60,      # degrees (more lenient for side approaches)
            TailgatingType.HORIZONTAL: 90    # degrees (more lenient for horizontal approaches)
        }
        print(f"ThreatCalculator initialized with thresholds: distance={self.distance_threshold}px, "
              f"speed={self.speed_threshold}px/s")
    
    def calculate_distance_threat(self, distance: float) -> float:
        """Calculate threat level based on distance with smooth transition:
        - Maximum threat (1.0) when distance is 0
        - Linear decrease until distance_threshold
        - Minimum threat (0.0) beyond threshold
        """
        if distance <= 0:
            return 1.0
        # Smooth linear transition from 1.0 to 0.0
        threat = max(0.0, 1.0 - (distance / self.distance_threshold))
        print(f"Distance threat calculation: distance={distance:.1f}px, threshold={self.distance_threshold}px, threat={threat:.3f}")
        return threat
    
    def calculate_speed_threat(self, speed: float) -> float:
        """Calculate threat level based on speed using S-curve (logistic function):
        threat = 1 / (1 + exp(-k * (speed - c)))
        where:
        - c is the inflection point (speed_threshold/2)
        - k controls the steepness of the curve (increased for better sensitivity)
        """
        import math

        if speed <= 0:
            return 0.0
            
        # Use speed_threshold/2 as inflection point
        c = self.speed_threshold / 2.0
        # Increased k for better sensitivity
        k = 0.2
        
        # Calculate threat using logistic function
        threat = 1.0 / (1.0 + math.exp(-k * (speed - c)))
        threat = min(1.0, threat)
        print(f"Speed threat calculation: speed={speed:.1f}px/s, threshold={self.speed_threshold}px/s, threat={threat:.3f}")
        return threat
    
    def calculate_angle_threat(self, angle: float) -> float:
        """Calculate threat level based on angle with quadratic emphasis:
        - Maximum threat (1.0) at 180° (directly behind)
        - Minimum threat (0.0) at 0° or 360° (directly ahead)
        - Quadratic scaling to emphasize angles closer to 180°
        """
        # Normalize angle to [0, 180] range and square for emphasis
        normalized = abs(180 - angle) / 180
        return normalized * normalized

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
            print(f"Angle consistency check: diff={angle_diff:.1f}°, tolerance={self.angle_tolerance}°, consistent={self.consistent_angle}")
        self.last_angle = angle
        self.frames_detected += 1
        self.target = target
        self.follower = follower
        # Reduced frame requirement from 3 to 2 for faster detection
        is_valid = self.consistent_angle and self.frames_detected >= 2
        print(f"Pair validation: angle_consistent={self.consistent_angle}, frames={self.frames_detected}, valid={is_valid}")
        return is_valid

class TailgatingDetector:
    def __init__(self, threat_calculator: ThreatCalculator, entropy_calculator: EntropyWeightCalculator):
        self.threat_calculator = threat_calculator
        self.entropy_calculator = entropy_calculator
        from ahp_calculator import compute_ahp_weights  # Import AHP calculator
        self.compute_ahp_weights = compute_ahp_weights
        self.contour_area_threshold = 300   # Further reduced threshold to detect smaller human shapes (~17x17px)
        self.motion_detection_threshold = 20  # Further reduced to improve detection of smaller movements
        self.min_speed_threshold = 0.5  # Minimum speed (pixels/second, reduced for better detection)
        self.tracked_pairs = {}  # Dictionary to track potential tailgating pairs
        # Weight balance between entropy and AHP (0.5 means equal weight)
        self.entropy_weight = 0.5
        
    def detect_tailgating(self, target: Person, follower: Person) -> Tuple[bool, Optional[TailgatingType], float]:
        """Detect if follower is tailgating target. Skips check if follower is authorized."""
        # More detailed authorization check logging
        print(f"Checking pair: Target({target.x:.1f}, {target.y:.1f}) auth={target.is_authorized} -> "
              f"Follower({follower.x:.1f}, {follower.y:.1f}) auth={follower.is_authorized}")
        
        # Only skip if both target and follower are authorized
        if target.is_authorized and follower.is_authorized:
            print(f"Skipping fully authorized pair")
            return False, None, 0.0
        print(f"Checking tailgating: Target({target.x:.1f}, {target.y:.1f}) -> Follower({follower.x:.1f}, {follower.y:.1f})")
        # Calculate distance
        dx = follower.x - target.x
        dy = follower.y - target.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate angle with detailed logging
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
        print(f"Angle calculation: dx={dx:.1f}, dy={dy:.1f}, angle={angle:.1f}°")
            
        # Check minimum speed threshold with detailed logging
        print(f"Speed check: follower speed={follower.speed:.1f}px/s, min threshold={self.min_speed_threshold}px/s")
        if follower.speed < self.min_speed_threshold:
            print(f"Rejected: Speed below minimum threshold")
            return False, None, 0.0
            
        # Create or update pair tracking with logging
        pair_key = (target.x, target.y, follower.x, follower.y)
        if pair_key not in self.tracked_pairs:
            print(f"New pair detected: creating tracking")
            self.tracked_pairs[pair_key] = TailgatingPair(target, follower)
        else:
            print(f"Updating existing pair: frames={self.tracked_pairs[pair_key].frames_detected}")
        
        # Update tracking and check if pair is consistent
        is_consistent = self.tracked_pairs[pair_key].update(target, follower, angle)
        print(f"Pair consistency check: consistent={is_consistent}, frames={self.tracked_pairs[pair_key].frames_detected}")
        if not is_consistent:
            print(f"Rejected: Pair not consistent enough")
            return False, None, 0.0
            
        # Calculate threats with detailed logging
        distance_threat = self.threat_calculator.calculate_distance_threat(distance)
        speed_threat = self.threat_calculator.calculate_speed_threat(follower.speed)
        angle_threat = self.threat_calculator.calculate_angle_threat(angle)
        
        print(f"THREAT_DETAILS: distance={distance:.1f}px threat={distance_threat:.3f}, "
              f"speed={follower.speed:.1f}px/s threat={speed_threat:.3f}, "
              f"angle={angle:.1f}° threat={angle_threat:.3f}")
        
        # Detailed debug logging for threat calculations
        print(f"THREAT_CALC: Target({target.x:.1f}, {target.y:.1f}) -> Follower({follower.x:.1f}, {follower.y:.1f})")
        print(f"THREAT_CALC: Distance={distance:.2f}px (threat={distance_threat:.2f})")
        print(f"THREAT_CALC: Speed={follower.speed:.2f}px/s (threat={speed_threat:.2f})")
        print(f"THREAT_CALC: Angle={angle:.2f}° (threat={angle_threat:.2f})")
        
        # Calculate entropy weights
        entropy_weights = self.entropy_calculator.calculate_weights(
            [distance_threat], [speed_threat], [angle_threat]
        )
        
        # Calculate AHP weights (with distance having highest importance)
        ahp_weights = self.compute_ahp_weights(
            distance_importance=5.0,  # Distance most important
            speed_importance=3.0,     # Speed moderately important
            angle_importance=1.0      # Angle least important
        )
        
        # Combine weights using weighted average
        weights = tuple(
            self.entropy_weight * ew + (1 - self.entropy_weight) * aw
            for ew, aw in zip(entropy_weights, ahp_weights)
        )
        
        # Calculate final threat score using combined weights
        threat_score = (
            weights[0] * distance_threat +
            weights[1] * speed_threat +
            weights[2] * angle_threat
        )
        print(f"THREAT_FINAL: score={threat_score:.3f} (weights: distance={weights[0]:.2f}, "
              f"speed={weights[1]:.2f}, angle={weights[2]:.2f})")
        
        # Determine tailgating type based on angle
        tailgating_type = None
        if threat_score > 0.2:  # Further lowered threshold to allow more varied tailgating detection
            print(f"THREAT_DETECTED: Tailgating detected with score {threat_score:.3f}")
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
                
        return threat_score > 0.2, tailgating_type, threat_score
