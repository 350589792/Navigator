import numpy as np
from tailgating_detector import Person, ThreatCalculator, EntropyWeightCalculator

def validate_startup_phase():
    """Validate against Table 1 (startup phase) data"""
    print("Validating Startup Phase Data (Table 1)")
    print("-" * 50)
    
    # Create calculators
    threat_calc = ThreatCalculator()
    entropy_calc = EntropyWeightCalculator()
    
    # Table 1 data
    frames = [60, 65, 70, 75, 80, 85, 90]
    speeds = [0.06, 0.06, 0.11, 0.18, 0.20, 0.60, 2.53]
    distances = [202, 206, 214, 240, 258, 313, 325]
    expected_j = [0.06, 0.06, 0.11, 0.18, 0.20, 0.69, 2.53]
    
    for frame, speed, distance, expected in zip(frames, speeds, distances, expected_j):
        # Create person objects
        target = Person(x=0, y=0, frame_num=frame)
        follower = Person(x=distance, y=0, frame_num=frame)
        follower.speed = speed
        
        # Calculate threats
        speed_threat = threat_calc.calculate_speed_threat(speed)
        distance_threat = threat_calc.calculate_distance_threat(distance)
        
        # Calculate weights
        weights = entropy_calc.calculate_weights([(speed_threat, distance_threat, 0.0)])
        
        # Calculate J value
        j_value = weights[0] * speed_threat + weights[1] * distance_threat
        
        print(f"Frame {frame}:")
        print(f"  Speed: {speed:.2f}, Distance: {distance}")
        print(f"  Speed Weight: {weights[0]:.3f}, Distance Weight: {weights[1]:.3f}")
        print(f"  J Value: {j_value:.2f} (Expected: {expected:.2f})")
        print()

def validate_following_detection():
    """Validate against Table 2 (successful following detection) data"""
    print("\nValidating Following Detection Data (Table 2)")
    print("-" * 50)
    
    # Create calculators
    threat_calc = ThreatCalculator()
    entropy_calc = EntropyWeightCalculator()
    
    # Table 2 data
    frames = [95, 100, 105, 110, 115, 120, 125]
    speeds = [2.7, 0.7, 3.97, 2.21, 0.63, 2.21, 0.93]
    distances = [315, 350, 313, 275, 247, 208, 204]
    expected_j = [0.20, 0.71, 3.95, 2.15, 0.62, 2.15, 0.91]
    
    for frame, speed, distance, expected in zip(frames, speeds, distances, expected_j):
        # Create person objects
        target = Person(x=0, y=0, frame_num=frame)
        follower = Person(x=distance, y=0, frame_num=frame)
        follower.speed = speed
        
        # Calculate threats
        speed_threat = threat_calc.calculate_speed_threat(speed)
        distance_threat = threat_calc.calculate_distance_threat(distance)
        
        # Calculate weights
        weights = entropy_calc.calculate_weights([(speed_threat, distance_threat, 0.0)])
        
        # Calculate J value
        j_value = weights[0] * speed_threat + weights[1] * distance_threat
        
        print(f"Frame {frame}:")
        print(f"  Speed: {speed:.2f}, Distance: {distance}")
        print(f"  Speed Weight: {weights[0]:.3f}, Distance Weight: {weights[1]:.3f}")
        print(f"  J Value: {j_value:.2f} (Expected: {expected:.2f})")
        print()

if __name__ == '__main__':
    validate_startup_phase()
    validate_following_detection()
