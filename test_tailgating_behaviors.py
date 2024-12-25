import unittest
from tailgating_detector import Person, ThreatCalculator, EntropyWeightCalculator, TailgatingDetector, TailgatingType

class TestTailgatingBehaviors(unittest.TestCase):
    def setUp(self):
        self.threat_calc = ThreatCalculator()
        self.entropy_calc = EntropyWeightCalculator()
        self.detector = TailgatingDetector(
            self.threat_calc,
            self.entropy_calc,
            gate_x=640.0,  # Center of 1280px frame
            detection_zone_width=600.0  # Increased for 1280px frame width
        )
        
    def test_gate_entrance_detection(self):
        """Test detection zone and gate passing logic"""
        # Create target approaching gate
        target = Person(x=500, y=360, frame_num=1)
        follower = Person(x=400, y=360, frame_num=1)
        
        # Initial detection - both in zone
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertTrue(340 <= target.x <= 940)  # In detection zone (600px width)
        self.assertTrue(340 <= follower.x <= 940)  # In detection zone (600px width)
        self.assertIsNotNone(target.gate_x)
        self.assertIsNotNone(follower.gate_x)
        
        # Move target to gate
        target.x = 635  # Near gate
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertTrue(target.is_passing_gate)
        self.assertFalse(target.passed_gate)
        
        # Move target past gate
        target.x = 645  # Past gate
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertTrue(target.passed_gate)
        
    def test_direct_rear_tailgating(self):
        """Test direct rear tailgating detection (正后尾随行为)"""
        # Setup direct rear following scenario
        target = Person(x=600, y=360, frame_num=1)
        follower = Person(x=500, y=360, frame_num=1)  # Directly behind
        follower.speed = 2.5  # High speed to trigger threat
        
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertEqual(type_, TailgatingType.DIRECT)
        self.assertTrue(is_tailgating)
        self.assertGreater(score, self.detector.threat_threshold)
        
    def test_lateral_tailgating(self):
        """Test lateral tailgating detection (侧向尾随行为)"""
        # Setup lateral following scenario
        target = Person(x=600, y=360, frame_num=1)
        follower = Person(x=500, y=460, frame_num=1)  # Diagonal approach
        follower.speed = 2.5
        
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertEqual(type_, TailgatingType.LATERAL)
        self.assertTrue(is_tailgating)
        self.assertGreater(score, self.detector.threat_threshold)
        
    def test_horizontal_tailgating(self):
        """Test horizontal tailgating detection (横线尾随行为)"""
        # Setup horizontal following scenario
        target = Person(x=600, y=360, frame_num=1)
        follower = Person(x=500, y=370, frame_num=1)  # Nearly parallel
        follower.speed = 2.5
        
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertEqual(type_, TailgatingType.HORIZONTAL)
        self.assertTrue(is_tailgating)
        self.assertGreater(score, self.detector.threat_threshold)
        
    def test_outside_detection_zone(self):
        """Test behavior when people are outside detection zone"""
        target = Person(x=200, y=360, frame_num=1)  # Outside zone
        follower = Person(x=100, y=360, frame_num=1)
        follower.speed = 2.5
        
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertFalse(is_tailgating)
        self.assertEqual(score, 0.0)
        
    def test_zero_speed_handling(self):
        """Test handling of zero speed values"""
        target = Person(x=600, y=360, frame_num=1)
        follower = Person(x=500, y=360, frame_num=1)
        follower.speed = 0.0  # Zero speed
        
        # Should not crash and return valid results
        is_tailgating, type_, score = self.detector.detect_tailgating(target, follower)
        self.assertIsNotNone(score)
        self.assertIsNotNone(type_)
        self.assertIsNotNone(is_tailgating)


if __name__ == '__main__':
    unittest.main()
