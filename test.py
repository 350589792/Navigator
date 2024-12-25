import unittest
import numpy as np
from tailgating_detector import (
    Person, ThreatCalculator, EntropyWeightCalculator,
    TailgatingDetector, TailgatingType
)

class TestTailgatingDetection(unittest.TestCase):
    def setUp(self):
        self.threat_calc = ThreatCalculator()
        self.entropy_calc = EntropyWeightCalculator()
        self.detector = TailgatingDetector(self.threat_calc, self.entropy_calc)

    def test_direct_rear_tailgating(self):
        """Test detection of direct rear tailgating (正后尾随行为)"""
        # Target at (100, 100) moving right
        target = Person(x=100, y=100, frame_num=1)
        target_next = Person(x=120, y=100, frame_num=2)
        target_next.calculate_speed(target)

        # Follower directly behind
        follower = Person(x=80, y=100, frame_num=1)
        follower_next = Person(x=100, y=100, frame_num=2)
        follower_next.calculate_speed(follower)

        is_tailgating, type_, score = self.detector.detect_tailgating(
            target_next, follower_next
        )
        self.assertTrue(is_tailgating)
        self.assertEqual(type_, TailgatingType.DIRECT)

    def test_lateral_tailgating(self):
        """Test detection of lateral tailgating (侧向尾随行为)"""
        # Target at (100, 100) moving right
        target = Person(x=100, y=100, frame_num=1)
        target_next = Person(x=120, y=100, frame_num=2)
        target_next.calculate_speed(target)

        # Follower approaching from side
        follower = Person(x=90, y=120, frame_num=1)
        follower_next = Person(x=110, y=110, frame_num=2)
        follower_next.calculate_speed(follower)

        is_tailgating, type_, score = self.detector.detect_tailgating(
            target_next, follower_next
        )
        self.assertTrue(is_tailgating)
        self.assertEqual(type_, TailgatingType.LATERAL)

    def test_horizontal_tailgating(self):
        """Test detection of horizontal tailgating (横线尾随行为)"""
        # Target at (100, 100) moving right
        target = Person(x=100, y=100, frame_num=1)
        target_next = Person(x=120, y=100, frame_num=2)
        target_next.calculate_speed(target)

        # Follower moving parallel
        follower = Person(x=100, y=120, frame_num=1)
        follower_next = Person(x=120, y=120, frame_num=2)
        follower_next.calculate_speed(follower)

        is_tailgating, type_, score = self.detector.detect_tailgating(
            target_next, follower_next
        )
        self.assertTrue(is_tailgating)
        self.assertEqual(type_, TailgatingType.HORIZONTAL)

    def test_zero_speed_handling(self):
        """Test handling of zero speed to prevent entropy calculation crash"""
        target = Person(x=100, y=100, frame_num=1)
        follower = Person(x=80, y=100, frame_num=1)
        
        # Both people not moving (zero speed)
        is_tailgating, type_, score = self.detector.detect_tailgating(
            target, follower
        )
        self.assertGreater(score, 0)  # Should not crash with zero speed

if __name__ == '__main__':
    unittest.main()
