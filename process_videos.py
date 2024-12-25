import cv2
import numpy as np
from tailgating_detector import TailgatingDetector, TailgatingType, Person, ThreatCalculator, EntropyWeightCalculator
import os
from typing import Tuple, List

class VideoProcessor:
    def __init__(self, input_dir: str = "/home/ubuntu/attachments/", 
                 output_dir: str = "/home/ubuntu/attachments/output/"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threat_calculator = ThreatCalculator()
        self.entropy_calculator = EntropyWeightCalculator()
        self.detector = TailgatingDetector(self.threat_calculator, self.entropy_calculator)
        self.prev_gray = None  # Store previous grayscale frame for motion detection
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Color mapping for different tailgating types
        self.color_map = {
            TailgatingType.DIRECT: (0, 0, 255),     # Red
            TailgatingType.LATERAL: (0, 255, 0),     # Green
            TailgatingType.HORIZONTAL: (255, 0, 0)   # Blue
        }

    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[List[Person], np.ndarray]:
        """Process a single frame and detect people"""
        # Create a copy of the frame for processing
        processed_frame = frame.copy()
        
        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Handle first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return [], processed_frame  # No motion detection on first frame
        
        # Apply background subtraction or motion detection
        motion_mask = cv2.absdiff(
            cv2.GaussianBlur(gray, (11, 11), 0),
            cv2.GaussianBlur(self.prev_gray, (11, 11), 0)
        )
        _, motion_mask = cv2.threshold(motion_mask, self.detector.motion_detection_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and create Person objects
        people = []
        for contour in contours:
            if cv2.contourArea(contour) > self.detector.contour_area_threshold:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    people.append(Person(cx, cy, frame_num))
                    # Draw contour and center point for visualization
                    cv2.drawContours(processed_frame, [contour], -1, (0, 255, 255), 2)  # Yellow contour
                    cv2.circle(processed_frame, (cx, cy), 5, (255, 255, 0), -1)  # Cyan center point
        
        # Update previous frame
        self.prev_gray = gray
        
        return people, processed_frame

    def detect_tailgating(self, people: List[Person], processed_frame: np.ndarray) -> np.ndarray:
        """Detect tailgating between people and draw visualization"""
        for i, target in enumerate(people):
            for j, follower in enumerate(people):
                if i != j:
                    is_tailgating, tailgating_type, threat_score = self.detector.detect_tailgating(target, follower)
                    
                    if is_tailgating:
                        # Draw line between target and follower with appropriate color
                        color = self.color_map[tailgating_type]
                        
                        # Calculate distance and angle for display
                        dx = follower.x - target.x
                        dy = follower.y - target.y
                        distance = np.sqrt(dx*dx + dy*dy)
                        angle = np.degrees(np.arctan2(dy, dx))
                        if angle < 0:
                            angle += 360
                            
                        # Draw line and circles
                        cv2.line(processed_frame, 
                               (int(target.x), int(target.y)),
                               (int(follower.x), int(follower.y)),
                               color, 3)
                        cv2.circle(processed_frame, (int(target.x), int(target.y)), 7, color, -1)
                        cv2.circle(processed_frame, (int(follower.x), int(follower.y)), 7, color, -1)
                        
                        # Add detailed overlay text
                        info_text = f"Type: {tailgating_type.value}"
                        cv2.putText(processed_frame, info_text,
                                  (int(follower.x), int(follower.y) - 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                        
                        metrics_text = f"Threat: {threat_score:.2f} Dist: {distance:.1f} Spd: {follower.speed:.1f}"
                        cv2.putText(processed_frame, metrics_text,
                                  (int(follower.x), int(follower.y) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        return processed_frame

    def process_video(self, video_name: str):
        """Process a single video file"""
        input_path = os.path.join(self.input_dir, video_name)
        output_path = os.path.join(self.output_dir, f"processed_{video_name}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Validate video properties
        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
        if width != 1280 or height != 720:
            print(f"Warning: Video resolution {width}x{height} differs from expected 1280x720")
        if fps != 25:
            print(f"Warning: Video FPS {fps} differs from expected 25")
            
        # Create video writer with mp4v codec (more widely supported)
        print("Initializing video writer with mp4v codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error: Failed to initialize video writer. Codec may not be supported.")
            return
        
        frame_num = 0
        prev_people = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and detect people
            people, processed_frame = self.process_frame(frame, frame_num)
            print(f"Frame {frame_num}: Detected {len(people)} people")
            
            # Update speeds based on previous positions
            for person in people:
                for prev_person in prev_people:
                    if abs(person.x - prev_person.x) < 50 and abs(person.y - prev_person.y) < 50:
                        person.calculate_speed(prev_person, fps)
                        print(f"Frame {frame_num}: Person speed: {person.speed:.2f} pixels/frame")
            
            # Detect and visualize tailgating
            processed_frame = self.detect_tailgating(people, processed_frame)
            # Add debug circle to verify frame modification
            print("Drawing debug circle at (50,50)")
            cv2.circle(processed_frame, (50, 50), 20, (0, 0, 255), -1)
            print(f"Frame {frame_num}: Added debug circle and detected {len(people)} people with tailgating visualization")
            
            # Write the frame
            out.write(processed_frame)
            
            # Update previous people
            prev_people = people
            frame_num += 1
        
        # Release resources
        cap.release()
        out.release()
        print(f"Processed video saved to {output_path}")

def main():
    processor = VideoProcessor()
    test_videos = [
        "WeChat_20241225162350.mp4",
        "WeChat_20241225162401.mp4",
        "WeChat_20241225162410.mp4",
        "WeChat_20241225162417.mp4"
    ]
    
    for video in test_videos:
        print(f"Processing video: {video}")
        processor.process_video(video)

if __name__ == "__main__":
    main()
