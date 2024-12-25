import cv2
import numpy as np
from tailgating_detector import (
    TailgatingDetector, Person, TailgatingType,
    ThreatCalculator, EntropyWeightCalculator
)
import os

def process_video(video_path, output_path):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize calculators and detector
    threat_calculator = ThreatCalculator()
    entropy_calculator = EntropyWeightCalculator()
    detector = TailgatingDetector(threat_calculator=threat_calculator,
                                entropy_calculator=entropy_calculator)
    
    # Colors for visualization
    colors = {
        TailgatingType.DIRECT: (0, 0, 255),    # Red
        TailgatingType.LATERAL: (0, 255, 0),    # Green
        TailgatingType.HORIZONTAL: (255, 0, 0)  # Blue
    }
    
    frame_count = 0
    prev_frame = None
    people_history = []  # Store previous positions for speed calculation
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create copy for drawing
        vis_frame = frame.copy()
        
        # Draw detection zone
        # Calculate detection zone coordinates
        x1 = int(round(detector.gate_x - detector.detection_zone_width//2))
        x2 = int(round(detector.gate_x + detector.detection_zone_width//2))
        cv2.rectangle(vis_frame, 
                     (x1, 0),
                     (x2, height),
                     (128, 128, 128), 2)
        
        # Draw gate line
        # Draw gate line with integer coordinates
        gate_x = int(round(detector.gate_x))
        cv2.line(vis_frame,
                 (gate_x, 0),
                 (gate_x, height),
                 (255, 0, 0), 2)
        
        # Basic motion detection and person tracking
        if frame_count == 0:
            prev_frame = frame.copy()
        else:
            # Motion detection
            diff = cv2.absdiff(prev_frame, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)  # Increased threshold for better motion detection
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process detected people
            current_people = []
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Increased threshold to reduce noise
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w//2
                    center_y = y + h//2
                    
                    # Create person object
                    person = Person(x=center_x, y=center_y, frame_num=frame_count)
                    
                    # Calculate speed if we have history
                    if len(people_history) > 0:
                        # Find closest previous person
                        min_dist = float('inf')
                        closest_prev = None
                        for prev_person in people_history[-1]:
                            dist = np.sqrt((prev_person.x - center_x)**2 + 
                                         (prev_person.y - center_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_prev = prev_person
                        
                        if closest_prev is not None:
                            # Update position history
                            person.prev_x = closest_prev.x
                            person.prev_y = closest_prev.y
                    
                    current_people.append(person)
                    
                    # Draw bounding box and ID
                    cv2.rectangle(vis_frame, (x,y), (x+w,y+h), (255,255,255), 2)
                    cv2.circle(vis_frame, (center_x, center_y), 5, (0,255,255), -1)
            
            # Update history
            people_history.append(current_people)
            if len(people_history) > 10:  # Keep last 10 frames
                people_history.pop(0)
            
            # Detect tailgating
            if len(current_people) >= 2:
                # Sort people by x-coordinate (distance from gate)
                current_people.sort(key=lambda p: p.x, reverse=True)
                
                # Check each potential target-follower pair
                for i in range(len(current_people)-1):
                    target = current_people[i]  # Person closer to gate
                    for j in range(i+1, len(current_people)):
                        follower = current_people[j]  # Person further from gate
                        
                        # Check for tailgating
                        is_tailgating, tailgating_type, threat_score = detector.detect_tailgating(target, follower)
                        if is_tailgating:
                            # Draw line between target and follower
                            color = colors[tailgating_type]
                            cv2.line(vis_frame, 
                                    (int(round(target.x)), int(round(target.y))),
                                    (int(round(follower.x)), int(round(follower.y))),
                                    color, 2)
                            
                            # Calculate metrics for display
                            distance = target.calculate_distance(follower)
                            # Find previous position for speed calculation
                            if len(people_history) > 1:
                                min_dist = float('inf')
                                prev_follower = None
                                for prev_person in people_history[-2]:  # Look at previous frame
                                    dist = np.sqrt((prev_person.x - follower.x)**2 + 
                                                (prev_person.y - follower.y)**2)
                                    if dist < min_dist:
                                        min_dist = dist
                                        prev_follower = prev_person
                                speed = follower.calculate_speed(prev_follower) if prev_follower else 0.0
                            else:
                                speed = 0.0
                            angle = target.calculate_angle(follower)
                            
                            # Draw metrics
                            y_offset = 30
                            cv2.putText(vis_frame,
                                   f"Threat Score: {threat_score:.2f}",
                                   (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   color,
                                   2)
                            cv2.putText(vis_frame,
                                   f"Type: {tailgating_type.name}",
                                   (10, y_offset + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   color,
                                   2)
                            cv2.putText(vis_frame,
                                   f"Distance: {distance:.1f}px",
                                   (10, y_offset + 60),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   color,
                                   2)
                            cv2.putText(vis_frame,
                                   f"Speed: {speed:.1f}px/frame",
                                   (10, y_offset + 90),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   color,
                                   2)
                            cv2.putText(vis_frame,
                                   f"Angle: {angle:.1f}°",
                                   (10, y_offset + 120),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   color,
                                   2)
            
            prev_frame = frame.copy()
        
        # Write frame
        out.write(vis_frame)
        frame_count += 1
        
    # Release everything
    cap.release()
    out.release()

def main():
    input_dir = "/home/ubuntu/attachments"
    output_dir = "/home/ubuntu/attachments/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test video
    for video_name in ["WeChat_20241225162350.mp4",
                      "WeChat_20241225162401.mp4",
                      "WeChat_20241225162410.mp4",
                      "WeChat_20241225162417.mp4"]:
        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, f"processed_{video_name}")
        
        print(f"Processing {video_name}...")
        process_video(input_path, output_path)
        print(f"Saved processed video to {output_path}")

if __name__ == "__main__":
    main()
