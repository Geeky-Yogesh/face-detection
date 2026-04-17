"""
Main Application
Real-time stable face detection and tracking with webcam
"""

import cv2
import time
import numpy as np
import sys
from typing import Optional

from face_detection import FaceDetector
from smoothing import FaceTracker
from tracking import MovementTracker, CoordinateDisplay, TrackingVisualizer


class FaceTrackingApp:
    def __init__(self, 
                 camera_index: int = 0,
                 smoothing_alpha: float = 0.7,
                 hold_frames: int = 7,
                 movement_threshold: int = 5,
                 demo_mode: bool = False):
        """
        Initialize face tracking application
        
        Args:
            camera_index: Webcam camera index
            smoothing_alpha: Alpha value for smoothing (0.0-1.0)
            hold_frames: Frames to hold position when face lost
            movement_threshold: Minimum movement in pixels
        """
        self.camera_index = camera_index
        self.demo_mode = demo_mode
        self.cap = None
        self.demo_frame_count = 0
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_tracker = FaceTracker(smoothing_alpha, hold_frames)
        self.movement_tracker = MovementTracker(movement_threshold)
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Trajectory tracking
        self.trajectory_centers = []
        self.max_trajectory_points = 50
        
    def create_demo_frame(self) -> np.ndarray:
        """Create a demo frame with simulated face movement"""
        h, w = 480, 640
        
        # Create gradient background
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            frame[i] = [i // 3, i // 2, i // 4]
        
        # Simulate face movement with sine wave
        self.demo_frame_count += 1
        t = self.demo_frame_count * 0.05
        
        # Face position moves in a pattern
        face_x = int(w // 2 + 100 * np.sin(t))
        face_y = int(h // 2 + 50 * np.cos(t * 0.7))
        face_size = int(80 + 20 * np.sin(t * 1.3))
        
        # Draw simulated face rectangle
        cv2.rectangle(frame, 
                     (face_x - face_size // 2, face_y - face_size // 2),
                     (face_x + face_size // 2, face_y + face_size // 2),
                     (100, 150, 200), -1)
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add demo text
        cv2.putText(frame, "DEMO MODE - Simulated Face", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def initialize_camera(self) -> bool:
        """Initialize webcam capture or demo mode"""
        if self.demo_mode:
            print("Running in demo mode (simulated face movement)")
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                print("Try running with: python main.py --demo")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Try running with: python main.py --demo")
            return False
    
    def draw_ui(self, frame: np.ndarray, 
                face_box: Optional[tuple],
                center: Optional[tuple],
                dx: int, dy: int, direction: str) -> np.ndarray:
        """
        Draw UI elements on frame
        
        Args:
            frame: Input frame
            face_box: Face bounding box or None
            center: Face center coordinates or None
            dx: Movement delta X
            dy: Movement delta Y
            direction: Movement direction
            
        Returns:
            Frame with UI elements
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), self.bg_color, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Status text
        status = self.face_tracker.get_hold_status()
        status_color = (0, 255, 0) if self.face_tracker.is_face_detected() else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, 35), 
                   self.font, self.font_scale, status_color, self.font_thickness)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, 60), 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Coordinates and movement
        if center:
            coord_text = CoordinateDisplay.format_coords(center[0], center[1], dx, dy)
            cv2.putText(frame, coord_text, (20, 85), 
                       self.font, self.font_scale, self.text_color, self.font_thickness)
            
            # Movement info
            avg_dx, avg_dy = self.movement_tracker.get_average_movement()
            total_dist = self.movement_tracker.get_total_distance()
            movement_text = CoordinateDisplay.format_movement_info(direction, avg_dx, avg_dy, total_dist)
            cv2.putText(frame, movement_text, (20, 110), 
                       self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (w - 150, h - 20), 
                   self.font, 0.5, self.text_color, 1)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for face detection and tracking
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with visualizations
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        current_face = self.face_detector.get_largest_face(faces) if faces else None
        
        # Update face tracker
        tracked_face = self.face_tracker.update(current_face)
        
        # Initialize variables
        center = None
        dx, dy, direction = 0, 0, "No Face"
        
        if tracked_face:
            # Draw face box
            self.face_detector.draw_face_box(frame, tracked_face, (0, 255, 0), 2)
            
            # Get and draw center
            center = self.face_detector.get_face_center(tracked_face)
            self.face_detector.draw_center_dot(frame, tracked_face, (0, 0, 255), 3)
            
            # Update movement tracking
            dx, dy, direction = self.movement_tracker.update(center)
            
            # Add to trajectory
            self.trajectory_centers.append(center)
            if len(self.trajectory_centers) > self.max_trajectory_points:
                self.trajectory_centers.pop(0)
            
            # Draw trajectory
            if len(self.trajectory_centers) > 1:
                TrackingVisualizer.draw_trajectory_points(
                    frame, self.trajectory_centers, (0, 255, 255), 2, 30
                )
                
                # Draw movement vector
                if len(self.trajectory_centers) >= 2:
                    prev_center = self.trajectory_centers[-2]
                    TrackingVisualizer.draw_movement_vector(
                        frame, prev_center, center, (255, 255, 0), 2
                    )
        else:
            # Reset movement tracker when face is lost
            self.movement_tracker.reset()
            self.trajectory_centers.clear()
        
        # Draw UI
        frame = self.draw_ui(frame, tracked_face, center, dx, dy, direction)
        
        return frame
    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        print("Face Tracking Application Started")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame (demo mode or camera)
                if self.demo_mode:
                    frame = self.create_demo_frame()
                    ret = True
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Face Tracking', processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if not self.demo_mode and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Main entry point"""
    # Check for demo mode
    demo_mode = '--demo' in sys.argv
    
    # Configuration
    config = {
        'camera_index': 0,        # Webcam index
        'smoothing_alpha': 0.7,   # Higher = more smoothing
        'hold_frames': 7,         # Frames to hold when face lost
        'movement_threshold': 5,  # Minimum movement in pixels
        'demo_mode': demo_mode    # Use demo mode if no camera
    }
    
    if demo_mode:
        print("=== DEMO MODE ===")
        print("Simulating face movement to demonstrate tracking features")
        print("Press 'q' to quit")
    
    # Create and run application
    app = FaceTrackingApp(**config)
    app.run()


if __name__ == "__main__":
    main()
