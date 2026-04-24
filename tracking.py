"""
Tracking Module
Handles center tracking and movement delta calculation
"""

import numpy as np
from typing import Tuple, Optional


class DistanceEstimator:
    """Simple distance estimation using face size"""
    
    def __init__(self):
        # Average face width in meters
        self.average_face_width = 0.15
        
        # Calibration factor (pixels per meter at 1m distance)
        # This is an approximation - you can calibrate it
        self.pixels_per_meter = 400  # Default: 400px = 1m at 1m distance
    
    def estimate_distance(self, face_width_pixels: int) -> float:
        """
        Estimate distance from camera based on face width
        
        Args:
            face_width_pixels: Detected face width in pixels
            
        Returns:
            Estimated distance in meters
        """
        if face_width_pixels <= 0:
            return 0.0
        
        # Using inverse proportion: distance = (known_width * pixel_ratio) / detected_width
        # Simplified: distance = average_face_width / (detected_width / pixels_per_meter)
        face_width_meters = face_width_pixels / self.pixels_per_meter
        
        if face_width_meters > 0:
            return self.average_face_width / face_width_meters
        
        return 0.0
    
    def calibrate(self, known_distance: float, measured_face_width: int):
        """
        Calibrate the distance estimation
        
        Args:
            known_distance: Actual distance from camera in meters
            measured_face_width: Face width measured at that distance in pixels
        """
        # Calculate pixels per meter based on known distance
        face_width_at_distance = measured_face_width / known_distance
        self.pixels_per_meter = face_width_at_distance / self.average_face_width
        
        print(f"Calibrated: {known_distance}m distance = {measured_face_width}px face width")
        print(f"Pixels per meter: {self.pixels_per_meter:.2f}")
    
    def get_distance_color(self, distance: float) -> tuple:
        """
        Get color based on distance (green = close, red = far)
        
        Args:
            distance: Distance in meters
            
        Returns:
            BGR color tuple
        """
        if distance < 0.5:
            return (0, 255, 0)  # Green - very close
        elif distance < 1.0:
            return (0, 255, 255)  # Yellow - close
        elif distance < 2.0:
            return (0, 165, 255)  # Orange - medium
        else:
            return (0, 0, 255)  # Red - far


class MovementTracker:
    def __init__(self, movement_threshold: int = 5):
        """
        Initialize movement tracker
        
        Args:
            movement_threshold: Minimum pixel movement to consider as movement
        """
        self.movement_threshold = movement_threshold
        self.prev_center = None
        self.movement_history = []
        self.max_history = 10
        
    def update(self, current_center: Tuple[int, int]) -> Tuple[int, int, str]:
        """
        Update tracking with new center position
        
        Args:
            current_center: Current center coordinates (cx, cy)
            
        Returns:
            Tuple of (dx, dy, direction_label)
        """
        if self.prev_center is None:
            self.prev_center = current_center
            return 0, 0, "Initial"
        
        # Calculate movement delta
        dx = current_center[0] - self.prev_center[0]
        dy = current_center[1] - self.prev_center[1]
        
        # Determine movement direction
        direction = self._get_direction_label(dx, dy)
        
        # Update movement history
        self.movement_history.append((dx, dy))
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
        
        # Update previous center
        self.prev_center = current_center
        
        return dx, dy, direction
    
    def _get_direction_label(self, dx: int, dy: int) -> str:
        """
        Get directional label based on movement delta
        
        Args:
            dx: X-axis movement
            dy: Y-axis movement
            
        Returns:
            Direction label string
        """
        # Check if movement is significant
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return "Still"
        
        directions = []
        
        # Horizontal movement
        if dx > self.movement_threshold:
            directions.append("Right")
        elif dx < -self.movement_threshold:
            directions.append("Left")
        
        # Vertical movement
        if dy > self.movement_threshold:
            directions.append("Down")
        elif dy < -self.movement_threshold:
            directions.append("Up")
        
        return "-".join(directions) if directions else "Still"
    
    def get_average_movement(self) -> Tuple[float, float]:
        """
        Get average movement from history
        
        Returns:
            Average (dx, dy) movement
        """
        if not self.movement_history:
            return 0.0, 0.0
        
        avg_dx = sum(m[0] for m in self.movement_history) / len(self.movement_history)
        avg_dy = sum(m[1] for m in self.movement_history) / len(self.movement_history)
        
        return avg_dx, avg_dy
    
    def get_total_distance(self) -> float:
        """
        Get total distance traveled from history
        
        Returns:
            Total pixel distance
        """
        if not self.movement_history:
            return 0.0
        
        total_distance = sum(np.sqrt(dx**2 + dy**2) for dx, dy in self.movement_history)
        return total_distance
    
    def reset(self):
        """Reset tracker state"""
        self.prev_center = None
        self.movement_history = []
    
    def set_movement_threshold(self, threshold: int):
        """
        Update movement threshold
        
        Args:
            threshold: New movement threshold in pixels
        """
        self.movement_threshold = threshold


class CoordinateDisplay:
    @staticmethod
    def format_coords(cx: int, cy: int, dx: int = 0, dy: int = 0) -> str:
        """
        Format coordinates for display
        
        Args:
            cx: Center X coordinate
            cy: Center Y coordinate
            dx: Movement delta X
            dy: Movement delta Y
            
        Returns:
            Formatted coordinate string
        """
        return f"Center: ({cx}, {cy}) | Delta: ({dx:+d}, {dy:+d})"
    
    @staticmethod
    def format_distance(distance: float) -> str:
        """
        Format distance for display
        
        Args:
            distance: Distance in meters
            
        Returns:
            Formatted distance string
        """
        return f"Distance: {distance:.2f}m"
    
    @staticmethod
    def format_movement_info(direction: str, avg_dx: float, avg_dy: float, 
                           total_distance: float) -> str:
        """
        Format movement information for display
        
        Args:
            direction: Current movement direction
            avg_dx: Average X movement
            avg_dy: Average Y movement
            total_distance: Total distance traveled
            
        Returns:
            Formatted movement string
        """
        return (f"Direction: {direction} | "
                f"Avg: ({avg_dx:+.1f}, {avg_dy:+.1f}) | "
                f"Distance: {total_distance:.1f}px")


class TrackingVisualizer:
    @staticmethod
    def draw_movement_vector(frame: np.ndarray,
                           start_pos: Tuple[int, int],
                           end_pos: Tuple[int, int],
                           color: Tuple[int, int, int] = (255, 255, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw movement vector arrow
        
        Args:
            frame: Input frame
            start_pos: Start position (x, y)
            end_pos: End position (x, y)
            color: Arrow color in BGR format
            thickness: Line thickness
            
        Returns:
            Frame with drawn movement vector
        """
        # Draw line
        cv2 = __import__('cv2')
        cv2.arrowedLine(frame, start_pos, end_pos, color, thickness, tipLength=0.3)
        return frame
    
    @staticmethod
    def draw_trajectory_points(frame: np.ndarray,
                              centers: list,
                              color: Tuple[int, int, int] = (0, 255, 255),
                              radius: int = 2,
                              max_points: int = 20) -> np.ndarray:
        """
        Draw trajectory points
        
        Args:
            frame: Input frame
            centers: List of center coordinates
            color: Point color in BGR format
            radius: Point radius
            max_points: Maximum number of points to draw
            
        Returns:
            Frame with drawn trajectory
        """
        cv2 = __import__('cv2')
        
        # Draw only recent points
        recent_centers = centers[-max_points:] if len(centers) > max_points else centers
        
        for i, center in enumerate(recent_centers):
            # Fade older points
            alpha = (i + 1) / len(recent_centers)
            point_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, center, radius, point_color, -1)
        
        return frame
