"""
Smoothing Module
Provides alpha-blend smoothing for stable face tracking
"""

import numpy as np
from typing import Tuple, Optional


class Smoother:
    def __init__(self, alpha: float = 0.7):
        """
        Initialize smoother with alpha value
        
        Args:
            alpha: Smoothing factor (0.0 to 1.0)
                  Higher values = more smoothing (prev weighted more)
                  Lower values = less smoothing (curr weighted more)
        """
        self.alpha = alpha
        self.prev_box = None
        self.prev_center = None
        
    def smooth_box(self, current_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Apply alpha-blend smoothing to bounding box
        
        Args:
            current_box: Current bounding box (x, y, w, h)
            
        Returns:
            Smoothed bounding box
        """
        if self.prev_box is None:
            self.prev_box = current_box
            return current_box
        
        # Alpha-blend: new = alpha * prev + (1-alpha) * curr
        x, y, w, h = current_box
        px, py, pw, ph = self.prev_box
        
        smoothed_x = int(self.alpha * px + (1 - self.alpha) * x)
        smoothed_y = int(self.alpha * py + (1 - self.alpha) * y)
        smoothed_w = int(self.alpha * pw + (1 - self.alpha) * w)
        smoothed_h = int(self.alpha * ph + (1 - self.alpha) * h)
        
        self.prev_box = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        return self.prev_box
    
    def smooth_center(self, current_center: Tuple[int, int]) -> Tuple[int, int]:
        """
        Apply alpha-blend smoothing to center coordinates
        
        Args:
            current_center: Current center coordinates (cx, cy)
            
        Returns:
            Smoothed center coordinates
        """
        if self.prev_center is None:
            self.prev_center = current_center
            return current_center
        
        cx, cy = current_center
        pcx, pcy = self.prev_center
        
        smoothed_cx = int(self.alpha * pcx + (1 - self.alpha) * cx)
        smoothed_cy = int(self.alpha * pcy + (1 - self.alpha) * cy)
        
        self.prev_center = (smoothed_cx, smoothed_cy)
        return self.prev_center
    
    def reset(self):
        """Reset smoother state"""
        self.prev_box = None
        self.prev_center = None
    
    def set_alpha(self, alpha: float):
        """
        Update smoothing factor
        
        Args:
            alpha: New smoothing factor (0.0 to 1.0)
        """
        if 0.0 <= alpha <= 1.0:
            self.alpha = alpha
        else:
            raise ValueError("Alpha must be between 0.0 and 1.0")


class FaceTracker:
    def __init__(self, alpha: float = 0.7, hold_frames: int = 7):
        """
        Initialize face tracker with smoothing and hold functionality
        
        Args:
            alpha: Smoothing factor for stable tracking
            hold_frames: Number of frames to hold last position when face is lost
        """
        self.smoother = Smoother(alpha)
        self.hold_frames = hold_frames
        self.frames_without_face = 0
        self.last_known_box = None
        self.last_known_center = None
        self.is_tracking = False
        self.prev_raw_box = None  # Store raw detection for size stabilization
        
    def update(self, current_box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker with new face detection
        
        Args:
            current_box: Current detected face box or None if no face
            
        Returns:
            Smoothed face box or None if tracking lost
        """
        if current_box is not None:
            # Face detected - update tracking
            self.is_tracking = True
            self.frames_without_face = 0
            
            # Apply size stabilization before smoothing
            from face_detection import FaceDetector
            stabilized_box = FaceDetector.stabilize_face_size(current_box, self.prev_raw_box)
            self.prev_raw_box = current_box
            
            # Smooth the stabilized box
            smoothed_box = self.smoother.smooth_box(stabilized_box)
            self.last_known_box = smoothed_box
            self.last_known_center = (smoothed_box[0] + smoothed_box[2] // 2,
                                    smoothed_box[1] + smoothed_box[3] // 2)
            return smoothed_box
        
        else:
            # No face detected
            if self.is_tracking and self.frames_without_face < self.hold_frames:
                # Hold last known position
                self.frames_without_face += 1
                return self.last_known_box
            else:
                # Lost tracking
                self.is_tracking = False
                self.frames_without_face = 0
                self.smoother.reset()
                self.prev_raw_box = None
                return None
    
    def get_center(self) -> Optional[Tuple[int, int]]:
        """Get current tracked center"""
        return self.last_known_center
    
    def is_face_detected(self) -> bool:
        """Check if face is currently being tracked"""
        return self.is_tracking
    
    def get_hold_status(self) -> str:
        """Get current hold status"""
        if not self.is_tracking:
            return "No Face"
        elif self.frames_without_face > 0:
            return f"Holding ({self.hold_frames - self.frames_without_face})"
        else:
            return "Face Detected"
    
    def reset(self):
        """Reset tracker state"""
        self.smoother.reset()
        self.frames_without_face = 0
        self.last_known_box = None
        self.last_known_center = None
        self.is_tracking = False
        self.prev_raw_box = None
