"""
Face Detection Module
Uses OpenCV Haar cascade for real-time face detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class FaceDetector:
    def __init__(self, cascade_path: str = None):
        """
        Initialize face detector with Haar cascade
        
        Args:
            cascade_path: Path to Haar cascade XML file. If None, uses default.
        """
        if cascade_path is None:
            # Use OpenCV's built-in Haar cascade
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            self.cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade loaded successfully
        if self.cascade.empty():
            raise ValueError("Failed to load Haar cascade classifier")
    
    def detect_faces(self, frame: np.ndarray, 
                    min_face_size: Tuple[int, int] = (30, 30),
                    scale_factor: float = 1.05,
                    min_neighbors: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input image frame (BGR format)
            min_face_size: Minimum face size (width, height)
            scale_factor: Scale factor for image pyramid
            min_neighbors: Minimum neighbors for detection
            
        Returns:
            List of face bounding boxes as (x, y, w, h) tuples
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_face_size
        )
        
        # Handle different return types
        if faces is None:
            return []
        elif isinstance(faces, tuple):
            return list(faces)
        else:
            return faces.tolist()
    
    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest face from detected faces
        
        Args:
            faces: List of face bounding boxes
            
        Returns:
            Largest face bounding box or None if no faces
        """
        if not faces:
            return None
        
        # Find face with largest area
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return largest_face
    
    def draw_face_box(self, frame: np.ndarray, 
                     face_box: Tuple[int, int, int, int],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on face
        
        Args:
            frame: Input frame
            face_box: Face bounding box (x, y, w, h)
            color: Box color in BGR format
            thickness: Line thickness
            
        Returns:
            Frame with drawn bounding box
        """
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame
    
    def draw_center_dot(self, frame: np.ndarray,
                       face_box: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (0, 0, 255),
                       radius: int = 3) -> np.ndarray:
        """
        Draw center dot on face
        
        Args:
            frame: Input frame
            face_box: Face bounding box (x, y, w, h)
            color: Dot color in BGR format
            radius: Dot radius
            
        Returns:
            Frame with drawn center dot
        """
        x, y, w, h = face_box
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), radius, color, -1)
        return frame
    
    @staticmethod
    def stabilize_face_size(current_box: Tuple[int, int, int, int], 
                           prev_box: Optional[Tuple[int, int, int, int]] = None,
                           size_stability_factor: float = 0.8) -> Tuple[int, int, int, int]:
        """
        Stabilize face bounding box size to prevent drastic changes
        
        Args:
            current_box: Current detected face box (x, y, w, h)
            prev_box: Previous face box for stabilization
            size_stability_factor: Factor for size smoothing (0.0-1.0)
            
        Returns:
            Stabilized face box
        """
        if prev_box is None:
            return current_box
        
        x, y, w, h = current_box
        px, py, pw, ph = prev_box
        
        # Stabilize position
        stable_x = int(size_stability_factor * px + (1 - size_stability_factor) * x)
        stable_y = int(size_stability_factor * py + (1 - size_stability_factor) * y)
        
        # Stabilize size more aggressively to prevent shrinking
        size_factor = 0.9  # Higher factor means more size stability
        stable_w = int(size_factor * pw + (1 - size_factor) * w)
        stable_h = int(size_factor * ph + (1 - size_factor) * h)
        
        # Ensure minimum size
        min_size = 50
        stable_w = max(stable_w, min_size)
        stable_h = max(stable_h, min_size)
        
        # Ensure reasonable aspect ratio (face is typically taller than wide)
        if stable_w < stable_h * 0.7:
            stable_w = int(stable_h * 0.8)
        
        return (stable_x, stable_y, stable_w, stable_h)
    
    @staticmethod
    def get_face_center(face_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get center coordinates of face bounding box
        
        Args:
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Center coordinates (cx, cy)
        """
        x, y, w, h = face_box
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy
