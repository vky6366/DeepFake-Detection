import dlib
import numpy as np
import cv2
import os

class FacialRegionAnalyzer:
    def __init__(self):
        # Initialize facial landmark detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Define base directory and predictor path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Define path relative to the base directory
        predictor_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
        
        # Check if the predictor file exists
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"\nError: Could not find shape_predictor_68_face_landmarks.dat at {predictor_path}\n"
                "Please make sure the file is in the Utils directory"
            )
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define facial regions
        self.FACIAL_REGIONS = {
            'forehead': list(range(17, 27)),
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'nose': list(range(27, 36)),
            'left_cheek': [1, 2, 3, 4, 5, 31, 32, 33],
            'right_cheek': [11, 12, 13, 14, 15, 35, 34, 33],
            'mouth': list(range(48, 68))
        }

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        
        landmarks = self.predictor(gray, faces[0])
        return landmarks

    def analyze_heatmap_regions(self, image, heatmap, landmarks):
        """
        Analyze which facial regions the heatmap is focusing on
        Returns a dictionary with intensity scores for each facial region
        """
        if landmarks is None:
            return None

        region_scores = {}
        landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        for region_name, region_points in self.FACIAL_REGIONS.items():
            # Get points for this region
            region_coords = landmark_points[region_points]
            
            # Create mask for this region
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hull = cv2.convexHull(region_coords)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Calculate average heatmap intensity for this region
            region_heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)
            if np.sum(mask) > 0:  # Avoid division by zero
                avg_intensity = np.sum(region_heatmap) / np.sum(mask > 0)
                region_scores[region_name] = float(avg_intensity)
            else:
                region_scores[region_name] = 0.0
                
        return region_scores

    def get_focused_regions(self, region_scores, threshold=0.5):
        """
        Returns a list of facial regions that have high attention (above threshold)
        """
        if not region_scores:
            return []
            
        # Normalize scores
        max_score = max(region_scores.values())
        if max_score == 0:
            return []
            
        normalized_scores = {
            region: score/max_score 
            for region, score in region_scores.items()
        }
        
        # Get regions above threshold
        focused_regions = [
            region 
            for region, score in normalized_scores.items() 
            if score > threshold
        ]
        
        return focused_regions