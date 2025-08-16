# -*- coding: utf-8 -*-
# src/models/bev_detector.py

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class BEVDetector:
    def __init__(self, bev_resolution=0.1):
        """
        Object detector for Bird's Eye View images
        
        Args:
            bev_resolution: meters per pixel in BEV
        """
        print("🎯 Initializing BEV Detector...")
        
        self.resolution = bev_resolution
        
        # Detection parameters
        self.min_object_size = int(1.0 / bev_resolution)  # 1 meter minimum
        self.max_object_size = int(10.0 / bev_resolution)  # 10 meters maximum
        
        # Object size templates (in meters, converted to pixels)
        self.object_templates = {
            'car': {'length': int(4.5 / bev_resolution), 'width': int(2.0 / bev_resolution)},
            'truck': {'length': int(8.0 / bev_resolution), 'width': int(2.5 / bev_resolution)},
            'person': {'length': int(0.8 / bev_resolution), 'width': int(0.8 / bev_resolution)},
            'bicycle': {'length': int(1.8 / bev_resolution), 'width': int(0.6 / bev_resolution)}
        }
        
        print(f"   Resolution: {bev_resolution}m per pixel")
        print(f"   Object size range: {self.min_object_size}-{self.max_object_size} pixels")
        print("✅ BEV Detector ready!")
    
    def preprocess_bev(self, bev_image):
        """
        Preprocess BEV image for object detection
        
        Args:
            bev_image: Input BEV image
            
        Returns:
            processed_image: Preprocessed image
            binary_mask: Binary mask of potential objects
        """
        # Convert to grayscale if color
        if len(bev_image.shape) == 3:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bev_image.copy()
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive thresholding to find objects
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return enhanced, cleaned
    
    def detect_objects_contour_based(self, bev_image):
        """
        Detect objects using contour-based method
        
        Args:
            bev_image: BEV image
            
        Returns:
            detections: List of detected objects [x, y, width, height, confidence, class]
        """
        # Preprocess
        processed, binary = self.preprocess_bev(bev_image)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size
            if (w < self.min_object_size or h < self.min_object_size or 
                w > self.max_object_size or h > self.max_object_size):
                continue
            
            if area < self.min_object_size ** 2:
                continue
            
            # Calculate features
            aspect_ratio = max(w, h) / min(w, h)
            solidity = area / (w * h)
            
            # Simple classification based on size and shape
            object_class = self._classify_object(w, h, aspect_ratio, solidity)
            
            # Calculate confidence based on shape regularity
            confidence = min(1.0, solidity * 0.8 + (1.0 / aspect_ratio) * 0.2)
            
            detections.append([x, y, w, h, confidence, object_class])
        
        return detections
    
    def _classify_object(self, width, height, aspect_ratio, solidity):
        """
        Simple object classification based on size and shape
        """
        size = max(width, height)
        
        # Size-based classification
        if size < int(1.5 / self.resolution):  # Small objects
            if aspect_ratio < 1.5:
                return 'person'
            else:
                return 'bicycle'
        elif size < int(6.0 / self.resolution):  # Medium objects
            return 'car'
        else:  # Large objects
            return 'truck'
    
    def detect_objects_template_matching(self, bev_image):
        """
        Detect objects using template matching approach
        
        Args:
            bev_image: BEV image
            
        Returns:
            detections: List of detected objects
        """
        detections = []
        
        # Convert to grayscale
        if len(bev_image.shape) == 3:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bev_image.copy()
        
        # For each object type, look for characteristic patterns
        for obj_class, template_size in self.object_templates.items():
            length, width = template_size['length'], template_size['width']
            
            # Create simple template (filled rectangle)
            template = np.ones((length, width), dtype=np.uint8) * 255
            
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Find peaks
            threshold = 0.3  # Adjust based on needs
            locations = np.where(result >= threshold)
            
            for y, x in zip(*locations):
                confidence = result[y, x]
                detections.append([x, y, width, length, confidence, obj_class])
        
        # Non-maximum suppression to remove overlapping detections
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def _non_max_suppression(self, detections, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        if len(detections) == 0:
            return []
        
        # Convert to numpy array for easier processing
        detections = np.array(detections)
        
        # Calculate areas
        areas = detections[:, 2] * detections[:, 3]
        
        # Sort by confidence
        indices = np.argsort(detections[:, 4])[::-1]
        
        keep = []
        
        while len(indices) > 0:
            # Keep the detection with highest confidence
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining detections
            remaining = indices[1:]
            
            # Get coordinates
            x1 = np.maximum(detections[current, 0], detections[remaining, 0])
            y1 = np.maximum(detections[current, 1], detections[remaining, 1])
            x2 = np.minimum(detections[current, 0] + detections[current, 2], 
                           detections[remaining, 0] + detections[remaining, 2])
            y2 = np.minimum(detections[current, 1] + detections[current, 3],
                           detections[remaining, 1] + detections[remaining, 3])
            
            # Calculate intersection
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate IoU
            union = areas[current] + areas[remaining] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep detections with low IoU
            indices = remaining[iou < overlap_threshold]
        
        return detections[keep].tolist()
    
    def bev_to_world_coordinates(self, detections, bev_range=(-50, 50, -50, 50)):
        """
        Convert BEV pixel coordinates to world coordinates
        
        Args:
            detections: List of detections in BEV coordinates
            bev_range: (x_min, x_max, y_min, y_max) in meters
            
        Returns:
            world_detections: Detections with world coordinates
        """
        if len(detections) == 0:
            return []
        
        x_min, x_max, y_min, y_max = bev_range
        
        world_detections = []
        
        for detection in detections:
            bev_x, bev_y, bev_w, bev_h, confidence, obj_class = detection
            
            # Convert BEV coordinates to world coordinates
            world_x = x_max - (bev_x + bev_h/2) * self.resolution
            world_y = y_min + (bev_y + bev_w/2) * self.resolution
            
            # Convert size to world units
            world_length = bev_h * self.resolution
            world_width = bev_w * self.resolution
            
            world_detections.append({
                'center_x': world_x,
                'center_y': world_y,
                'length': world_length,
                'width': world_width,
                'confidence': confidence,
                'class': obj_class,
                'bev_bbox': [bev_x, bev_y, bev_w, bev_h]
            })
        
        return world_detections
    
    def visualize_detections(self, bev_image, detections, title="BEV Object Detection"):
        """
        Visualize detections on BEV image
        """
        result_image = bev_image.copy()
        
        # Colors for different object classes
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (0, 0, 255),    # Red
            'person': (255, 0, 0),   # Blue
            'bicycle': (255, 255, 0) # Cyan
        }
        
        for detection in detections:
            if len(detection) == 6:  # BEV coordinates
                x, y, w, h, confidence, obj_class = detection
                color = colors.get(obj_class, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(result_image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                
                # Draw label
                label = f"{obj_class} {confidence:.2f}"
                cv2.putText(result_image, label, (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            elif isinstance(detection, dict):  # World coordinates
                bbox = detection['bev_bbox']
                x, y, w, h = bbox
                obj_class = detection['class']
                confidence = detection['confidence']
                
                color = colors.get(obj_class, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(result_image, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                
                # Draw label with world coordinates
                label = f"{obj_class} {confidence:.2f}"
                world_info = f"({detection['center_x']:.1f}, {detection['center_y']:.1f})"
                cv2.putText(result_image, label, (int(x), int(y-20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(result_image, world_info, (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_image

def test_bev_detector():
    """
    Test BEV detector with real data
    """
    print("🧪 Testing BEV Detector...")
    
    # Import required modules
    import sys, os
    sys.path.insert(0, os.getcwd())
    
    try:
        from bev_generator import BEVGenerator
        from lidar_processor import LiDARProcessor
    except ImportError:
        print("❌ Cannot import required modules")
        return
    
    # Initialize components
    bev_gen = BEVGenerator(bev_range=(-50, 50, -50, 50), bev_resolution=0.1)
    bev_detector = BEVDetector(bev_resolution=0.1)
    lidar_proc = LiDARProcessor()
    
    # Find data files
    data_dir = Path("../../data/raw/samples")
    camera_dir = data_dir / "CAM_FRONT"
    lidar_dir = data_dir / "LIDAR_TOP"
    
    if not (camera_dir.exists() and lidar_dir.exists()):
        print("❌ Data directories not found")
        return
    
    # Get first files
    camera_files = sorted(list(camera_dir.glob("*.jpg")))
    lidar_files = sorted(list(lidar_dir.glob("*.pcd.bin")))
    
    if not camera_files or not lidar_files:
        print("❌ No data files found")
        return
    
    print(f"📊 Testing BEV detection with:")
    print(f"  📷 {camera_files[0].name}")
    print(f"  📡 {lidar_files[0].name}")
    
    # Load data
    camera_image = cv2.imread(str(camera_files[0]))
    lidar_points = lidar_proc.load_point_cloud(lidar_files[0])
    
    if camera_image is None or lidar_points is None:
        print("❌ Failed to load data")
        return
    
    # Generate BEV
    filtered_lidar = lidar_proc.filter_points(lidar_points)
    combined_bev, lidar_bev, camera_bev = bev_gen.create_multi_modal_bev(
        filtered_lidar, camera_image
    )
    
    # Add ego vehicle
    final_bev = bev_gen.add_ego_vehicle_to_bev(combined_bev)
    
    print(f"🗺️ BEV generated: {final_bev.shape}")
    
    # Test both detection methods
    print("\n🎯 Testing contour-based detection...")
    contour_detections = bev_detector.detect_objects_contour_based(final_bev)
    print(f"   Found {len(contour_detections)} objects")
    
    print("\n🎯 Testing template-based detection...")
    template_detections = bev_detector.detect_objects_template_matching(final_bev)
    print(f"   Found {len(template_detections)} objects")
    
    # Convert to world coordinates
    world_detections_contour = bev_detector.bev_to_world_coordinates(
        contour_detections, bev_range=(-50, 50, -50, 50)
    )
    
    world_detections_template = bev_detector.bev_to_world_coordinates(
        template_detections, bev_range=(-50, 50, -50, 50)
    )
    
    # Create visualizations
    results_dir = Path("../../results/bev_detection")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize contour-based detections
    contour_result = bev_detector.visualize_detections(
        final_bev, contour_detections, "Contour-based BEV Detection"
    )
    cv2.imwrite(str(results_dir / "bev_detection_contour.jpg"), contour_result)
    
    # Visualize template-based detections
    template_result = bev_detector.visualize_detections(
        final_bev, template_detections, "Template-based BEV Detection"
    )
    cv2.imwrite(str(results_dir / "bev_detection_template.jpg"), template_result)
    
    # Print detailed results
    print(f"\n📊 Contour-based Detection Results:")
    for i, det in enumerate(world_detections_contour[:5]):  # Show top 5
        print(f"  {i+1}. {det['class']} at ({det['center_x']:.1f}, {det['center_y']:.1f})m")
        print(f"     Size: {det['length']:.1f}m x {det['width']:.1f}m (conf: {det['confidence']:.2f})")
    
    print(f"\n📊 Template-based Detection Results:")
    for i, det in enumerate(world_detections_template[:5]):  # Show top 5
        print(f"  {i+1}. {det['class']} at ({det['center_x']:.1f}, {det['center_y']:.1f})m")
        print(f"     Size: {det['length']:.1f}m x {det['width']:.1f}m (conf: {det['confidence']:.2f})")
    
    print(f"\n✅ BEV detection test completed!")
    print(f"📁 Results saved: {results_dir}")
    
    return world_detections_contour, world_detections_template

if __name__ == "__main__":
    test_bev_detector()