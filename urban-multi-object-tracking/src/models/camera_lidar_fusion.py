# -*- coding: utf-8 -*-
# src/models/camera_lidar_fusion.py

import numpy as np
import cv2
import json
from pathlib import Path

class CameraLiDARFusion:
    def __init__(self, data_root="data/raw"):
        """
        Camera-LiDAR sensor fusion for object detection
        """
        print("🔗 Initializing Camera-LiDAR Fusion...")
        
        self.data_root = Path(data_root)
        self.calibration_data = None
        
        # Load calibration data if available
        self._load_calibration()
        
        # Fusion parameters
        self.depth_threshold = 2.0    # meters
        self.roi_expand_ratio = 0.1   # expand detection ROI by 10%
        
        print("✅ Camera-LiDAR Fusion ready!")
    
    def _load_calibration(self):
        """
        Load camera-LiDAR calibration matrices from nuScenes
        """
        calib_file = self.data_root / "v1.0-mini" / "calibrated_sensor.json"
        
        if calib_file.exists():
            try:
                with open(calib_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"📊 Loaded calibration data: {len(self.calibration_data)} sensors")
            except Exception as e:
                print(f"⚠️  Failed to load calibration: {e}")
                self._use_default_calibration()
        else:
            print("⚠️  No calibration file found, using defaults")
            self._use_default_calibration()
    
    def _use_default_calibration(self):
        """
        Use default calibration parameters for nuScenes
        """
        # Simplified calibration for nuScenes CAM_FRONT
        self.camera_intrinsic = np.array([
            [1266.4, 0, 816.3],
            [0, 1266.4, 491.5],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # LiDAR to camera transformation (simplified)
        self.lidar_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, -1.8],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        print("📐 Using default calibration parameters")
    
    def project_lidar_to_camera(self, lidar_points, camera_intrinsic=None, transform_matrix=None):
        """
        Project LiDAR points to camera image coordinates
        
        Args:
            lidar_points: (N, 3) or (N, 4) array of LiDAR points [x, y, z, intensity]
            camera_intrinsic: Camera intrinsic matrix (3x3)
            transform_matrix: LiDAR to camera transformation matrix (4x4)
            
        Returns:
            image_points: (N, 2) array of image coordinates [u, v]
            depths: (N,) array of depths
            valid_mask: (N,) boolean mask for points in front of camera
        """
        if camera_intrinsic is None:
            camera_intrinsic = self.camera_intrinsic
        
        if transform_matrix is None:
            transform_matrix = self.lidar_to_camera
        
        # Convert to homogeneous coordinates
        if lidar_points.shape[1] == 3:
            points_3d = lidar_points
        else:
            points_3d = lidar_points[:, :3]  # Use only x, y, z
        
        # Add homogeneous coordinate
        points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # Transform to camera coordinate system
        camera_points = (transform_matrix @ points_homogeneous.T).T
        
        # Get depths (z-coordinate in camera frame)
        depths = camera_points[:, 2]
        
        # Filter points behind camera
        valid_mask = depths > 0.1  # At least 10cm in front
        
        # Project to image coordinates
        camera_coords = camera_points[valid_mask, :3]
        
        if len(camera_coords) == 0:
            return np.array([]), np.array([]), valid_mask
        
        # Project using camera intrinsic matrix
        image_coords = (camera_intrinsic @ camera_coords.T).T
        
        # Convert from homogeneous to image coordinates
        image_points = np.zeros((len(valid_mask), 2))
        image_points[valid_mask, 0] = image_coords[:, 0] / image_coords[:, 2]  # u
        image_points[valid_mask, 1] = image_coords[:, 1] / image_coords[:, 2]  # v
        
        return image_points, depths, valid_mask
    
    def filter_lidar_in_image_roi(self, lidar_points, detection_bbox, image_shape):
        """
        Filter LiDAR points that fall within a 2D detection bounding box
        
        Args:
            lidar_points: (N, 3) or (N, 4) array of LiDAR points
            detection_bbox: [x1, y1, x2, y2] detection bounding box
            image_shape: (height, width) of the image
            
        Returns:
            roi_points: LiDAR points within the ROI
            roi_indices: Original indices of the ROI points
        """
        # Project LiDAR to image
        image_points, depths, valid_mask = self.project_lidar_to_camera(lidar_points)
        
        if len(image_points) == 0:
            return np.array([]), np.array([])
        
        # Extract bounding box
        x1, y1, x2, y2 = detection_bbox
        
        # Expand ROI slightly
        expand_x = (x2 - x1) * self.roi_expand_ratio
        expand_y = (y2 - y1) * self.roi_expand_ratio
        
        x1_exp = max(0, x1 - expand_x)
        y1_exp = max(0, y1 - expand_y)
        x2_exp = min(image_shape[1], x2 + expand_x)
        y2_exp = min(image_shape[0], y2 + expand_y)
        
        # Find points within expanded bounding box
        u_coords = image_points[:, 0]
        v_coords = image_points[:, 1]
        
        roi_mask = (
            (u_coords >= x1_exp) & (u_coords <= x2_exp) &
            (v_coords >= y1_exp) & (v_coords <= y2_exp) &
            valid_mask
        )
        
        roi_indices = np.where(roi_mask)[0]
        roi_points = lidar_points[roi_indices]
        
        return roi_points, roi_indices
    
    def estimate_3d_bbox_from_lidar(self, roi_points):
        """
        Estimate 3D bounding box from LiDAR points in ROI
        
        Args:
            roi_points: LiDAR points within detection ROI
            
        Returns:
            bbox_3d: Dictionary with 3D bounding box information
        """
        if len(roi_points) < 10:  # Need minimum points for reliable estimation
            return None
        
        points_3d = roi_points[:, :3] if roi_points.shape[1] > 3 else roi_points
        
        # Calculate bounding box statistics
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        # Calculate distance from origin (camera/ego vehicle)
        distance = np.linalg.norm(center)
        
        # Estimate object properties
        bbox_3d = {
            'center': center,
            'size': size,
            'distance': distance,
            'num_points': len(roi_points),
            'min_coords': min_coords,
            'max_coords': max_coords
        }
        
        return bbox_3d
    
    def fuse_detections(self, camera_detections, lidar_points, image_shape):
        """
        Fuse camera detections with LiDAR data
        
        Args:
            camera_detections: List of [x1, y1, x2, y2, confidence, class_name]
            lidar_points: (N, 3) or (N, 4) array of LiDAR points
            image_shape: (height, width) of camera image
            
        Returns:
            fused_detections: Enhanced detections with 3D information
        """
        print(f"🔗 Fusing {len(camera_detections)} detections with LiDAR data...")
        
        fused_detections = []
        
        for i, detection in enumerate(camera_detections):
            x1, y1, x2, y2, confidence, class_name = detection
            
            # Get LiDAR points in this detection's ROI
            roi_points, roi_indices = self.filter_lidar_in_image_roi(
                lidar_points, [x1, y1, x2, y2], image_shape
            )
            
            # Estimate 3D bounding box
            bbox_3d = self.estimate_3d_bbox_from_lidar(roi_points)
            
            # Create fused detection
            fused_detection = {
                'bbox_2d': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_name,
                'lidar_points': roi_points,
                'bbox_3d': bbox_3d,
                'roi_indices': roi_indices
            }
            
            # Add reliability score based on LiDAR support
            if bbox_3d is not None:
                fused_detection['fusion_confidence'] = min(1.0, bbox_3d['num_points'] / 100.0)
                fused_detection['distance'] = bbox_3d['distance']
            else:
                fused_detection['fusion_confidence'] = 0.0
                fused_detection['distance'] = float('inf')
            
            fused_detections.append(fused_detection)
        
        # Sort by fusion confidence (best LiDAR support first)
        fused_detections.sort(key=lambda x: x['fusion_confidence'], reverse=True)
        
        print(f"✅ Fusion complete. {sum(1 for d in fused_detections if d['bbox_3d'] is not None)} detections have 3D support")
        
        return fused_detections
    
    def visualize_fusion(self, image, fused_detections, lidar_points=None):
        """
        Visualize fused detections on camera image
        """
        result_image = image.copy()
        
        # Project all LiDAR points to image (for background)
        if lidar_points is not None:
            image_points, depths, valid_mask = self.project_lidar_to_camera(lidar_points)
            
            # Draw LiDAR points as small dots
            for i, (u, v) in enumerate(image_points[valid_mask]):
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    depth = depths[valid_mask][i]
                    # Color by depth (blue=close, red=far)
                    color_intensity = min(255, int(depth * 10))
                    cv2.circle(result_image, (int(u), int(v)), 1, (color_intensity, 0, 255-color_intensity), -1)
        
        # Draw fused detections
        for detection in fused_detections:
            x1, y1, x2, y2 = [int(c) for c in detection['bbox_2d']]
            confidence = detection['confidence']
            class_name = detection['class_name']
            fusion_conf = detection['fusion_confidence']
            
            # Color based on fusion quality
            if fusion_conf > 0.5:
                color = (0, 255, 0)  # Green for good fusion
            elif fusion_conf > 0.2:
                color = (0, 255, 255)  # Yellow for medium fusion
            else:
                color = (0, 0, 255)  # Red for poor fusion
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with fusion information
            if detection['bbox_3d'] is not None:
                distance = detection['distance']
                num_points = detection['bbox_3d']['num_points']
                label = f"{class_name} {confidence:.2f} | {distance:.1f}m ({num_points}pts)"
            else:
                label = f"{class_name} {confidence:.2f} | No LiDAR"
            
            # Label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            
            # Label text
            cv2.putText(result_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

# Test function
def test_fusion():
    """
    Test camera-LiDAR fusion with dummy data
    """
    print("🧪 Testing Camera-LiDAR Fusion...")
    
    fusion = CameraLiDARFusion()
    
    # Create dummy data
    dummy_image = np.zeros((900, 1600, 3), dtype=np.uint8)
    
    # Dummy camera detections
    dummy_detections = [
        [400, 300, 600, 500, 0.9, 'car'],
        [800, 200, 1000, 400, 0.8, 'person']
    ]
    
    # Dummy LiDAR points (some points near the detections)
    dummy_lidar = np.array([
        [10, 2, 0.5, 100],    # Points for first detection
        [12, 2, 0.8, 120],
        [11, 1.5, 0.6, 110],
        [20, -1, 1.2, 80],    # Points for second detection
        [21, -0.8, 1.0, 90],
        [5, 5, 0, 150],       # Background points
        [30, 10, -0.5, 60]
    ])
    
    # Test fusion
    fused_detections = fusion.fuse_detections(dummy_detections, dummy_lidar, dummy_image.shape[:2])
    
    print(f"✅ Fusion test completed!")
    print(f"📊 Input: {len(dummy_detections)} camera detections")
    print(f"📊 Output: {len(fused_detections)} fused detections")
    
    for i, det in enumerate(fused_detections):
        print(f"  Detection {i+1}: {det['class_name']} (fusion_conf: {det['fusion_confidence']:.2f})")

if __name__ == "__main__":
    test_fusion()