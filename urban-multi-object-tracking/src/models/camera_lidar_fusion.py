# -*- coding: utf-8 -*-
# src/models/camera_lidar_fusion.py

import numpy as np
import cv2
from pathlib import Path

class CameraLiDARFusion:
    def __init__(self):
        """
        Simple Camera-LiDAR Fusion for nuScenes
        """
        print("🔗 Initializing Camera-LiDAR Fusion...")
        
        # nuScenes CAM_FRONT approximate parameters
        self.camera_intrinsic = np.array([
            [1266.4, 0, 816.3],
            [0, 1266.4, 491.5],
            [0, 0, 1]
        ])
        
        print("✅ Camera-LiDAR Fusion ready!")
    
    def project_lidar_to_image(self, lidar_points, image_shape):
        """
        Project LiDAR points to camera image (simplified for nuScenes)
        
        Args:
            lidar_points: (N, 3) or (N, 4) LiDAR points [x, y, z, intensity]
            image_shape: (height, width) of camera image
            
        Returns:
            image_points: (N, 2) projected image coordinates
            valid_indices: indices of valid points
        """
        if len(lidar_points) == 0:
            return np.array([]), np.array([])
        
        # Extract 3D coordinates
        points_3d = lidar_points[:, :3]
        
        # Simple filtering - keep points in reasonable range
        # Front direction (x > 0), reasonable side distance (|y| < 50), height (-3 < z < 3)
        valid_mask = (
            (points_3d[:, 0] > 1.0) & (points_3d[:, 0] < 80.0) &  # 1-80m ahead
            (np.abs(points_3d[:, 1]) < 40.0) &                    # within 40m left-right
            (points_3d[:, 2] > -3.0) & (points_3d[:, 2] < 3.0)    # reasonable height
        )
        
        valid_points = points_3d[valid_mask]
        
        if len(valid_points) == 0:
            return np.array([]), np.array([])
        
        # Simplified coordinate transformation for nuScenes
        # LiDAR: x=forward, y=left, z=up
        # Camera: need to transform to camera coordinate system
        cam_x = valid_points[:, 0]  # depth (forward)
        cam_y = -valid_points[:, 1]  # horizontal (left becomes right)
        cam_z = -valid_points[:, 2]  # vertical (up becomes down)
        
        # Perspective projection
        u = (cam_y / cam_x) * self.camera_intrinsic[0, 0] + self.camera_intrinsic[0, 2]
        v = (cam_z / cam_x) * self.camera_intrinsic[1, 1] + self.camera_intrinsic[1, 2]
        
        # Filter points within image bounds
        height, width = image_shape[:2]
        image_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        # Get final valid points
        final_points = np.column_stack([u[image_mask], v[image_mask]])
        valid_indices = np.where(valid_mask)[0][image_mask]
        
        return final_points, valid_indices
    
    def get_lidar_in_detection(self, lidar_points, detection_bbox, image_shape):
        """
        Get LiDAR points that fall within a detection bounding box
        
        Args:
            lidar_points: LiDAR point cloud
            detection_bbox: [x1, y1, x2, y2] bounding box
            image_shape: camera image shape
            
        Returns:
            roi_points: LiDAR points in the detection area
            roi_distances: distances of these points
        """
        # Project LiDAR to image
        image_points, valid_indices = self.project_lidar_to_image(lidar_points, image_shape)
        
        if len(image_points) == 0:
            return np.array([]), np.array([])
        
        x1, y1, x2, y2 = detection_bbox
        
        # Expand bounding box slightly (10% expansion)
        w, h = x2 - x1, y2 - y1
        x1_exp = max(0, x1 - w * 0.1)
        y1_exp = max(0, y1 - h * 0.1)
        x2_exp = min(image_shape[1], x2 + w * 0.1)
        y2_exp = min(image_shape[0], y2 + h * 0.1)
        
        # Find points within bounding box
        u_coords = image_points[:, 0]
        v_coords = image_points[:, 1]
        
        bbox_mask = (
            (u_coords >= x1_exp) & (u_coords <= x2_exp) &
            (v_coords >= y1_exp) & (v_coords <= y2_exp)
        )
        
        if not np.any(bbox_mask):
            return np.array([]), np.array([])
        
        # Get corresponding 3D points
        roi_indices = valid_indices[bbox_mask]
        roi_points = lidar_points[roi_indices]
        
        # Calculate distances
        distances = np.sqrt(roi_points[:, 0]**2 + roi_points[:, 1]**2 + roi_points[:, 2]**2)
        
        return roi_points, distances
    
    def fuse_camera_lidar(self, camera_detections, lidar_points, image_shape):
        """
        Fuse camera detections with LiDAR data
        
        Args:
            camera_detections: List of [x1, y1, x2, y2, confidence, class_name]
            lidar_points: LiDAR point cloud
            image_shape: camera image shape
            
        Returns:
            fused_detections: Enhanced detections with 3D info
        """
        print(f"🔗 Fusing {len(camera_detections)} detections with LiDAR...")
        
        fused_results = []
        
        for detection in camera_detections:
            x1, y1, x2, y2, confidence, class_name = detection
            
            # Get LiDAR points for this detection
            roi_points, distances = self.get_lidar_in_detection(
                lidar_points, [x1, y1, x2, y2], image_shape
            )
            
            # Create enhanced detection
            enhanced_detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_name': class_name,
                'lidar_points': len(roi_points),
                'has_lidar_support': len(roi_points) > 0
            }
            
            # Add distance information if LiDAR support exists
            if len(distances) > 0:
                enhanced_detection['distance'] = float(np.mean(distances))
                enhanced_detection['min_distance'] = float(np.min(distances))
                enhanced_detection['max_distance'] = float(np.max(distances))
                enhanced_detection['fusion_quality'] = min(1.0, len(roi_points) / 50.0)  # 0-1 scale
            else:
                enhanced_detection['distance'] = float('inf')
                enhanced_detection['fusion_quality'] = 0.0
            
            fused_results.append(enhanced_detection)
        
        # Sort by fusion quality (better LiDAR support first)
        fused_results.sort(key=lambda x: x['fusion_quality'], reverse=True)
        
        successful_fusions = sum(1 for det in fused_results if det['has_lidar_support'])
        print(f"✅ Fusion complete: {successful_fusions}/{len(fused_results)} detections have LiDAR support")
        
        return fused_results
    
    def visualize_fusion(self, image, fused_detections, lidar_points):
        """
        Visualize fusion results
        """
        result_image = image.copy()
        
        # Project and draw LiDAR points as background
        image_points, valid_indices = self.project_lidar_to_image(lidar_points, image.shape)
        
        # Draw LiDAR points colored by distance
        for i, (u, v) in enumerate(image_points):
            original_idx = valid_indices[i]
            distance = np.sqrt(np.sum(lidar_points[original_idx, :3]**2))
            
            # Color by distance: green=close, yellow=medium, red=far
            if distance < 20:
                color = (0, 255, 0)  # Green
            elif distance < 40:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.circle(result_image, (int(u), int(v)), 1, color, -1)
        
        # Draw fused detections
        for detection in fused_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Color based on fusion quality
            if detection['has_lidar_support']:
                if detection['fusion_quality'] > 0.5:
                    bbox_color = (0, 255, 0)  # Green - good fusion
                else:
                    bbox_color = (0, 255, 255)  # Yellow - medium fusion
            else:
                bbox_color = (0, 0, 255)  # Red - no LiDAR support
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), bbox_color, 3)
            
            # Create label
            if detection['has_lidar_support']:
                label = f"{class_name} {confidence:.2f} | {detection['distance']:.1f}m ({detection['lidar_points']}pts)"
            else:
                label = f"{class_name} {confidence:.2f} | No LiDAR"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_image, (x1, y1-text_height-10), (x1+text_width, y1), bbox_color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add summary info
        total_detections = len(fused_detections)
        successful_fusions = sum(1 for det in fused_detections if det['has_lidar_support'])
        lidar_points_shown = len(image_points)
        
        summary = f"Detections: {total_detections} | LiDAR Support: {successful_fusions} | Points: {lidar_points_shown}"
        cv2.putText(result_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result_image

def test_fusion():
    """
    Test fusion with real nuScenes data
    """
    print("🧪 Testing Camera-LiDAR Fusion...")
    
    # Import required modules
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '..', '..'))
    
    try:
        from simple_detector import SimpleDetector
        from lidar_processor import LiDARProcessor
    except ImportError:
        print("❌ Cannot import required modules")
        print("Make sure you're running from the right directory")
        return
    
    # Initialize components
    fusion = CameraLiDARFusion()
    detector = SimpleDetector()
    lidar_proc = LiDARProcessor()
    
    # Find data files
    data_dir = Path("../../data/raw/samples")
    camera_dir = data_dir / "CAM_FRONT"
    lidar_dir = data_dir / "LIDAR_TOP"
    
    if not (camera_dir.exists() and lidar_dir.exists()):
        print("❌ Data directories not found")
        print(f"Looking for: {camera_dir} and {lidar_dir}")
        return
    
    # Get first files
    camera_files = sorted(list(camera_dir.glob("*.jpg")))
    lidar_files = sorted(list(lidar_dir.glob("*.pcd.bin")))
    
    if not camera_files or not lidar_files:
        print("❌ No data files found")
        return
    
    print(f"📊 Testing with first files:")
    print(f"  📷 Camera: {camera_files[0].name}")
    print(f"  📡 LiDAR: {lidar_files[0].name}")
    
    # Load data
    image = cv2.imread(str(camera_files[0]))
    lidar_points = lidar_proc.load_point_cloud(lidar_files[0])
    
    if image is None or lidar_points is None:
        print("❌ Failed to load data")
        return
    
    # Process
    filtered_lidar = lidar_proc.filter_points(lidar_points)
    camera_detections = detector.detect_objects(image)
    
    print(f"📊 Data loaded:")
    print(f"  Image: {image.shape}")
    print(f"  LiDAR: {len(lidar_points)} → {len(filtered_lidar)} points")
    print(f"  Detections: {len(camera_detections)}")
    
    # Fusion
    fused_detections = fusion.fuse_camera_lidar(camera_detections, filtered_lidar, image.shape)
    
    # Visualize
    result_image = fusion.visualize_fusion(image, fused_detections, filtered_lidar)
    
    # Save result
    results_dir = Path("../../results/fusion_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "fusion_test_result.jpg"
    cv2.imwrite(str(output_file), result_image)
    
    print(f"✅ Test completed!")
    print(f"💾 Result saved: {output_file}")
    
    # Print detailed results
    print(f"\n📊 Detailed Results:")
    for i, det in enumerate(fused_detections[:3]):  # Show top 3
        print(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f})")
        if det['has_lidar_support']:
            print(f"     Distance: {det['distance']:.1f}m ({det['lidar_points']} points)")
        else:
            print(f"     No LiDAR support")

if __name__ == "__main__":
    test_fusion()