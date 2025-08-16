# -*- coding: utf-8 -*-
# src/models/bev_generator.py

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

class BEVGenerator:
    def __init__(self, bev_range=(-50, 50, -50, 50), bev_resolution=0.1):
        """
        Bird's Eye View Generator for nuScenes data
        
        Args:
            bev_range: (x_min, x_max, y_min, y_max) in meters
            bev_resolution: meters per pixel
        """
        print("🗺️ Initializing BEV Generator...")
        
        self.x_min, self.x_max, self.y_min, self.y_max = bev_range
        self.resolution = bev_resolution
        
        # Calculate BEV image size
        self.bev_width = int((self.x_max - self.x_min) / self.resolution)
        self.bev_height = int((self.y_max - self.y_min) / self.resolution)
        
        print(f"   BEV range: {bev_range} meters")
        print(f"   Resolution: {bev_resolution}m per pixel")
        print(f"   BEV image size: {self.bev_width} x {self.bev_height}")
        
        # Camera intrinsics for nuScenes CAM_FRONT (approximate)
        self.camera_intrinsic = np.array([
            [1266.4, 0, 816.3],
            [0, 1266.4, 491.5],
            [0, 0, 1]
        ])
        
        print("✅ BEV Generator ready!")
    
    def world_to_bev_coords(self, world_points):
        """
        Convert world coordinates to BEV image coordinates
        
        Args:
            world_points: (N, 2) or (N, 3) array of world coordinates [x, y, z]
            
        Returns:
            bev_coords: (N, 2) array of BEV image coordinates [u, v]
        """
        if len(world_points) == 0:
            return np.array([])
        
        if world_points.shape[1] >= 2:
            x_world = world_points[:, 0]
            y_world = world_points[:, 1]
        else:
            return np.array([])
        
        # Convert to BEV coordinates
        # X-axis: forward direction (top of BEV image)
        # Y-axis: left-right direction (left-right of BEV image)
        bev_x = ((self.x_max - x_world) / self.resolution).astype(int)  # Flip for image coordinates
        bev_y = ((y_world - self.y_min) / self.resolution).astype(int)
        
        # Filter points within BEV range
        valid_mask = (
            (bev_x >= 0) & (bev_x < self.bev_height) &
            (bev_y >= 0) & (bev_y < self.bev_width)
        )
        
        bev_coords = np.column_stack([bev_x[valid_mask], bev_y[valid_mask]])
        return bev_coords, valid_mask
    
    def lidar_to_bev(self, lidar_points, intensity_threshold=20):
        """
        Generate BEV image from LiDAR point cloud
        
        Args:
            lidar_points: (N, 4) array [x, y, z, intensity]
            intensity_threshold: minimum intensity for point inclusion
            
        Returns:
            bev_image: BEV representation as image
            height_map: Height information for each pixel
        """
        if len(lidar_points) == 0:
            return np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8), np.zeros((self.bev_height, self.bev_width))
        
        # Filter points by height (ground level objects)
        height_mask = (lidar_points[:, 2] > -2.5) & (lidar_points[:, 2] < 2.0)
        valid_points = lidar_points[height_mask]
        
        if len(valid_points) == 0:
            return np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8), np.zeros((self.bev_height, self.bev_width))
        
        # Convert to BEV coordinates
        bev_coords, coord_mask = self.world_to_bev_coords(valid_points[:, :3])
        
        if len(bev_coords) == 0:
            return np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8), np.zeros((self.bev_height, self.bev_width))
        
        # Get corresponding point attributes
        valid_points_filtered = valid_points[coord_mask]
        heights = valid_points_filtered[:, 2]
        intensities = valid_points_filtered[:, 3] if valid_points_filtered.shape[1] > 3 else np.ones(len(valid_points_filtered))
        
        # Create BEV images
        bev_image = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        height_map = np.full((self.bev_height, self.bev_width), -10.0)  # Initialize with low value
        density_map = np.zeros((self.bev_height, self.bev_width))
        
        # Populate BEV image
        for i, (bev_x, bev_y) in enumerate(bev_coords):
            height = heights[i]
            intensity = intensities[i]
            
            # Update height map (keep highest point)
            if height > height_map[bev_x, bev_y]:
                height_map[bev_x, bev_y] = height
            
            # Update density
            density_map[bev_x, bev_y] += 1
            
            # Color based on height and intensity
            # Red channel: Height (normalized)
            height_normalized = np.clip((height + 2.0) / 4.0, 0, 1)  # -2 to 2 meters
            # Green channel: Intensity (normalized)
            intensity_normalized = np.clip(intensity / 255.0, 0, 1)
            # Blue channel: Density
            
            bev_image[bev_x, bev_y, 0] = int(height_normalized * 255)
            bev_image[bev_x, bev_y, 1] = int(intensity_normalized * 255)
        
        # Add density information to blue channel
        density_normalized = np.clip(density_map / np.max(density_map + 1e-6), 0, 1)
        bev_image[:, :, 2] = (density_normalized * 255).astype(np.uint8)
        
        return bev_image, height_map
    
    def camera_to_bev(self, image, camera_pose=None):
        """
        Project camera image to BEV (simplified implementation)
        
        Args:
            image: Camera image
            camera_pose: Camera pose information (optional)
            
        Returns:
            bev_projection: Camera image projected to BEV space
        """
        # This is a simplified implementation
        # In practice, you'd need proper camera calibration and ground plane estimation
        
        height, width = image.shape[:2]
        bev_projection = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # Simple perspective transformation (rough approximation)
        # This assumes camera is looking forward and down at the ground
        
        # Define source points (in image) and destination points (in BEV)
        # This is very simplified - real implementation needs proper calibration
        src_points = np.float32([
            [width * 0.2, height * 0.6],   # Bottom left
            [width * 0.8, height * 0.6],   # Bottom right
            [width * 0.4, height * 0.4],   # Top left
            [width * 0.6, height * 0.4]    # Top right
        ])
        
        # Corresponding points in BEV (meters converted to pixels)
        dst_points = np.float32([
            [self.bev_height * 0.7, self.bev_width * 0.3],   # Bottom left in BEV
            [self.bev_height * 0.7, self.bev_width * 0.7],   # Bottom right in BEV
            [self.bev_height * 0.9, self.bev_width * 0.4],   # Top left in BEV
            [self.bev_height * 0.9, self.bev_width * 0.6]    # Top right in BEV
        ])
        
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        bev_projection = cv2.warpPerspective(image, M, (self.bev_width, self.bev_height))
        
        return bev_projection
    
    def create_multi_modal_bev(self, lidar_points, camera_image=None):
        """
        Create multi-modal BEV combining LiDAR and camera data
        
        Args:
            lidar_points: LiDAR point cloud
            camera_image: Camera image (optional)
            
        Returns:
            combined_bev: Multi-modal BEV representation
            lidar_bev: LiDAR-only BEV
            camera_bev: Camera-only BEV (if provided)
        """
        print("🔄 Creating multi-modal BEV...")
        
        # Generate LiDAR BEV
        lidar_bev, height_map = self.lidar_to_bev(lidar_points)
        print(f"   LiDAR BEV: {lidar_bev.shape}")
        
        camera_bev = None
        if camera_image is not None:
            # Generate camera BEV
            camera_bev = self.camera_to_bev(camera_image)
            print(f"   Camera BEV: {camera_bev.shape}")
            
            # Combine LiDAR and camera BEV
            # Weighted combination: LiDAR for structure, camera for texture
            alpha = 0.7  # Weight for LiDAR
            beta = 0.3   # Weight for camera
            
            combined_bev = cv2.addWeighted(lidar_bev, alpha, camera_bev, beta, 0)
        else:
            combined_bev = lidar_bev
        
        print("✅ Multi-modal BEV created!")
        
        return combined_bev, lidar_bev, camera_bev
    
    def add_ego_vehicle_to_bev(self, bev_image):
        """
        Add ego vehicle representation to BEV
        """
        result_bev = bev_image.copy()
        
        # Ego vehicle position (center bottom of BEV)
        ego_x = int(self.bev_height * 0.1)  # Near bottom (close to ego)
        ego_y = int(self.bev_width * 0.5)   # Center
        
        # Draw ego vehicle as rectangle (approximate car size: 4m x 2m)
        car_length_pixels = int(4.0 / self.resolution)
        car_width_pixels = int(2.0 / self.resolution)
        
        # Draw ego vehicle
        cv2.rectangle(
            result_bev,
            (ego_y - car_width_pixels//2, ego_x - car_length_pixels//2),
            (ego_y + car_width_pixels//2, ego_x + car_length_pixels//2),
            (255, 255, 255),  # White color
            -1
        )
        
        # Add arrow showing forward direction
        arrow_end = (ego_y, max(0, ego_x - car_length_pixels))
        cv2.arrowedLine(result_bev, (ego_y, ego_x), arrow_end, (0, 255, 0), 3)
        
        return result_bev
    
    def visualize_bev(self, bev_image, title="Bird's Eye View", save_path=None):
        """
        Visualize BEV with proper scale and labels
        """
        plt.figure(figsize=(12, 12))
        
        # Display BEV
        plt.imshow(bev_image, origin='lower')
        
        # Add grid and labels
        x_ticks = np.arange(0, self.bev_width, int(10 / self.resolution))  # Every 10 meters
        y_ticks = np.arange(0, self.bev_height, int(10 / self.resolution))
        
        x_labels = [f"{self.y_min + x * self.resolution:.0f}m" for x in x_ticks]
        y_labels = [f"{self.x_max - y * self.resolution:.0f}m" for y in y_ticks]
        
        plt.xticks(x_ticks, x_labels)
        plt.yticks(y_ticks, y_labels)
        
        plt.xlabel('Y (Left-Right) [meters]')
        plt.ylabel('X (Forward-Backward) [meters]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 BEV visualization saved: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()

def test_bev_generator():
    """
    Test BEV generator with real nuScenes data
    """
    print("🧪 Testing BEV Generator...")
    
    # Import required modules
    import sys, os
    sys.path.insert(0, os.getcwd())
    
    try:
        from lidar_processor import LiDARProcessor
    except ImportError:
        print("❌ Cannot import LiDARProcessor")
        return
    
    # Initialize components
    bev_gen = BEVGenerator(bev_range=(-50, 50, -50, 50), bev_resolution=0.1)
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
    
    print(f"📊 Testing with:")
    print(f"  📷 {camera_files[0].name}")
    print(f"  📡 {lidar_files[0].name}")
    
    # Load data
    camera_image = cv2.imread(str(camera_files[0]))
    lidar_points = lidar_proc.load_point_cloud(lidar_files[0])
    
    if camera_image is None or lidar_points is None:
        print("❌ Failed to load data")
        return
    
    # Filter LiDAR
    filtered_lidar = lidar_proc.filter_points(lidar_points)
    print(f"📡 LiDAR: {len(lidar_points)} → {len(filtered_lidar)} points")
    
    # Generate BEV
    combined_bev, lidar_bev, camera_bev = bev_gen.create_multi_modal_bev(
        filtered_lidar, camera_image
    )
    
    # Add ego vehicle
    final_bev = bev_gen.add_ego_vehicle_to_bev(combined_bev)
    
    # Save results
    results_dir = Path("../../results/bev_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save BEV images
    cv2.imwrite(str(results_dir / "bev_lidar.jpg"), lidar_bev)
    cv2.imwrite(str(results_dir / "bev_combined.jpg"), final_bev)
    
    if camera_bev is not None:
        cv2.imwrite(str(results_dir / "bev_camera.jpg"), camera_bev)
    
    # Create visualization
    fig = bev_gen.visualize_bev(final_bev, "Multi-Modal BEV", results_dir / "bev_visualization.png")
    plt.show()
    
    print(f"✅ BEV generation test completed!")
    print(f"📁 Results saved: {results_dir}")
    print(f"🗺️ BEV size: {final_bev.shape}")

if __name__ == "__main__":
    test_bev_generator()