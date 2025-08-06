# -*- coding: utf-8 -*-
# src/models/lidar_processor.py

import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
import os

class LiDARProcessor:
    def __init__(self):
        """
        LiDAR point cloud processor
        """
        print("📡 Initializing LiDAR Processor...")
        
        # LiDAR parameters for nuScenes
        self.min_distance = 1.0    # 최소 거리 (m)
        self.max_distance = 50.0   # 최대 거리 (m)
        self.min_height = -3.0     # 최소 높이 (m)
        self.max_height = 2.0      # 최대 높이 (m)
        
        print("✅ LiDAR Processor ready!")
        print(f"   Distance range: {self.min_distance}m - {self.max_distance}m")
        print(f"   Height range: {self.min_height}m - {self.max_height}m")
    
    def load_point_cloud(self, lidar_file):
        """
        Load LiDAR point cloud from .pcd.bin file
        
        Args:
            lidar_file: Path to .pcd.bin file
            
        Returns:
            points: numpy array of shape (N, 4) [x, y, z, intensity]
        """
        if not Path(lidar_file).exists():
            print(f"❌ LiDAR file not found: {lidar_file}")
            return None
        
        # Read binary point cloud file
        points = np.fromfile(str(lidar_file), dtype=np.float32)
        
        # Reshape to (N, 5) - [x, y, z, intensity, ring_index]
        # nuScenes format has 5 values per point
        if len(points) % 5 != 0:
            print(f"⚠️  Point cloud has unexpected format: {len(points)} values")
            # Try 4 values per point format
            if len(points) % 4 == 0:
                points = points.reshape(-1, 4)
                print(f"📊 Loaded {len(points)} points (4D format)")
                return points
            else:
                print(f"❌ Cannot parse point cloud format")
                return None
        
        points = points.reshape(-1, 5)
        # Use only [x, y, z, intensity]
        points = points[:, :4]
        
        print(f"📊 Loaded {len(points)} LiDAR points")
        return points
    
    def filter_points(self, points):
        """
        Filter point cloud by distance and height
        """
        if points is None:
            return None
        
        # Calculate distance from origin
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Apply filters
        mask = (
            (distances >= self.min_distance) & 
            (distances <= self.max_distance) &
            (points[:, 2] >= self.min_height) & 
            (points[:, 2] <= self.max_height)
        )
        
        filtered_points = points[mask]
        
        print(f"🔍 Filtered: {len(points)} → {len(filtered_points)} points")
        print(f"   Removed {len(points) - len(filtered_points)} points")
        
        return filtered_points
    
    def visualize_2d_bev(self, points, save_path=None):
        """
        Create 2D Bird's Eye View visualization
        
        Args:
            points: Point cloud array
            save_path: Optional path to save image
        """
        if points is None:
            return None
        
        print("📊 Creating Bird's Eye View...")
        
        # Extract x, y coordinates
        x = points[:, 0]
        y = points[:, 1]
        intensity = points[:, 3] if points.shape[1] > 3 else np.ones(len(points))
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Scatter plot colored by intensity
        scatter = plt.scatter(x, y, c=intensity, s=0.5, cmap='viridis', alpha=0.6)
        
        # Add colorbar
        plt.colorbar(scatter, label='Intensity')
        
        # Labels and formatting
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('LiDAR Bird\'s Eye View')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add distance circles
        for radius in [10, 20, 30, 40, 50]:
            circle = plt.Circle((0, 0), radius, fill=False, color='red', alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)
            plt.text(radius, 0, f'{radius}m', color='red', fontsize=8)
        
        # Mark origin (vehicle position)
        plt.plot(0, 0, 'ro', markersize=10, label='Vehicle')
        plt.legend()
        
        # Save if requested
        if save_path:
            # 폴더가 없으면 생성
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 BEV saved: {save_path}")
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_3d(self, points, save_path=None):
        """
        Create 3D visualization using Open3D
        """
        if points is None:
            return None
        
        print("🎯 Creating 3D visualization...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Color by height (z-coordinate)
        z_values = points[:, 2]
        z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        
        # Create color map (blue=low, red=high)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_normalized  # Red channel
        colors[:, 2] = 1 - z_normalized  # Blue channel
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY file if requested
        if save_path:
            # 폴더가 없으면 생성
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            ply_path = Path(save_path).with_suffix('.ply')
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"💾 3D point cloud saved: {ply_path}")
        
        # Display (optional - might not work in all environments)
        try:
            print("🎨 Displaying 3D point cloud...")
            print("   (Close the window to continue)")
            o3d.visualization.draw_geometries([pcd])
        except:
            print("⚠️  3D display not available in this environment")
        
        return pcd
    
    def get_point_statistics(self, points):
        """
        Get statistics about the point cloud
        """
        if points is None:
            return {}
        
        stats = {
            'total_points': len(points),
            'x_range': (points[:, 0].min(), points[:, 0].max()),
            'y_range': (points[:, 1].min(), points[:, 1].max()),
            'z_range': (points[:, 2].min(), points[:, 2].max()),
            'distances': np.sqrt(points[:, 0]**2 + points[:, 1]**2),
        }
        
        stats['distance_range'] = (stats['distances'].min(), stats['distances'].max())
        stats['mean_distance'] = stats['distances'].mean()
        
        if points.shape[1] > 3:
            stats['intensity_range'] = (points[:, 3].min(), points[:, 3].max())
            stats['mean_intensity'] = points[:, 3].mean()
        
        return stats
    
    def print_statistics(self, points):
        """
        Print point cloud statistics
        """
        stats = self.get_point_statistics(points)
        
        if not stats:
            print("❌ No statistics available")
            return
        
        print("\n📊 Point Cloud Statistics:")
        print(f"   Total points: {stats['total_points']:,}")
        print(f"   X range: {stats['x_range'][0]:.1f} to {stats['x_range'][1]:.1f} m")
        print(f"   Y range: {stats['y_range'][0]:.1f} to {stats['y_range'][1]:.1f} m") 
        print(f"   Z range: {stats['z_range'][0]:.1f} to {stats['z_range'][1]:.1f} m")
        print(f"   Distance range: {stats['distance_range'][0]:.1f} to {stats['distance_range'][1]:.1f} m")
        print(f"   Mean distance: {stats['mean_distance']:.1f} m")
        
        if 'intensity_range' in stats:
            print(f"   Intensity range: {stats['intensity_range'][0]:.1f} to {stats['intensity_range'][1]:.1f}")
            print(f"   Mean intensity: {stats['mean_intensity']:.1f}")

# Test function
def test_lidar_processor():
    """
    Test LiDAR processor with dummy data
    """
    print("🧪 Testing LiDAR Processor...")
    
    processor = LiDARProcessor()
    
    # Create dummy point cloud
    n_points = 10000
    dummy_points = np.random.randn(n_points, 4)
    dummy_points[:, 0] *= 20  # x
    dummy_points[:, 1] *= 20  # y  
    dummy_points[:, 2] *= 3   # z
    dummy_points[:, 3] = np.random.rand(n_points)  # intensity
    
    print(f"📊 Created dummy point cloud: {len(dummy_points)} points")
    
    # Filter points
    filtered_points = processor.filter_points(dummy_points)
    
    # Get statistics
    processor.print_statistics(filtered_points)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 2D Bird's Eye View
    processor.visualize_2d_bev(filtered_points, "results/test_lidar_bev.png")
    
    # 3D visualization
    processor.visualize_3d(filtered_points, "results/test_lidar_3d.ply")
    
    print("✅ LiDAR processor test completed!")

if __name__ == "__main__":
    test_lidar_processor()