# debug_bev.py
print("🚀 Starting BEV debug...")

try:
    print("1. Testing basic imports...")
    import numpy as np
    print("   ✅ numpy OK")
    
    import cv2
    print("   ✅ cv2 OK")
    
    from pathlib import Path
    print("   ✅ pathlib OK")
    
    import matplotlib.pyplot as plt
    print("   ✅ matplotlib OK")
    
    print("2. Testing BEV Generator import...")
    from bev_generator import BEVGenerator
    print("   ✅ BEVGenerator import OK")
    
    print("3. Creating BEV Generator...")
    bev_gen = BEVGenerator()
    print("   ✅ BEVGenerator creation OK")
    
    print("4. Testing data directory...")
    data_dir = Path("../../data/raw/samples")
    print(f"   Data dir: {data_dir}")
    print(f"   Exists: {data_dir.exists()}")
    
    if data_dir.exists():
        camera_dir = data_dir / "CAM_FRONT"
        lidar_dir = data_dir / "LIDAR_TOP"
        
        print(f"   Camera dir: {camera_dir.exists()}")
        print(f"   LiDAR dir: {lidar_dir.exists()}")
        
        if camera_dir.exists():
            camera_files = list(camera_dir.glob("*.jpg"))
            print(f"   Camera files: {len(camera_files)}")
        
        if lidar_dir.exists():
            lidar_files = list(lidar_dir.glob("*.pcd.bin"))
            print(f"   LiDAR files: {len(lidar_files)}")
    
    print("5. Testing other imports...")
    try:
        from lidar_processor import LiDARProcessor
        print("   ✅ LiDARProcessor import OK")
    except ImportError as e:
        print(f"   ❌ LiDARProcessor import failed: {e}")
    
    print("\n✅ Debug completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Debug finished.")