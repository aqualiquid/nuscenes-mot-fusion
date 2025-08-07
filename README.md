# Urban Multi-Object Tracking

A comprehensive multi-modal object detection and tracking system for urban autonomous driving scenarios using the nuScenes dataset.

## Project Overview

This project implements an end-to-end pipeline for multi-object tracking in urban environments, combining camera and LiDAR sensor data to achieve robust object detection and tracking. The system progresses from basic 2D detection to advanced Bird's Eye View (BEV) multi-modal fusion.

### Key Features

- **Multi-Modal Sensor Fusion**: Combines camera images and LiDAR point clouds for enhanced object detection
- **Bird's Eye View (BEV) Representation**: Projects 3D sensor data into unified BEV space for better spatial understanding
- **Real-Time Object Tracking**: Maintains consistent object identities across time using advanced tracking algorithms
- **Urban Scenario Optimization**: Specifically designed for complex urban driving scenarios with multiple vehicles, pedestrians, and traffic infrastructure

### Technical Stack

- **Dataset**: nuScenes (1000 scenes, 40K keyframes)
- **Sensors**: 6 cameras + LiDAR + Radar
- **Detection**: YOLOv8 + Custom LiDAR processing
- **Fusion**: Camera-LiDAR calibration and projection
- **Tracking**: Kalman filtering + Hungarian algorithm
- **Visualization**: BEV maps + 3D point clouds

## System Architecture

```
Camera Images ──┐
                ├─► Sensor Fusion ─► BEV Generation ─► Object Detection ─► Multi-Object Tracking
LiDAR Points ───┘
```

### Pipeline Stages

1. **Data Loading**: nuScenes dataset preprocessing and synchronization
2. **Sensor Processing**: Individual camera and LiDAR object detection
3. **Multi-Modal Fusion**: Coordinate transformation and sensor data alignment  
4. **BEV Generation**: Unified bird's eye view representation
5. **Object Tracking**: Temporal consistency and ID management
6. **Visualization**: Real-time tracking results and performance metrics

## Performance Highlights

- **Detection Accuracy**: Enhanced precision through sensor fusion
- **Tracking Consistency**: Reduced ID switches in complex urban scenarios
- **Real-Time Processing**: Optimized for autonomous driving applications
- **Robustness**: Handles occlusions, weather variations, and dense traffic

## Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision processing
- **Open3D** - 3
