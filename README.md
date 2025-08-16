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
- **Open3D** - 3D point cloud processing
- **NumPy/SciPy** - Scientific computing
- **Matplotlib** - Visualization and analysis

## Project Structure

```
urban-multi-object-tracking/
├── src/
│   ├── models/          # Core detection and tracking models
│   ├── data/           # Data loading and preprocessing
│   └── utils/          # Utility functions and helpers
├── data/               # nuScenes dataset
├── results/            # Output visualizations and metrics
├── notebooks/          # Analysis and experimentation
└── experiments/        # Performance evaluation
```

## Installation and Usage

```bash
# Clone repository
git clone https://github.com/your-username/urban-multi-object-tracking.git
cd urban-multi-object-tracking

# Install dependencies
pip install -r requirements.txt

# Download nuScenes mini dataset
python download_nuscenes.py

# Run basic detection test
python test_detection.py

# Run full fusion pipeline
python test_full_fusion.py
```

## Results

The system demonstrates significant improvements in object detection and tracking accuracy through multi-modal sensor fusion, particularly in challenging urban scenarios with occlusions and dense traffic.

## Development Status

- [x] Basic camera object detection (YOLO)
- [x] LiDAR point cloud processing
- [x] Camera-LiDAR sensor fusion
- [x] Simple object tracking
- [ ] Bird's Eye View (BEV) generation
- [ ] BEV multi-object detection
- [ ] Advanced tracking algorithms
- [ ] Performance optimization
- [ ] Real-time deployment

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- nuScenes dataset by Motional
- YOLO by Ultralytics
- Open3D community
