# -*- coding: utf-8 -*-
# src/simple_detection.py

import cv2
import torch
from ultralytics import YOLO
import numpy as np

class SimpleDetector:
    """
    Custom wrapper class for YOLO object detection
    This is NOT a built-in class - it's a custom implementation for this project
    """
    def __init__(self):
        """
        Initialize simple object detector using YOLO model
        """
        # Load YOLOv8 model (will download automatically on first run)
        self.model = YOLO('yolov8n.pt')  # nano version (lightweight)
        
        # Target classes for urban scenarios
        self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        
    def detect_objects(self, frame):
        """
        Detect objects in a single frame
        
        Args:
            frame: OpenCV image (numpy array)
            
        Returns:
            detections: List of detected objects
                       Each object: [x1, y1, x2, y2, confidence, class_name]
        """
        # Run YOLO detection
        results = self.model(frame)
        
        detections = []
        
        # Parse results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Filter only target classes with high confidence
                    if class_name in self.target_classes and confidence > 0.5:
                        detections.append([
                            int(x1), int(y1), int(x2), int(y2), 
                            float(confidence), class_name
                        ])
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection results on frame
        """
        for detection in detections:
            x1, y1, x2, y2, conf, class_name = detection
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label text
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def test_webcam():
    """
    Test real-time object detection with webcam
    """
    detector = SimpleDetector()
    cap = cv2.VideoCapture(0)
    
    print("Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break
            
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Draw results
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Show detection count
        cv2.putText(frame_with_detections, f"Objects detected: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Object Detection', frame_with_detections)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_video(video_path):
    """
    Test object detection on video file
    """
    detector = SimpleDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit, SPACE to pause")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
            
        frame_count += 1
        
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Print detection info every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(detections)} objects detected")
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, class_name = det
                print(f"  {i+1}. {class_name} (confidence: {conf:.2f})")
        
        # Draw results
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Add frame info
        cv2.putText(frame_with_detections, f"Frame: {frame_count}, Objects: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize for display if too large
        height, width = frame_with_detections.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_with_detections = cv2.resize(frame_with_detections, (new_width, new_height))
        
        # Display frame
        cv2.imshow('Video Object Detection', frame_with_detections)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause on spacebar
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing completed. Total frames: {frame_count}")

def test_image(image_path):
    """
    Test object detection on single image
    """
    detector = SimpleDetector()
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Detect objects
    detections = detector.detect_objects(frame)
    
    # Print results
    print(f"Objects detected: {len(detections)}")
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, class_name = det
        print(f"  {i+1}. {class_name} (confidence: {conf:.2f}) at ({x1},{y1})-({x2},{y2})")
    
    # Draw results
    frame_with_detections = detector.draw_detections(frame, detections)
    
    # Display image
    cv2.imshow('Image Object Detection', frame_with_detections)
    print("Press any key to close the image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_direct_yolo():
    """
    Test using YOLO directly without wrapper class
    This shows what happens 'under the hood' of SimpleDetector
    """
    print("Testing direct YOLO usage...")
    
    # Load YOLO model directly
    model = YOLO('yolov8n.pt')
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Direct YOLO inference
        results = model(frame)
        
        # Process results manually
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    if confidence > 0.5:
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('Direct YOLO', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":ㅌ
    print("=== Object Detection Test ===")
    print()
    print("Choose test mode:")
    print("1. Webcam with SimpleDetector (custom wrapper)")
    print("2. Video file with SimpleDetector")
    print("3. Image file with SimpleDetector") 
    print("4. Direct YOLO usage (no wrapper)")
    
    choice = input("Enter choice (1/2/3/4): ")
    
    if choice == "1":
        test_webcam()
    elif choice == "2":
        video_path = input("Enter video file path: ")
        test_video(video_path)
    elif choice == "3":
        image_path = input("Enter image file path: ")
        test_image(image_path)
    elif choice == "4":
        test_direct_yolo()
    else:
        print("Invalid choice. Running webcam test by default.")
        test_webcam()