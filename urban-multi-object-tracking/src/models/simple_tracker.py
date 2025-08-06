# -*- coding: utf-8 -*-
# src/models/simple_tracker.py

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from collections import OrderedDict

class SimpleTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Simple multi-object tracker
        
        Args:
            max_disappeared: Max frames an object can disappear before removal
            max_distance: Max distance to consider same object (pixels)
        """
        # Initialize tracking variables
        self.next_object_id = 0 
        self.objects = OrderedDict()  # {object_id: [center_x, center_y]}
        self.disappeared = OrderedDict()  # {object_id: disappeared_count}
        self.object_classes = OrderedDict()  # {object_id: class_name}
        
        # Parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        print(f"🎯 SimpleTracker initialized")
        print(f"   • Max disappeared frames: {max_disappeared}")
        print(f"   • Max tracking distance: {max_distance} pixels")
    
    def register(self, center, class_name):
        """
        Register a new object for tracking
        """
        self.objects[self.next_object_id] = center
        self.disappeared[self.next_object_id] = 0
        self.object_classes[self.next_object_id] = class_name
        
        print(f"📝 Registered new object ID: {self.next_object_id} ({class_name})")
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        Remove an object from tracking
        """
        if object_id in self.objects:
            class_name = self.object_classes.get(object_id, "unknown")
            print(f"🗑️ Deregistered object ID: {object_id} ({class_name})")
            
            del self.objects[object_id]
            del self.disappeared[object_id]
            del self.object_classes[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_name]
            
        Returns:
            tracked_objects: {object_id: [center_x, center_y, class_name]}
        """
        # If no detections, increment disappeared counter for all objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove objects that disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self._get_current_objects()
        
        # Convert detections to centers
        input_centers = []
        input_classes = []
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_name = detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            input_centers.append([center_x, center_y])
            input_classes.append(class_name)
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for i, center in enumerate(input_centers):
                self.register(center, input_classes[i])
        
        else:
            # Match existing objects with new detections
            self._match_objects(input_centers, input_classes)
        
        return self._get_current_objects()
    
    def _match_objects(self, input_centers, input_classes):
        """
        Match existing tracked objects with new detections
        """
        # Get current object centers and IDs
        object_centers = list(self.objects.values())
        object_ids = list(self.objects.keys())
        
        # Calculate distance matrix between existing objects and new detections
        distances = cdist(np.array(object_centers), np.array(input_centers))
        
        # Find the minimum distance assignments
        # Simple greedy assignment (could use Hungarian algorithm for better results)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        # Track which rows and columns we've already used
        used_row_indices = set()
        used_col_indices = set()
        
        # Update matched objects
        for (row, col) in zip(rows, cols):
            # Skip if already used
            if row in used_row_indices or col in used_col_indices:
                continue
            
            # Skip if distance is too large
            if distances[row, col] > self.max_distance:
                continue
            
            # Update object position
            object_id = object_ids[row]
            self.objects[object_id] = input_centers[col]
            self.disappeared[object_id] = 0
            
            # Update class (in case detection improved)
            self.object_classes[object_id] = input_classes[col]
            
            # Mark as used
            used_row_indices.add(row)
            used_col_indices.add(col)
        
        # Handle unmatched detections (new objects)
        unused_col_indices = set(range(0, len(input_centers))).difference(used_col_indices)
        for col in unused_col_indices:
            self.register(input_centers[col], input_classes[col])
        
        # Handle unmatched existing objects (disappeared)
        unused_row_indices = set(range(0, len(object_ids))).difference(used_row_indices)
        for row in unused_row_indices:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            
            # Remove if disappeared too long
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
    
    def _get_current_objects(self):
        """
        Get current tracked objects
        """
        tracked_objects = {}
        for object_id, center in self.objects.items():
            class_name = self.object_classes.get(object_id, "unknown")
            tracked_objects[object_id] = [center[0], center[1], class_name]
        
        return tracked_objects
    
    def draw_tracks(self, frame, tracked_objects):
        """
        Draw tracked objects on frame
        """
        # Color palette for different object IDs
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        result_frame = frame.copy()
        
        for object_id, (center_x, center_y, class_name) in tracked_objects.items():
            # Get color for this object ID
            color = colors[object_id % len(colors)]
            
            # Draw center point
            cv2.circle(result_frame, (center_x, center_y), 8, color, -1)
            
            # Draw object ID and class
            label = f"ID:{object_id} {class_name}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle
            cv2.rectangle(
                result_frame,
                (center_x - text_width//2 - 5, center_y - 25 - text_height),
                (center_x + text_width//2 + 5, center_y - 20),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result_frame,
                label,
                (center_x - text_width//2, center_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
        
        return result_frame
    
    def get_stats(self):
        """
        Get tracking statistics
        """
        total_objects = len(self.objects)
        active_objects = sum(1 for d in self.disappeared.values() if d == 0)
        disappeared_objects = total_objects - active_objects
        
        stats = {
            'total_objects': total_objects,
            'active_objects': active_objects,
            'disappeared_objects': disappeared_objects,
            'next_id': self.next_object_id
        }
        
        return stats

# Test function
def test_tracker():
    """
    Simple test for the tracker
    """
    print("🧪 Testing SimpleTracker...")
    
    tracker = SimpleTracker(max_disappeared=5, max_distance=50)
    
    # Simulate some detections
    test_detections = [
        # Frame 1: 2 cars
        [[100, 100, 150, 150, 0.9, 'car'], [300, 200, 350, 250, 0.8, 'car']],
        # Frame 2: Same cars moved slightly
        [[105, 105, 155, 155, 0.9, 'car'], [305, 205, 355, 255, 0.8, 'car']],
        # Frame 3: One car disappeared, one new car
        [[110, 110, 160, 160, 0.9, 'car'], [500, 300, 550, 350, 0.7, 'car']],
    ]
    
    for frame_num, detections in enumerate(test_detections, 1):
        print(f"\n--- Frame {frame_num} ---")
        tracked_objects = tracker.update(detections)
        
        print(f"Detections: {len(detections)}")
        print(f"Tracked objects: {len(tracked_objects)}")
        
        for obj_id, (x, y, class_name) in tracked_objects.items():
            print(f"  ID {obj_id}: {class_name} at ({x}, {y})")
        
        stats = tracker.get_stats()
        print(f"Stats: {stats}")
    
    print("\n✅ Tracker test completed!")

if __name__ == "__main__":
    test_tracker()