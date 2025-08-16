# -*- coding: utf-8 -*-
# src/models/bev_tracker.py

import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial.distance import cdist
import json

class BEVTracker:
    def __init__(self, max_disappeared=10, max_distance=5.0, min_confidence=0.3):
        """
        Multi-object tracker for BEV space
        
        Args:
            max_disappeared: Max frames an object can disappear
            max_distance: Max distance for association (meters)
            min_confidence: Minimum confidence for accepting detections
        """
        print("🎯 Initializing BEV Tracker...")
        
        # Tracking parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        
        # Tracking state
        self.next_object_id = 0
        self.tracked_objects = OrderedDict()  # {id: object_state}
        self.disappeared_counts = OrderedDict()  # {id: disappeared_frames}
        
        # Object state includes: position, velocity, size, class, confidence_history
        
        print(f"   Max disappeared frames: {max_disappeared}")
        print(f"   Max association distance: {max_distance}m")
        print(f"   Min confidence threshold: {min_confidence}")
        print("✅ BEV Tracker ready!")
    
    def _initialize_object_state(self, detection):
        """
        Initialize state for a new tracked object
        
        Args:
            detection: Detection dict with world coordinates
            
        Returns:
            object_state: Initial state dictionary
        """
        state = {
            'position': np.array([detection['center_x'], detection['center_y']]),
            'velocity': np.array([0.0, 0.0]),  # Will be estimated
            'size': np.array([detection['length'], detection['width']]),
            'class': detection['class'],
            'confidence': detection['confidence'],
            'confidence_history': [detection['confidence']],
            'position_history': [np.array([detection['center_x'], detection['center_y']])],
            'age': 0,
            'total_visible_count': 1,
            'consecutive_invisible_count': 0
        }
        return state
    
    def _predict_position(self, object_state):
        """
        Predict next position using simple constant velocity model
        
        Args:
            object_state: Current object state
            
        Returns:
            predicted_position: Predicted [x, y] position
        """
        current_pos = object_state['position']
        velocity = object_state['velocity']
        
        # Simple linear prediction
        predicted_pos = current_pos + velocity
        
        return predicted_pos
    
    def _update_velocity(self, object_state, new_position):
        """
        Update velocity estimate using position history
        """
        position_history = object_state['position_history']
        
        if len(position_history) >= 2:
            # Calculate velocity as difference between last two positions
            velocity = new_position - position_history[-1]
            
            # Smooth velocity with previous estimate (simple low-pass filter)
            alpha = 0.3  # Smoothing factor
            object_state['velocity'] = alpha * velocity + (1 - alpha) * object_state['velocity']
        
        # Update position history
        position_history.append(new_position.copy())
        
        # Keep only recent history (last 10 positions)
        if len(position_history) > 10:
            position_history.pop(0)
    
    def _calculate_association_cost(self, tracked_objects, detections):
        """
        Calculate cost matrix for object-detection association
        
        Args:
            tracked_objects: List of tracked object states
            detections: List of detection dictionaries
            
        Returns:
            cost_matrix: (N_tracked, N_detections) cost matrix
        """
        if len(tracked_objects) == 0 or len(detections) == 0:
            return np.array([])
        
        # Get predicted positions for tracked objects
        predicted_positions = []
        for obj_state in tracked_objects:
            pred_pos = self._predict_position(obj_state)
            predicted_positions.append(pred_pos)
        
        predicted_positions = np.array(predicted_positions)
        
        # Get detection positions
        detection_positions = []
        for detection in detections:
            detection_positions.append([detection['center_x'], detection['center_y']])
        
        detection_positions = np.array(detection_positions)
        
        # Calculate distance matrix
        distance_matrix = cdist(predicted_positions, detection_positions)
        
        # Create cost matrix (distance + confidence penalty)
        cost_matrix = distance_matrix.copy()
        
        # Add confidence penalty (lower confidence = higher cost)
        for j, detection in enumerate(detections):
            confidence_penalty = (1.0 - detection['confidence']) * 2.0
            cost_matrix[:, j] += confidence_penalty
        
        return cost_matrix
    
    def _associate_detections(self, detections):
        """
        Associate detections with tracked objects
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            matches: List of (object_id, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_objects: List of object IDs
        """
        if len(self.tracked_objects) == 0:
            return [], list(range(len(detections))), []
        
        # Get current tracked objects
        object_ids = list(self.tracked_objects.keys())
        object_states = [self.tracked_objects[obj_id] for obj_id in object_ids]
        
        # Calculate cost matrix
        cost_matrix = self._calculate_association_cost(object_states, detections)
        
        if cost_matrix.size == 0:
            return [], list(range(len(detections))), object_ids
        
        # Simple greedy assignment (could use Hungarian algorithm for optimal)
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_objects = object_ids.copy()
        
        # Sort by minimum cost
        while cost_matrix.size > 0 and len(unmatched_detections) > 0 and len(unmatched_objects) > 0:
            # Find minimum cost
            min_cost_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            min_cost = cost_matrix[min_cost_idx]
            
            # Check if cost is acceptable
            if min_cost > self.max_distance:
                break
            
            # Make assignment
            obj_idx, det_idx = min_cost_idx
            object_id = unmatched_objects[obj_idx]
            detection_idx = unmatched_detections[det_idx]
            
            matches.append((object_id, detection_idx))
            
            # Remove from unmatched lists
            unmatched_objects.remove(object_id)
            unmatched_detections.remove(detection_idx)
            
            # Remove row and column from cost matrix
            cost_matrix = np.delete(cost_matrix, obj_idx, axis=0)
            cost_matrix = np.delete(cost_matrix, det_idx, axis=1)
        
        return matches, unmatched_detections, unmatched_objects
    
    def update(self, detections, frame_id=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with world coordinates
            frame_id: Optional frame identifier
            
        Returns:
            tracked_objects: Current tracked objects with IDs
        """
        # Filter detections by confidence
        valid_detections = [det for det in detections if det['confidence'] >= self.min_confidence]
        
        print(f"🔄 Tracking update: {len(valid_detections)}/{len(detections)} valid detections")
        
        # Associate detections with existing tracks
        matches, unmatched_detections, unmatched_objects = self._associate_detections(valid_detections)
        
        # Update matched objects
        for object_id, detection_idx in matches:
            detection = valid_detections[detection_idx]
            object_state = self.tracked_objects[object_id]
            
            # Update position and velocity
            new_position = np.array([detection['center_x'], detection['center_y']])
            object_state['position'] = new_position
            self._update_velocity(object_state, new_position)
            
            # Update other attributes
            object_state['size'] = np.array([detection['length'], detection['width']])
            object_state['confidence'] = detection['confidence']
            object_state['confidence_history'].append(detection['confidence'])
            
            # Update counters
            object_state['age'] += 1
            object_state['total_visible_count'] += 1
            object_state['consecutive_invisible_count'] = 0
            
            # Reset disappeared count
            self.disappeared_counts[object_id] = 0
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = valid_detections[detection_idx]
            
            # Initialize new object
            object_state = self._initialize_object_state(detection)
            
            self.tracked_objects[self.next_object_id] = object_state
            self.disappeared_counts[self.next_object_id] = 0
            
            print(f"📝 New object ID {self.next_object_id}: {detection['class']} at ({detection['center_x']:.1f}, {detection['center_y']:.1f})")
            
            self.next_object_id += 1
        
        # Update unmatched objects (mark as disappeared)
        for object_id in unmatched_objects:
            self.disappeared_counts[object_id] += 1
            self.tracked_objects[object_id]['consecutive_invisible_count'] += 1
            self.tracked_objects[object_id]['age'] += 1
        
        # Remove objects that have been disappeared too long
        objects_to_remove = []
        for object_id, disappeared_count in self.disappeared_counts.items():
            if disappeared_count > self.max_disappeared:
                objects_to_remove.append(object_id)
        
        for object_id in objects_to_remove:
            print(f"🗑️ Removing object ID {object_id} (disappeared {self.disappeared_counts[object_id]} frames)")
            del self.tracked_objects[object_id]
            del self.disappeared_counts[object_id]
        
        # Return current tracked objects
        return self._get_current_tracked_objects()
    
    def _get_current_tracked_objects(self):
        """
        Get current tracked objects in a standardized format
        """
        tracked_objects = {}
        
        for object_id, state in self.tracked_objects.items():
            # Only return visible objects (recently seen)
            if self.disappeared_counts[object_id] <= 3:  # Allow short temporary disappearances
                tracked_objects[object_id] = {
                    'position': state['position'].copy(),
                    'velocity': state['velocity'].copy(),
                    'size': state['size'].copy(),
                    'class': state['class'],
                    'confidence': state['confidence'],
                    'age': state['age'],
                    'visible_count': state['total_visible_count'],
                    'is_confirmed': state['total_visible_count'] >= 3  # Confirmed after 3 detections
                }
        
        return tracked_objects
    
    def get_trajectories(self, min_length=5):
        """
        Get trajectories of tracked objects
        
        Args:
            min_length: Minimum trajectory length to return
            
        Returns:
            trajectories: Dict of object trajectories
        """
        trajectories = {}
        
        for object_id, state in self.tracked_objects.items():
            position_history = state['position_history']
            
            if len(position_history) >= min_length:
                trajectories[object_id] = {
                    'positions': np.array(position_history),
                    'class': state['class'],
                    'age': state['age']
                }
        
        return trajectories
    
    def visualize_tracking(self, bev_image, tracked_objects=None, show_trajectories=True):
        """
        Visualize tracking results on BEV image
        
        Args:
            bev_image: BEV image
            tracked_objects: Optional tracked objects (uses current if None)
            show_trajectories: Whether to show trajectory history
            
        Returns:
            result_image: Image with tracking visualization
        """
        if tracked_objects is None:
            tracked_objects = self._get_current_tracked_objects()
        
        result_image = bev_image.copy()
        
        # Color palette for different object IDs
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        # BEV parameters (should match BEVGenerator)
        bev_range = (-50, 50, -50, 50)
        resolution = 0.1
        x_min, x_max, y_min, y_max = bev_range
        bev_height, bev_width = bev_image.shape[:2]
        
        for object_id, obj_info in tracked_objects.items():
            color = colors[object_id % len(colors)]
            
            # Convert world coordinates to BEV image coordinates
            world_x, world_y = obj_info['position']
            bev_x = int((x_max - world_x) / resolution)
            bev_y = int((world_y - y_min) / resolution)
            
            # Check if position is within image bounds
            if 0 <= bev_x < bev_height and 0 <= bev_y < bev_width:
                # Draw object center
                cv2.circle(result_image, (bev_y, bev_x), 5, color, -1)
                
                # Draw object ID and class
                label = f"ID:{object_id} {obj_info['class']}"
                if obj_info['is_confirmed']:
                    label += " ✓"
                
                cv2.putText(result_image, label, (bev_y + 10, bev_x - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw velocity vector
                vel_x, vel_y = obj_info['velocity']
                if np.linalg.norm([vel_x, vel_y]) > 0.1:  # Only draw if moving
                    end_x = int(bev_x - vel_x * 10 / resolution)  # Scale for visualization
                    end_y = int(bev_y + vel_y * 10 / resolution)
                    
                    if 0 <= end_x < bev_height and 0 <= end_y < bev_width:
                        cv2.arrowedLine(result_image, (bev_y, bev_x), (end_y, end_x), color, 2)
                
                # Draw trajectory if requested
                if show_trajectories and object_id in self.tracked_objects:
                    position_history = self.tracked_objects[object_id]['position_history']
                    
                    if len(position_history) > 1:
                        trajectory_points = []
                        
                        for pos in position_history[-10:]:  # Last 10 positions
                            traj_world_x, traj_world_y = pos
                            traj_bev_x = int((x_max - traj_world_x) / resolution)
                            traj_bev_y = int((traj_world_y - y_min) / resolution)
                            
                            if 0 <= traj_bev_x < bev_height and 0 <= traj_bev_y < bev_width:
                                trajectory_points.append((traj_bev_y, traj_bev_x))
                        
                        # Draw trajectory line
                        if len(trajectory_points) > 1:
                            trajectory_points = np.array(trajectory_points, dtype=np.int32)
                            cv2.polylines(result_image, [trajectory_points], False, color, 1)
        
        # Add tracking statistics
        total_objects = len(tracked_objects)
        confirmed_objects = sum(1 for obj in tracked_objects.values() if obj['is_confirmed'])
        
        stats_text = f"Tracked: {total_objects} | Confirmed: {confirmed_objects}"
        cv2.putText(result_image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image

def test_bev_tracker():
    """
    Test BEV tracker with simulated sequence
    """
    print("🧪 Testing BEV Tracker...")
    
    tracker = BEVTracker(max_disappeared=5, max_distance=3.0, min_confidence=0.5)
    
    # Simulate object detections across multiple frames
    test_sequence = [
        # Frame 1: Two cars
        [
            {'center_x': 10, 'center_y': 5, 'length': 4.5, 'width': 2.0, 'confidence': 0.8, 'class': 'car'},
            {'center_x': 15, 'center_y': -3, 'length': 4.2, 'width': 1.9, 'confidence': 0.7, 'class': 'car'}
        ],
        # Frame 2: Cars moved, one person appears
        [
            {'center_x': 9.5, 'center_y': 5.2, 'length': 4.5, 'width': 2.0, 'confidence': 0.9, 'class': 'car'},
            {'center_x': 14.8, 'center_y': -2.8, 'length': 4.2, 'width': 1.9, 'confidence': 0.6, 'class': 'car'},
            {'center_x': 8, 'center_y': 0, 'length': 0.8, 'width': 0.8, 'confidence': 0.8, 'class': 'person'}
        ],
        # Frame 3: One car disappears temporarily
        [
            {'center_x': 9.0, 'center_y': 5.5, 'length': 4.5, 'width': 2.0, 'confidence': 0.85, 'class': 'car'},
            {'center_x': 7.8, 'center_y': 0.2, 'length': 0.8, 'width': 0.8, 'confidence': 0.9, 'class': 'person'}
        ],
        # Frame 4: Car reappears, new truck
        [
            {'center_x': 8.5, 'center_y': 5.8, 'length': 4.5, 'width': 2.0, 'confidence': 0.8, 'class': 'car'},
            {'center_x': 14.5, 'center_y': -2.5, 'length': 4.2, 'width': 1.9, 'confidence': 0.7, 'class': 'car'},
            {'center_x': 7.6, 'center_y': 0.5, 'length': 0.8, 'width': 0.8, 'confidence': 0.85, 'class': 'person'},
            {'center_x': 20, 'center_y': 8, 'length': 8.0, 'width': 2.5, 'confidence': 0.9, 'class': 'truck'}
        ]
    ]
    
    # Process sequence
    for frame_id, detections in enumerate(test_sequence):
        print(f"\n--- Frame {frame_id + 1} ---")
        print(f"Input detections: {len(detections)}")
        
        tracked_objects = tracker.update(detections, frame_id)
        
        print(f"Tracked objects: {len(tracked_objects)}")
        for obj_id, obj_info in tracked_objects.items():
            pos = obj_info['position']
            vel = obj_info['velocity']
            print(f"  ID {obj_id}: {obj_info['class']} at ({pos[0]:.1f}, {pos[1]:.1f}) "
                  f"vel ({vel[0]:.1f}, {vel[1]:.1f}) conf: {obj_info['confidence']:.2f}")
    
    # Get final trajectories
    trajectories = tracker.get_trajectories(min_length=2)
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total objects ever tracked: {tracker.next_object_id}")
    print(f"   Currently active: {len(tracked_objects)}")
    print(f"   Objects with trajectories: {len(trajectories)}")
    
    print(f"\n✅ BEV tracker test completed!")

if __name__ == "__main__":
    test_bev_tracker()