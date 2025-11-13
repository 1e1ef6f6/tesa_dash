import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# PyTorch 2.6+ compatibility: Add safe globals for YOLOv5/YOLOv8 models
try:
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except AttributeError:
    # Older PyTorch versions don't have this method
    pass

# ByteTrack imports
try:
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: boxmot not installed. Will use simple IoU tracking.")


class AdvancedTracker:

    def __init__(self, max_age=120, min_hits=1, iou_threshold=0.15, distance_threshold=200, 
                 appearance_weight=0.3, velocity_smooth=0.7):
        self.max_age = max_age  # Much longer for cloud occlusions (4 sec at 30fps)
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold  # Very lenient
        self.distance_threshold = distance_threshold  # Larger for post-occlusion
        self.appearance_weight = appearance_weight  # Weight for appearance matching
        self.velocity_smooth = velocity_smooth  # Velocity smoothing factor
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.frame_buffer = None  # Store current frame for appearance
        
    def update(self, detections, frame=None):
        self.frame_count += 1
        self.frame_buffer = frame
        
        if len(detections) == 0:
            # Predict positions for occluded tracks and age them
            self._age_tracks()
            return self._get_active_tracks()
        
        # Predict positions for all tracks based on velocity
        self._predict_positions()
        
        # Calculate matching scores between tracks and detections
        match_scores = self._calculate_match_scores(detections)
        
        # Greedy matching based on scores
        matched, unmatched_detections = self._match_greedy(match_scores, detections)
        
        # Update matched tracks
        self._update_matched_tracks(matched, detections)
        
        # Age unmatched tracks
        self._age_tracks(exclude=set(matched.keys()))
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(unmatched_detections, detections)
        
        # Remove very old tracks
        self._remove_dead_tracks()
        
        return self._get_active_tracks()
    
    def _predict_positions(self):
        """Predict next positions for all tracks based on velocity"""
        for track_id, track in self.tracks.items():
            if track['velocity'] is not None and track['age'] > 0:
                # Predict position during occlusion
                predicted_center = self._predict_center(track, track['age'])
                predicted_bbox = self._center_to_bbox(predicted_center, track['bbox'])
                track['predicted_bbox'] = predicted_bbox
            else:
                track['predicted_bbox'] = track['bbox']
    
    def _calculate_match_scores(self, detections):
        """Calculate matching scores between all tracks and detections"""
        match_scores = []
        
        for track_id, track in self.tracks.items():
            for det_idx in range(len(detections)):
                score = self._compute_similarity(track, detections[det_idx])
                match_scores.append((score, track_id, det_idx))
        
        # Sort by score (highest first)
        match_scores.sort(key=lambda x: x[0], reverse=True)
        return match_scores
    
    def _compute_similarity(self, track, detection):
        """
        Compute similarity between track and detection
        Uses: IoU, distance, motion prediction, and appearance
        """
        det_bbox = detection[:4]
        
        # 1. IoU score
        iou = self._calculate_iou(track['predicted_bbox'], det_bbox)
        
        # 2. Distance score
        distance = self._calculate_center_distance(track['predicted_bbox'], det_bbox)
        if distance < self.distance_threshold:
            distance_score = 1.0 - (distance / self.distance_threshold)
        else:
            distance_score = 0
        
        # 3. Motion prediction score
        motion_score = 0
        if track['velocity'] is not None:
            predicted_center = self._predict_center(track, track['age'] + 1)
            det_center = self._get_center(det_bbox)
            pred_distance = np.linalg.norm(np.array(predicted_center) - np.array(det_center))
            
            # Scale distance threshold based on age (longer occlusion = larger threshold)
            scaled_threshold = self.distance_threshold * (1 + track['age'] * 0.1)
            
            if pred_distance < scaled_threshold:
                motion_score = 0.5 * (1.0 - pred_distance / scaled_threshold)
        
        # 4. Appearance score (if frame available)
        appearance_score = 0
        if self.frame_buffer is not None and track['appearance'] is not None:
            appearance_score = self._compute_appearance_similarity(track, det_bbox)
        
        # Combine scores with adaptive weights
        if iou > 0.01:
            # Strong overlap: trust IoU most
            final_score = 0.6 * iou + 0.2 * distance_score + 0.1 * motion_score + 0.1 * appearance_score
        elif track['age'] == 0:
            # Recently seen: use distance and appearance
            final_score = 0.4 * distance_score + 0.3 * motion_score + 0.3 * appearance_score
        else:
            # Occluded: rely on prediction and appearance
            final_score = 0.3 * distance_score + 0.4 * motion_score + 0.3 * appearance_score
        
        # Boost score for tracks that were recently visible
        if track['age'] < 5:
            final_score *= 1.2
        
        return min(1.0, final_score)
    
    def _compute_appearance_similarity(self, track, det_bbox):
        """Compute appearance similarity using color histogram"""
        try:
            if self.frame_buffer is None:
                return 0
            
            # Extract detection region
            x1, y1, x2, y2 = [int(x) for x in det_bbox]
            h, w = self.frame_buffer.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0
            
            det_region = self.frame_buffer[y1:y2, x1:x2]
            
            if det_region.size == 0:
                return 0
            
            # Compute color histogram
            det_hist = self._compute_color_histogram(det_region)
            
            # Compare with stored appearance
            similarity = cv2.compareHist(track['appearance'], det_hist, cv2.HISTCMP_CORREL)
            return max(0, similarity)  # Correlation is in [-1, 1], clamp to [0, 1]
            
        except:
            return 0
    
    def _compute_color_histogram(self, region):
        """Compute color histogram for appearance matching"""
        # Use HSV for better color representation
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        
        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def _match_greedy(self, match_scores, detections):
        """Greedy matching based on scores"""
        matched = {}
        matched_tracks = set()
        matched_detections = set()
        
        for score, track_id, det_idx in match_scores:
            # Lower threshold for better matching after occlusion
            threshold = 0.05 if self.tracks[track_id]['age'] > 0 else 0.15
            
            if score > threshold and track_id not in matched_tracks and det_idx not in matched_detections:
                matched[track_id] = det_idx
                matched_tracks.add(track_id)
                matched_detections.add(det_idx)
        
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        return matched, unmatched_detections
    
    def _update_matched_tracks(self, matched, detections):
        """Update tracks that were matched to detections"""
        for track_id, det_idx in matched.items():
            track = self.tracks[track_id]
            old_bbox = track['bbox']
            new_bbox = detections[det_idx][:4]
            
            # Update velocity with smoothing
            old_center = self._get_center(old_bbox)
            new_center = self._get_center(new_bbox)
            new_velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            
            if track['velocity'] is not None:
                # Smooth velocity
                track['velocity'] = (
                    self.velocity_smooth * track['velocity'][0] + (1 - self.velocity_smooth) * new_velocity[0],
                    self.velocity_smooth * track['velocity'][1] + (1 - self.velocity_smooth) * new_velocity[1]
                )
            else:
                track['velocity'] = new_velocity
            
            # Update track info
            track['bbox'] = new_bbox
            track['confidence'] = detections[det_idx][4]
            track['age'] = 0
            track['hits'] += 1
            track['last_update'] = self.frame_count
            
            # Update appearance
            if self.frame_buffer is not None:
                try:
                    x1, y1, x2, y2 = [int(x) for x in new_bbox]
                    h, w = self.frame_buffer.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        region = self.frame_buffer[y1:y2, x1:x2]
                        if region.size > 0:
                            new_appearance = self._compute_color_histogram(region)
                            
                            # Smooth appearance update
                            if track['appearance'] is not None:
                                track['appearance'] = 0.7 * track['appearance'] + 0.3 * new_appearance
                            else:
                                track['appearance'] = new_appearance
                except:
                    pass
    
    def _age_tracks(self, exclude=None):
        """Age tracks that weren't matched"""
        if exclude is None:
            exclude = set()
        
        for track_id in self.tracks:
            if track_id not in exclude:
                self.tracks[track_id]['age'] += 1
    
    def _create_new_tracks(self, unmatched_detections, detections):
        """Create new tracks for unmatched detections"""
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            
            # Compute appearance
            appearance = None
            if self.frame_buffer is not None:
                try:
                    x1, y1, x2, y2 = [int(x) for x in det[:4]]
                    h, w = self.frame_buffer.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        region = self.frame_buffer[y1:y2, x1:x2]
                        if region.size > 0:
                            appearance = self._compute_color_histogram(region)
                except:
                    pass
            
            self.tracks[self.next_id] = {
                'bbox': det[:4],
                'confidence': det[4],
                'age': 0,
                'hits': 1,
                'velocity': None,
                'last_update': self.frame_count,
                'predicted_bbox': det[:4],
                'appearance': appearance
            }
            self.next_id += 1
    
    def _remove_dead_tracks(self):
        """Remove tracks that are too old"""
        dead_tracks = [tid for tid, track in self.tracks.items() if track['age'] > self.max_age]
        for track_id in dead_tracks:
            del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """Return active tracks in standard format"""
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                active_tracks.append([
                    *track['bbox'],
                    track_id,
                    track['confidence']
                ])
        
        return np.array(active_tracks) if active_tracks else np.empty((0, 6))
    
    def _predict_center(self, track, steps):
        """Predict center position after N steps"""
        center = self._get_center(track['bbox'])
        if track['velocity'] is None:
            return center
        
        # Linear prediction with velocity
        predicted_x = center[0] + track['velocity'][0] * steps
        predicted_y = center[1] + track['velocity'][1] * steps
        
        return (predicted_x, predicted_y)
    
    def _center_to_bbox(self, center, ref_bbox):
        """Convert center point back to bbox using reference bbox size"""
        width = ref_bbox[2] - ref_bbox[0]
        height = ref_bbox[3] - ref_bbox[1]
        
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        
        return np.array([x1, y1, x2, y2])
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_center_distance(self, box1, box2):
        """Calculate Euclidean distance between box centers"""
        center1 = self._get_center(box1)
        center2 = self._get_center(box2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_center(self, box):
        """Get center point of a box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# Keep SimpleTracker as fallback
class SimpleTracker:
    """
    Improved IoU-based tracker with distance matching and motion prediction
    Better for tracking distant objects that may temporarily disappear
    """
    def __init__(self, max_age=60, min_hits=1, iou_threshold=0.2, distance_threshold=150):
        self.max_age = max_age  # Increased to keep tracks longer
        self.min_hits = min_hits  # Reduced to assign IDs faster
        self.iou_threshold = iou_threshold  # Lowered for distant objects
        self.distance_threshold = distance_threshold  # Max center distance for matching
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracks with new detections using IoU + distance matching
        detections: numpy array of shape (N, 5) where each row is [x1, y1, x2, y2, confidence]
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age out old tracks
            dead_tracks = []
            for track_id in self.tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    dead_tracks.append(track_id)
            for track_id in dead_tracks:
                del self.tracks[track_id]
            return np.empty((0, 6))
        
        # Match detections to existing tracks using both IoU and distance
        matched = {}
        unmatched_detections = list(range(len(detections)))
        
        # Calculate matching scores for all track-detection pairs
        match_scores = []
        for track_id, track in self.tracks.items():
            for det_idx in unmatched_detections:
                iou = self._calculate_iou(track['bbox'], detections[det_idx][:4])
                distance = self._calculate_center_distance(track['bbox'], detections[det_idx][:4])
                
                # Combined score: prioritize IoU, but use distance for far objects
                # If IoU is very low (far objects), rely more on distance
                if iou > 0.01:
                    # For objects with some overlap, use IoU primarily
                    score = iou
                else:
                    # For distant objects (no overlap), use distance
                    # Convert distance to similarity score (lower distance = higher score)
                    if distance < self.distance_threshold:
                        score = 1.0 - (distance / self.distance_threshold)
                    else:
                        score = 0
                
                # Boost score if consistent with predicted motion
                if 'velocity' in track and track['velocity'] is not None:
                    predicted_center = self._predict_position(track)
                    det_center = self._get_center(detections[det_idx][:4])
                    pred_distance = np.linalg.norm(np.array(predicted_center) - np.array(det_center))
                    
                    # If detection is close to prediction, boost the score
                    if pred_distance < self.distance_threshold:
                        motion_bonus = 0.3 * (1.0 - pred_distance / self.distance_threshold)
                        score = min(1.0, score + motion_bonus)
                
                match_scores.append((score, track_id, det_idx))
        
        # Sort by score (highest first)
        match_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Assign matches greedily
        matched_tracks = set()
        matched_detections = set()
        
        for score, track_id, det_idx in match_scores:
            # Use a lower threshold for matching
            if score > 0.1 and track_id not in matched_tracks and det_idx not in matched_detections:
                matched[track_id] = det_idx
                matched_tracks.add(track_id)
                matched_detections.add(det_idx)
        
        # Update unmatched detections list
        unmatched_detections = [idx for idx in unmatched_detections if idx not in matched_detections]
        
        # Update matched tracks
        for track_id, det_idx in matched.items():
            old_bbox = self.tracks[track_id]['bbox']
            new_bbox = detections[det_idx][:4]
            
            # Calculate velocity for motion prediction
            old_center = self._get_center(old_bbox)
            new_center = self._get_center(new_bbox)
            velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            
            self.tracks[track_id]['bbox'] = new_bbox
            self.tracks[track_id]['confidence'] = detections[det_idx][4]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['velocity'] = velocity
            self.tracks[track_id]['last_update'] = self.frame_count
        
        # Age tracks that weren't matched
        for track_id in self.tracks:
            if track_id not in matched:
                self.tracks[track_id]['age'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks[self.next_id] = {
                'bbox': detections[det_idx][:4],
                'confidence': detections[det_idx][4],
                'age': 0,
                'hits': 1,
                'velocity': None,
                'last_update': self.frame_count
            }
            self.next_id += 1
        
        # Remove old tracks
        dead_tracks = []
        for track_id in self.tracks:
            if self.tracks[track_id]['age'] > self.max_age:
                dead_tracks.append(track_id)
        for track_id in dead_tracks:
            del self.tracks[track_id]
        
        # Return active tracks in format [x1, y1, x2, y2, track_id, confidence]
        active_tracks = []
        for track_id, track in self.tracks.items():
            # Lower hit requirement so IDs appear faster
            if track['hits'] >= self.min_hits:
                active_tracks.append([
                    *track['bbox'],
                    track_id,
                    track['confidence']
                ])
        
        return np.array(active_tracks) if active_tracks else np.empty((0, 6))
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_center_distance(self, box1, box2):
        """Calculate Euclidean distance between box centers"""
        center1 = self._get_center(box1)
        center2 = self._get_center(box2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_center(self, box):
        """Get center point of a box"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def _predict_position(self, track):
        """Predict next position based on velocity"""
        if track['velocity'] is None:
            return self._get_center(track['bbox'])
        
        center = self._get_center(track['bbox'])
        velocity = track['velocity']
        
        # Predict position based on velocity
        predicted_x = center[0] + velocity[0]
        predicted_y = center[1] + velocity[1]
        
        return (predicted_x, predicted_y)


class DroneDetectorWithTracking:
    """
    Enhanced drone detector with tracking capabilities
    """
    
    def __init__(self, model_path, device='auto', conf_threshold=0.5, tracker_type='advanced',
                 max_age=120, min_hits=1, iou_threshold=0.15, distance_threshold=200):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Tracker parameters
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load the model
        self.load_model()
        
        # Initialize tracker
        self.tracker_type = tracker_type
        self.tracker = None
        self.initialize_tracker()
        
        # Track history for visualization
        self.track_history = defaultdict(list)
        
    def load_model(self):
        """Load the PyTorch model from .pt file"""
        try:
            # Method 1: Try loading as YOLOv8 model (ultralytics)
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model_type = 'yolov8'
                print("✓ Model loaded as YOLOv8 model (ultralytics)")
                return
            except ImportError:
                print("Note: ultralytics not installed, trying YOLOv5...")
            except Exception as e:
                print(f"Note: YOLOv8 loading failed ({str(e)[:50]}...), trying other methods...")
            
            # Method 2: Try loading as YOLOv5 model via torch.hub
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=self.model_path, force_reload=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = 'yolov5'
                print("✓ Model loaded as YOLOv5 model via torch.hub")
                return
            except Exception as hub_error:
                print(f"Note: torch.hub loading failed ({str(hub_error)[:50]}...), trying checkpoint loading...")
            
            # Method 3: Try loading checkpoint and extracting model
            try:
                # PyTorch 2.6+ requires weights_only=False for models with custom classes
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if checkpoint is a dict (YOLOv5/v8 format) and extract model
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # YOLOv5/v8 checkpoint format
                    self.model = checkpoint['model']
                    # Convert to float and eval mode
                    if hasattr(self.model, 'float'):
                        self.model = self.model.float()
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    if hasattr(self.model, 'fuse'):
                        self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers
                    print("✓ Model loaded from YOLOv5/v8 checkpoint")
                elif 'ema' in checkpoint:
                    # Try EMA model
                    self.model = checkpoint['ema']
                    if hasattr(self.model, 'float'):
                        self.model = self.model.float()
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    print("✓ Model loaded from EMA checkpoint")
                else:
                    raise ValueError("Checkpoint is a dict but doesn't contain 'model' or 'ema' keys")
            else:
                # Direct model object
                self.model = checkpoint
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                print("✓ Model loaded as direct model object")
            
            # Move to device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self.model_type = 'yolov5'  # Assume v5 format for checkpoint-loaded models
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting tips:")
            print("1. For YOLOv8: pip install ultralytics")
            print("2. For YOLOv5: pip install seaborn")
            print("3. Ensure your .pt file is a valid YOLO model")
            raise
    
    def initialize_tracker(self):
        """Initialize the tracking algorithm"""
        if self.tracker_type == 'bytetrack' and BYTETRACK_AVAILABLE:
            try:
                self.tracker = BYTETracker(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=30
                )
                print("✓ ByteTrack tracker initialized")
            except Exception as e:
                print(f"Warning: Could not initialize ByteTrack: {e}")
                print("Falling back to advanced tracker")
                self.tracker = AdvancedTracker(
                    max_age=self.max_age,
                    min_hits=self.min_hits,
                    iou_threshold=self.iou_threshold,
                    distance_threshold=self.distance_threshold
                )
                self.tracker_type = 'advanced'
        elif self.tracker_type == 'advanced':
            self.tracker = AdvancedTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
                distance_threshold=self.distance_threshold
            )
            print(f"✓ Advanced tracker initialized (max_age={self.max_age}, "
                  f"min_hits={self.min_hits}, iou_thresh={self.iou_threshold}, "
                  f"distance_thresh={self.distance_threshold})")
            print("  Features: Occlusion handling, appearance matching, motion prediction")
        else:  # simple
            self.tracker = SimpleTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
                distance_threshold=self.distance_threshold
            )
            print(f"✓ Simple tracker initialized (max_age={self.max_age}, "
                  f"min_hits={self.min_hits}, iou_thresh={self.iou_threshold}, "
                  f"distance_thresh={self.distance_threshold})")
    
    def detect_frame(self, frame):
        """
        Run detection on a single frame
        
        Args:
            frame: numpy array (BGR image from cv2)
            
        Returns:
            detections: numpy array of shape (N, 5) [x1, y1, x2, y2, confidence]
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == 'yolov8':
                # YOLOv8 inference (ultralytics)
                try:
                    results = self.model(frame_rgb, verbose=False)
                    
                    # YOLOv8 returns a list of Results objects
                    if len(results) > 0:
                        result = results[0]  # Get first result
                        boxes = result.boxes  # Get boxes object
                        
                        # Extract detections
                        if boxes is not None and len(boxes) > 0:
                            # Get xyxy format (x1, y1, x2, y2)
                            xyxy = boxes.xyxy.cpu().numpy()
                            # Get confidences
                            conf = boxes.conf.cpu().numpy()
                            
                            # Filter by confidence threshold
                            mask = conf >= self.conf_threshold
                            xyxy = xyxy[mask]
                            conf = conf[mask]
                            
                            # Combine into detections array
                            if len(xyxy) > 0:
                                detections = np.column_stack([xyxy, conf])
                            else:
                                detections = np.empty((0, 5))
                        else:
                            detections = np.empty((0, 5))
                    else:
                        detections = np.empty((0, 5))
                        
                except Exception as e:
                    print(f"Warning: YOLOv8 detection error: {e}")
                    detections = np.empty((0, 5))
                    
            elif self.model_type == 'yolov5':
                # YOLOv5 inference
                try:
                    results = self.model(frame_rgb)
                    
                    # Check if results have pandas method (hub-loaded models)
                    if hasattr(results, 'pandas'):
                        predictions = results.pandas().xyxy[0]
                    # Handle results from checkpoint-loaded models
                    elif isinstance(results, (list, tuple)):
                        # YOLOv5 checkpoint format returns tuple/list
                        pred = results[0] if isinstance(results, (list, tuple)) else results
                        
                        # Convert to numpy if tensor
                        if torch.is_tensor(pred):
                            pred = pred.cpu().numpy()
                        
                        # Create DataFrame-like structure
                        if len(pred) > 0:
                            predictions = pd.DataFrame(
                                pred,
                                columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                            )
                        else:
                            predictions = pd.DataFrame(
                                columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                            )
                    else:
                        # Try to convert results to usable format
                        predictions = pd.DataFrame(
                            columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                        )
                    
                    # Filter by confidence
                    if len(predictions) > 0:
                        predictions = predictions[predictions['confidence'] >= self.conf_threshold]
                        detections = predictions[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
                    else:
                        detections = np.empty((0, 5))
                        
                except Exception as e:
                    print(f"Warning: YOLOv5 detection error: {e}")
                    detections = np.empty((0, 5))
            else:
                # For custom models - adapt as needed
                detections = np.empty((0, 5))
        
        return detections
    
    def process_video(self, video_path, output_path=None, show_window=True, skip_frames=1):
        """
        Process video with detection and tracking
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            show_window (bool): Whether to display processing window
            skip_frames (int): Process every Nth frame (1 = all frames, 2 = every other frame)
        
        Returns:
            dict: Statistics about the processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'unique_tracks': set()
        }
        
        frame_idx = 0
        
        # Reset tracker for new video
        self.initialize_tracker()
        self.track_history.clear()
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames for speed if needed
            if frame_idx % skip_frames != 0:
                if writer:
                    writer.write(frame)
                continue
            
            # Detect drones in frame
            detections = self.detect_frame(frame)
            
            # Update tracker (pass frame for appearance matching in AdvancedTracker)
            if self.tracker_type == 'advanced':
                if len(detections) > 0:
                    tracks = self.tracker.update(detections, frame=frame)
                else:
                    tracks = self.tracker.update(np.empty((0, 5)), frame=frame)
            else:
                if len(detections) > 0:
                    tracks = self.tracker.update(detections)
                else:
                    tracks = self.tracker.update(np.empty((0, 5)))
            
            # Draw tracks on frame
            frame = self.draw_tracks(frame, tracks)
            
            # Update statistics
            stats['frames_processed'] += 1
            stats['total_detections'] += len(detections)
            if len(tracks) > 0:
                stats['unique_tracks'].update(tracks[:, 4].astype(int))
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display frame
            if show_window:
                cv2.imshow('Drone Detection & Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break
            
            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames} - "
                      f"Active tracks: {len(tracks)}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        stats['unique_tracks'] = len(stats['unique_tracks'])
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Unique drone tracks: {stats['unique_tracks']}")
        if output_path:
            print(f"Output saved to: {output_path}")
        print("="*60)
        
        return stats
    
    def draw_tracks(self, frame, tracks):
        """
        Draw bounding boxes and track IDs on frame
        
        Args:
            frame: numpy array (BGR image)
            tracks: numpy array of shape (N, 6) [x1, y1, x2, y2, track_id, confidence]
        
        Returns:
            frame with drawn tracks
        """
        if len(tracks) == 0:
            return frame
        
        # Colors for different track IDs (cycling through)
        colors = [
            (255, 0, 0),    # Blue
            # (0, 255, 0),    # Green
            # (0, 0, 255),    # Red
            # (255, 255, 0),  # Cyan
            # (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            # (128, 0, 128),  # Purple
            # (255, 165, 0),  # Orange
        ]
        
        for track in tracks:
            x1, y1, x2, y2, track_id, confidence = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Choose color based on track ID
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {confidence:.2f}"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store track history (for trail visualization if needed)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.track_history[track_id].append((center_x, center_y))
            
            # Keep only last 30 points
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Draw track trail (optional - makes it easier to see movement)
            points = self.track_history[track_id]
            if len(points) > 1:
                for i in range(1, len(points)):
                    # Draw line with decreasing thickness
                    thickness = max(1, int(2 * (i / len(points))))
                    cv2.line(frame, points[i-1], points[i], color, thickness)
        
        # Draw frame info
        # info_text = f"IDs {len(tracks)}"
        # cv2.putText(frame, info_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame




def main():
    """
    Example usage
    """
    import sys
    
    # Configuration
    MODEL_PATH = "tesa_drone_model3.pt"      # Path to your .pt file
    VIDEO_PATH = "P3_VIDEO.mp4"     # Path to your input video
    OUTPUT_PATH = "output_tracked.mp4"  # Path for output video
    
    print("="*60)
    print("DRONE DETECTION WITH TRACKING")
    print("="*60)
    
    try:
        # Initialize detector with tracking
        detector = DroneDetectorWithTracking(
            model_path=MODEL_PATH,
            device='auto',  # Use 'cuda' for GPU, 'cpu' for CPU, 'auto' for automatic
            conf_threshold=0.5,  # Confidence threshold
            tracker_type='bytetrack'  # or 'simple'
        )
        
        # Process video
        stats = detector.process_video(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            show_window=True,  # Set False for headless processing
            skip_frames=1  # Process every frame (increase for speed)
        )
        
        print("\n✓ Done!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease update the paths:")
        print(f"  MODEL_PATH = 'path/to/your/model.pt'")
        print(f"  VIDEO_PATH = 'path/to/your/video.mp4'")
        print(f"  OUTPUT_PATH = 'path/to/output.mp4'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()