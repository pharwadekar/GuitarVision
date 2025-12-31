"""
Data collection script for capturing guitar chord hand landmarks.

Captures webcam frames with MediaPipe hand detection and saves landmarks as JSON.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np


class GuitarDataCollector:
    """Collects training data for guitar chord recognition."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Mesh for eyebrow detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_eyebrow_height = None
        
        # Session ID for unique filenames
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sample_count = 0
    
    def extract_landmarks(self, results) -> Optional[List[List[Dict[str, float]]]]:
        """
        Extract hand landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe hand detection results
            
        Returns:
            List of hands, each containing 21 landmarks with x, y, z coordinates
        """
        if not results.multi_hand_landmarks:
            return None
        
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                })
            hands_data.append(landmarks)
        
        return hands_data
    
    def detect_eyebrow_raise(self, frame: np.ndarray) -> bool:
        """
        Detect eyebrow raise gesture from face landmarks.
        
        Args:
            frame: RGB video frame
            
        Returns:
            True if eyebrow raise detected
        """
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            self.prev_eyebrow_height = None
            return False
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get eyebrow and eye landmarks (indices from MediaPipe Face Mesh)
        # Right eyebrow top: 55, 65  |  Right eye top: 159
        # Left eyebrow top: 285, 295 |  Left eye top: 386
        right_brow = (face_landmarks.landmark[55].y + face_landmarks.landmark[65].y) / 2
        left_brow = (face_landmarks.landmark[285].y + face_landmarks.landmark[295].y) / 2
        right_eye = face_landmarks.landmark[159].y
        left_eye = face_landmarks.landmark[386].y
        
        # Calculate distance between eyebrow and eye
        current_height = abs(right_brow - right_eye) + abs(left_brow - left_eye)
        
        # Detect raise (significant increase in distance)
        if self.prev_eyebrow_height is not None:
            diff = current_height - self.prev_eyebrow_height
            if diff > 0.015:  # Threshold for eyebrow raise
                self.prev_eyebrow_height = current_height
                return True
        
        self.prev_eyebrow_height = current_height
        return False
    
    def save_sample(self, frame: np.ndarray, hands_data: List[List[Dict]], chord_name: str, fret: int = 1):
        """
        Save captured sample (image + landmarks JSON).
        
        Args:
            frame: Captured video frame
            hands_data: Hand landmark data
            chord_name: Name of the chord
        """
        # Generate filename
        filename = f"{self.session_id}_{self.sample_count:04d}"
        
        # Save image
        img_path = self.output_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        # Save landmarks as JSON
        json_path = self.output_dir / f"{filename}.json"
        data = {
            'hands': hands_data,
            'chord': chord_name,
            'fret': fret,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'sample_id': self.sample_count
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.sample_count += 1
    
    def collect_chord_samples(
        self,
        chord_name: str,
        num_samples: int,
        auto_capture: bool = True,
        capture_interval: float = 0.5,
        fret: int = 1
    ):
        """
        Collect samples for a specific chord.
        
        Args:
            chord_name: Name of the chord (e.g., 'C', 'G', 'D')
            num_samples: Number of samples to collect
            auto_capture: If True, auto-capture at intervals; if False, manual SPACE key
            capture_interval: Seconds between auto-captures (if auto_capture=True)
        """
        self.fret = fret
        
        print(f"\n=== Collecting {num_samples} samples for chord '{chord_name}' ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Mode: {'Auto-capture' if auto_capture else 'Manual (eyebrow raise or SPACE)'}")
        print("\nInstructions:")
        print("  - Position your hands on the guitar in the chord formation")
        print("  - Ensure both hands are visible to the camera")
        if not auto_capture:
            print("  - Raise your eyebrows to capture each sample")
            print("  - Or press SPACE as alternative")
        print("  - Press Q to quit early\n")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        print("Camera opened. Starting collection in 3 seconds...")
        time.sleep(3)
        
        last_capture_time = 0
        
        while self.sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Draw landmarks on frame
            display_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display progress
            progress_text = f"Chord: {chord_name} | Captured: {self.sample_count}/{num_samples}"
            cv2.putText(
                display_frame,
                progress_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display hands detected
            hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            hands_text = f"Hands detected: {hands_detected}"
            color = (0, 255, 0) if hands_detected > 0 else (0, 0, 255)
            cv2.putText(
                display_frame,
                hands_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            cv2.imshow('Guitar Chord Data Collection', display_frame)
            
            # Check for capture
            current_time = time.time()
            should_capture = False
            
            if auto_capture:
                if current_time - last_capture_time >= capture_interval:
                    should_capture = True
            else:
                # Check for eyebrow raise
                if self.detect_eyebrow_raise(rgb_frame):
                    should_capture = True
                
                # Also check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    should_capture = True
                elif key == ord('q'):
                    print("\nQuitting early...")
                    break
            
            # Capture sample if hands detected
            if should_capture and results.multi_hand_landmarks:
                hands_data = self.extract_landmarks(results)
                if hands_data:
                    self.save_sample(frame, hands_data, chord_name, fret)
                    last_capture_time = current_time
                    print(f"Captured sample {self.sample_count}/{num_samples}")
            
            if not auto_capture:
                cv2.waitKey(1)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.face_mesh.close()
        
        print(f"\n✓ Collection complete! Saved {self.sample_count} samples to {self.output_dir}")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
        except (ValueError, AttributeError):
            pass  # Already closed or not initialized
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh:
                self.face_mesh.close()
        except (ValueError, AttributeError):
            pass  # Already closed or not initialized


def main():
    """Main data collection script."""
    parser = argparse.ArgumentParser(description="Collect guitar chord training data")
    parser.add_argument(
        '--chord',
        type=str,
        required=True,
        help='Name of the chord (e.g., C, G, D, Em, Am)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of samples to collect'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: ml/data/processed/<chord>)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-capture mode (vs manual SPACE key)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=0.5,
        help='Capture interval in seconds (auto mode only)'
    )
    parser.add_argument(
        '--fret',
        type=int,
        default=1,
        help='Fret number being held (e.g., 1, 3, 5)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"ml/data/processed/{args.chord}")
    
    # Interactive mode selection if not specified
    auto_capture = args.auto
    if not auto_capture:
        print("\nSelect capture mode:")
        print("  1. Auto-capture (captures automatically every 0.5-1s)")
        print("  2. Manual (press SPACE for each capture)")
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        auto_capture = (choice != '2')
    
    # Create collector and start collection
    collector = GuitarDataCollector(args.output_dir)
    collector.collect_chord_samples(
        chord_name=args.chord,
        num_samples=args.samples,
        auto_capture=auto_capture,
        capture_interval=args.interval,
        fret=args.fret
    )


if __name__ == "__main__":
    main()
