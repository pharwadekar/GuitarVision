"""
Data collection script for capturing guitar fingering images.

Captures webcam frames with MediaPipe hand landmarks for training.
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np


class GuitarDataCollector:
    """Captures images and hand landmarks for guitar chord training."""

    def __init__(self, output_dir: Path = Path("ml/data/raw")):
        self.output_dir = output_dir
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def collect_chord_samples(self, chord_name: str, num_samples: int = 200):
        """
        Collect training samples for a specific guitar chord.

        Args:
            chord_name: Name of the chord (e.g., 'C', 'G', 'D', 'Am')
            num_samples: Number of images to capture
        """
        chord_dir = self.output_dir / f"chord_{chord_name}"
        chord_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        print(f"\n{'='*60}")
        print(f"Collecting {num_samples} samples for chord: {chord_name}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Position your hands to form the chord")
        print("- Press SPACE to start capturing")
        print("- Press Q to quit early")
        print("\nWaiting for you to get ready...")

        samples_collected = 0
        capturing = False
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            while samples_collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                # Display status
                status_text = f"Chord: {chord_name} | Captured: {samples_collected}/{num_samples}"
                cv2.putText(
                    frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                if capturing:
                    cv2.putText(
                        frame, "CAPTURING...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                else:
                    cv2.putText(
                        frame, "Press SPACE to start", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )

                cv2.imshow("Guitar Chord Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting early...")
                    break
                elif key == ord(' ') and not capturing:
                    capturing = True
                    print("\nStarting capture...")

                if capturing and results.multi_hand_landmarks:
                    # Save image
                    img_path = chord_dir / f"{session_id}_{samples_collected:04d}.jpg"
                    cv2.imwrite(str(img_path), frame)

                    # Save landmarks
                    landmarks_data = {
                        "chord": chord_name,
                        "timestamp": datetime.now().isoformat(),
                        "hands": []
                    }

                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_data = []
                        for landmark in hand_landmarks.landmark:
                            hand_data.append({
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z
                            })
                        landmarks_data["hands"].append(hand_data)

                    landmarks_path = chord_dir / f"{session_id}_{samples_collected:04d}.json"
                    with open(landmarks_path, 'w') as f:
                        json.dump(landmarks_data, f, indent=2)

                    samples_collected += 1
                    if samples_collected % 10 == 0:
                        print(f"Progress: {samples_collected}/{num_samples}")

                    time.sleep(0.1)  # Small delay between captures

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"Collection complete! Saved {samples_collected} samples to {chord_dir}")
        print(f"{'='*60}\n")

        return samples_collected


def main():
    parser = argparse.ArgumentParser(description="Collect guitar chord training data")
    parser.add_argument(
        "--chord",
        type=str,
        required=True,
        help="Chord name (e.g., C, G, D, Am, Em)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to collect (default: 200)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/raw"),
        help="Output directory for collected data"
    )

    args = parser.parse_args()

    collector = GuitarDataCollector(output_dir=args.output_dir)
    collector.collect_chord_samples(args.chord, args.samples)


if __name__ == "__main__":
    main()
