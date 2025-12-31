"""
Data collection script for GuitarVision.

Captures fretting-hand (LEFT hand for right-handed players) landmarks
using MediaPipe, normalizes them, and saves ML-ready samples for
chord + fret recognition.
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


# =========================
# Utility Functions
# =========================

def normalize_landmarks(landmarks: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Normalize landmarks:
    - Wrist is origin
    - Wrist → Middle MCP is scale
    """
    wrist = landmarks[0]

    def dist(a, b):
        return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5

    scale = dist(landmarks[0], landmarks[9]) + 1e-6  # wrist → middle MCP

    normalized = []
    for lm in landmarks:
        normalized.append({
            "x": (lm["x"] - wrist["x"]) / scale,
            "y": (lm["y"] - wrist["y"]) / scale,
            "z": (lm["z"] - wrist["z"]) / scale,
        })

    return normalized


# =========================
# Data Collector
# =========================

class GuitarDataCollector:
    """
    Collects ML-ready guitar chord + fret data.
    Assumes RIGHT-HANDED guitarist → fretting hand = LEFT.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sample_count = 0

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # MediaPipe Face Mesh (for eyebrow trigger)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )
        self.prev_eyebrow_height = None
        self.eyebrow_baseline = None
        self.smoothed_height = None
        self.last_trigger_time = 0.0
        self.cooldown_sec = 1.0
        self.eyebrow_latch_until = 0
        self.eyebrow_latch_duration = 0.7  # seconds
        self.last_eyebrow_capture_time = 0
        self.eyebrow_cooldown = 1.0  # seconds

    # =========================
    # Eyebrow Detection
    # =========================

    def detect_eyebrow_raise(self, frame: np.ndarray) -> bool:
        results = self.face_mesh.process(frame)
        if not results.multi_face_landmarks:
            self.prev_eyebrow_height = None
            return False

        face_landmarks = results.multi_face_landmarks[0]

        right_brow = (face_landmarks.landmark[55].y +
                      face_landmarks.landmark[65].y) / 2
        left_brow = (face_landmarks.landmark[285].y +
                     face_landmarks.landmark[295].y) / 2

        right_eye = face_landmarks.landmark[159].y
        left_eye = face_landmarks.landmark[386].y

        current_height = abs(right_brow - right_eye) + abs(left_brow - left_eye)

        if self.prev_eyebrow_height is not None:
            diff = current_height - self.prev_eyebrow_height
            if diff > 0.015:
                self.prev_eyebrow_height = current_height
                return True

        self.prev_eyebrow_height = current_height
        return False


    # =========================
    # Landmark Extraction
    # =========================

    def extract_landmarks(self, hand_landmarks) -> List[Dict[str, float]]:
        return [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in hand_landmarks.landmark
        ]

    # =========================
    # Save Sample
    # =========================

    def save_sample(
        self,
        raw_landmarks: List[Dict],
        normalized_landmarks: List[Dict],
        chord: str,
        fret: int
    ):
        sample = {
            "metadata": {
                "chord": chord,
                "fret": fret,
                "handedness": "right",
                "fretting_hand": "left",
                "session_id": self.session_id,
                "sample_id": self.sample_count,
                "timestamp": datetime.now().isoformat()
            },
            "landmarks": {
                "raw": raw_landmarks,
                "normalized": normalized_landmarks
            }
        }

        out_path = self.output_dir / f"{self.session_id}_{self.sample_count:04d}.json"
        with open(out_path, "w") as f:
            json.dump(sample, f, indent=2)

        self.sample_count += 1
        print(f"✓ Saved sample {self.sample_count}")

    def save_sample(self, frame: np.ndarray, hands_data: List[List[Dict]], chord_name: str, fret: int = 1):
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

    # =========================
    # Main Collection Loop
    # =========================

    def collect(
        self,
        chord: str,
        fret: int,
        num_samples: int,
        auto_capture: bool,
        interval: float
    ):
    
        cap = cv2.VideoCapture(0)
        last_capture = 0

        print("\nStarting collection...")
        print(f"Chord: {chord} | Fret: {fret}")
        print("Press Q to quit.\n")

        time.sleep(2)

        while self.sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb)

            should_capture = False

            if auto_capture:
                if time.time() - last_capture > interval:
                    should_capture = True
            else:
                # Eyebrow raise with cooldown
                if self.detect_eyebrow_raise(rgb):
                    now = time.time()
                    if now - self.last_eyebrow_capture_time > self.eyebrow_cooldown:
                        should_capture = True
                        self.last_eyebrow_capture_time = now
                # Also check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    should_capture = True
                elif key == ord("q"):
                    print("\nQuitting early...")
                    break

            if results.multi_hand_landmarks:
                for hand_lms, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    if handedness.classification[0].label == "Left":
                        raw = self.extract_landmarks(hand_lms)
                        norm = normalize_landmarks(raw)

                        if should_capture:
                            self.save_sample(frame, [raw], chord, fret)
                            last_capture = time.time()

                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_lms,
                            self.mp_hands.HAND_CONNECTIONS
                        )

            cv2.putText(
                frame,
                f"{chord} | fret {fret} | {self.sample_count}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display eyebrow diff for debugging
            if self.smoothed_height is not None and self.eyebrow_baseline is not None:
                current_height = self.smoothed_height
                eyebrow_diff = current_height - (self.prev_eyebrow_height if self.prev_eyebrow_height is not None else current_height)
                cv2.putText(
                    frame,
                    f"Eyebrow diff: {eyebrow_diff:.4f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                self.prev_eyebrow_height = current_height

            cv2.imshow("GuitarVision Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.face_mesh.close()

        print("\n✔ Collection complete.")


# =========================
# Entry Point
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chord", required=True, type=str)
    parser.add_argument("--fret", default=1, type=int)
    parser.add_argument("--samples", default=50, type=int)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--interval", default=0.5, type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/processed")
    )

    args = parser.parse_args()

    collector = GuitarDataCollector(args.output_dir / args.chord)
    collector.collect(
        chord=args.chord,
        fret=args.fret,
        num_samples=args.samples,
        auto_capture=args.auto,
        interval=args.interval
    )
    # cv2.putText(
    #     frame,
    #     f"Eyebrow Δ: {self.smoothed_height - self.eyebrow_baseline:.4f}",
    #     (10, 70),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 0),
    #     2
    # )


if __name__ == "__main__":
    main()

