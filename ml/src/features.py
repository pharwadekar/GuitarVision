"""
Feature extraction from hand landmarks.

Extracts numerical features from MediaPipe hand landmarks for model training.
"""


import numpy as np
from typing import List, Dict


def _calculate_fingertip_distances(normalized_coords: np.ndarray) -> List[float]:
    """
    Calculate distances from wrist to each fingertip.
    
    Args:
        normalized_coords: Hand landmarks normalized to wrist position
        
    Returns:
        List of 5 distances (thumb, index, middle, ring, pinky)
    """
    fingertip_indices = [4, 8, 12, 16, 20]
    return [np.linalg.norm(normalized_coords[idx]) for idx in fingertip_indices]


def _calculate_bend_ratios(normalized_coords: np.ndarray) -> List[float]:
    """
    Calculate finger bend ratios (straight line / actual path length).
    
    Args:
        normalized_coords: Hand landmarks normalized to wrist position
        
    Returns:
        List of 5 bend ratios (one per finger)
    """
    finger_segments = [
        [1, 2, 3, 4],      # thumb
        [5, 6, 7, 8],      # index
        [9, 10, 11, 12],   # middle
        [13, 14, 15, 16],  # ring
        [17, 18, 19, 20]   # pinky
    ]
    
    bend_ratios = []
    for segment in finger_segments:
        mcp = normalized_coords[segment[0]]
        tip = normalized_coords[segment[-1]]
        
        tip_to_mcp = np.linalg.norm(tip - mcp)
        full_length = sum(
            np.linalg.norm(normalized_coords[segment[i+1]] - normalized_coords[segment[i]])
            for i in range(len(segment) - 1)
        )
        
        bend_ratio = tip_to_mcp / full_length if full_length > 0 else 0
        bend_ratios.append(bend_ratio)
    
    return bend_ratios


def _calculate_interfinger_angles(normalized_coords: np.ndarray) -> List[float]:
    """
    Calculate cosine of angles between adjacent fingertips.
    
    Args:
        normalized_coords: Hand landmarks normalized to wrist position
        
    Returns:
        List of 4 angle cosines (thumb-index, index-middle, middle-ring, ring-pinky)
    """
    fingertip_indices = [4, 8, 12, 16, 20]
    angles = []
    
    for i in range(len(fingertip_indices) - 1):
        v1 = normalized_coords[fingertip_indices[i]]
        v2 = normalized_coords[fingertip_indices[i+1]]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(cos_angle)
    
    return angles


def _calculate_palm_dimensions(coords: np.ndarray) -> List[float]:
    """
    Calculate palm width and height.
    
    Args:
        coords: Hand landmarks with wrist normalization applied
        
    Returns:
        List of 2 values: [palm_width, palm_height]
    """
    wrist = coords[0]
    palm_width = np.linalg.norm(coords[5] - coords[17])  # Index MCP to pinky MCP
    palm_height = np.linalg.norm(coords[9] - wrist)      # Middle MCP to wrist
    return [palm_width, palm_height]


def extract_hand_features(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Extract features from a single hand's landmarks.
    
    Extracts 16 features:
    - 5 fingertip distances from wrist
    - 5 finger bend ratios
    - 4 inter-finger angle cosines
    - 2 palm dimensions (width, height)
    
    Args:
        landmarks: List of 21 landmarks with x, y, z coordinates
        
    Returns:
        Feature vector of shape (16,)
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    
    coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
    
    # Normalize to wrist position
    wrist = coords[0]
    normalized_coords = coords - wrist
    
    # Compute all feature groups
    features = []
    features.extend(_calculate_fingertip_distances(normalized_coords))
    features.extend(_calculate_bend_ratios(normalized_coords))
    features.extend(_calculate_interfinger_angles(normalized_coords))
    features.extend(_calculate_palm_dimensions(coords))
    
    return np.array(features, dtype=np.float32)


def extract_features(hands_data: List[List[Dict[str, float]]]) -> np.ndarray:
    """
    Extract features from MediaPipe hand landmarks (supports 1 or 2 hands).
    
    Args:
        hands_data: List of hands, each hand is a list of 21 landmarks
        
    Returns:
        Combined feature vector as numpy array
    """
    if len(hands_data) == 0:
        raise ValueError("No hands detected")
    
    if len(hands_data) == 1:
        # Single hand - pad with zeros for second hand
        hand1_features = extract_hand_features(hands_data[0])
        hand2_features = np.zeros_like(hand1_features)
        return np.concatenate([hand1_features, hand2_features])
    
    # Two hands
    hand1_features = extract_hand_features(hands_data[0])
    hand2_features = extract_hand_features(hands_data[1])
    
    return np.concatenate([hand1_features, hand2_features])


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Feature array (n_samples, n_features)
        
    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    return (features - mean) / std


if __name__ == "__main__":
    print("Feature extraction module ready")
    print("Single hand features: 16 dimensions")
    print("Two hands features: 32 dimensions")
