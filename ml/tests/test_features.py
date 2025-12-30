"""Tests for feature extraction module."""
import numpy as np
import pytest
from ml.src.features import extract_hand_features, extract_features, normalize_features


def create_mock_landmarks(num_points=21):
    """Create mock landmark data for testing."""
    return [
        {"x": np.random.rand(), "y": np.random.rand(), "z": np.random.rand()}
        for _ in range(num_points)
    ]


def test_extract_hand_features_shape():
    """Test that single hand feature extraction produces correct shape."""
    landmarks = create_mock_landmarks(21)
    features = extract_hand_features(landmarks)
    
    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0  # Should have some features
    assert features.dtype == np.float32


def test_extract_hand_features_invalid_input():
    """Test that invalid landmark count raises error."""
    landmarks = create_mock_landmarks(10)  # Wrong number
    
    with pytest.raises(ValueError, match="Expected 21 landmarks"):
        extract_hand_features(landmarks)


def test_extract_features_single_hand():
    """Test feature extraction with single hand (should pad)."""
    hands_data = [create_mock_landmarks(21)]
    features = extract_features(hands_data)
    
    assert isinstance(features, np.ndarray)
    # Should have double the features (padded for second hand)
    single_hand_size = extract_hand_features(hands_data[0]).shape[0]
    assert features.shape[0] == single_hand_size * 2


def test_extract_features_two_hands():
    """Test feature extraction with two hands."""
    hands_data = [create_mock_landmarks(21), create_mock_landmarks(21)]
    features = extract_features(hands_data)
    
    assert isinstance(features, np.ndarray)
    single_hand_size = extract_hand_features(hands_data[0]).shape[0]
    assert features.shape[0] == single_hand_size * 2


def test_extract_features_no_hands():
    """Test that no hands raises error."""
    with pytest.raises(ValueError, match="No hands detected"):
        extract_features([])


def test_normalize_features():
    """Test feature normalization."""
    # Create sample features
    features = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    normalized = normalize_features(features)
    
    # Check mean is close to 0 and std is close to 1
    assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-7)


def test_normalize_features_zero_variance():
    """Test normalization handles zero variance columns."""
    features = np.array([
        [1.0, 5.0, 3.0],
        [1.0, 5.0, 6.0],
        [1.0, 5.0, 9.0]
    ])
    
    normalized = normalize_features(features)
    
    # First column should remain unchanged (constant value)
    assert np.allclose(normalized[:, 0], 0, atol=1e-7)


def test_feature_consistency():
    """Test that same input produces same features."""
    landmarks = create_mock_landmarks(21)
    
    features1 = extract_hand_features(landmarks)
    features2 = extract_hand_features(landmarks)
    
    assert np.allclose(features1, features2)
