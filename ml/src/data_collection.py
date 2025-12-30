"""
Data collection script for capturing guitar fingering images.

Captures webcam frames with MediaPipe hand landmarks for training.
"""


def collect_samples(chord_name: str, num_samples: int = 200):
    """
    Collect training samples for a specific guitar chord.
    
    Args:
        chord_name: Name of the chord (e.g., 'C', 'G', 'D')
        num_samples: Number of images to capture
    """
    raise NotImplementedError("Data collection pipeline coming in next commit")


if __name__ == "__main__":
    print("Data collection module - implementation pending")
