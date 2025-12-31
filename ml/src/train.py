"""
Model training script.

Trains guitar chord classification model using RandomForest.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.src.features import extract_features


def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load training data from processed directory.
    
    Args:
        data_dir: Path to processed data directory
        
    Returns:
        Tuple of (features, labels, label_names)
    """
    features_list = []
    labels_list = []
    label_names = []
    
    # Scan for chord directories
    chord_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not chord_dirs:
        raise ValueError(f"No chord directories found in {data_dir}")
    
    print(f"Found {len(chord_dirs)} chord classes")
    
    for chord_dir in sorted(chord_dirs):
        chord_name = chord_dir.name
        if chord_name not in label_names:
            label_names.append(chord_name)
        
        label_idx = label_names.index(chord_name)
        
        # Load all JSON landmark files
        json_files = list(chord_dir.glob("*.json"))
        print(f"  {chord_name}: {len(json_files)} samples")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract hands data
                hands_data = data.get('hands', [])
                
                if not hands_data:
                    continue
                
                # Extract features
                feature_vector = extract_features(hands_data)
                features_list.append(feature_vector)
                labels_list.append(label_idx)
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"    Skipping {json_file.name}: {e}")
                continue
    
    if not features_list:
        raise ValueError("No valid training samples found")
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    print(f"\nLoaded {len(features)} total samples")
    print(f"Feature shape: {features.shape}")
    print(f"Classes: {label_names}")
    
    return features, labels, label_names


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 20
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train RandomForest classifier with train/val/test splits.
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        test_size: Fraction for test set
        val_size: Fraction of remaining data for validation
        random_state: Random seed
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("\n=== Splitting Dataset ===")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Normalize features
    print("\n=== Normalizing Features ===")
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1.0
    
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    
    # Train RandomForest
    print("\n=== Training RandomForest ===")
    print(f"n_estimators={n_estimators}, max_depth={max_depth}")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train_norm, y_train)
    
    # Evaluate
    print("\n=== Evaluation ===")
    
    train_pred = clf.predict(X_train_norm)
    val_pred = clf.predict(X_val_norm)
    test_pred = clf.predict(X_test_norm)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    metrics = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'n_samples_train': int(len(X_train)),
        'n_samples_val': int(len(X_val)),
        'n_samples_test': int(len(X_test)),
        'normalization_mean': train_mean.tolist(),
        'normalization_std': train_std.tolist(),
        'feature_importance': clf.feature_importances_.tolist()
    }
    
    return clf, metrics


def save_model(
    model: RandomForestClassifier,
    label_names: List[str],
    metrics: Dict,
    output_dir: Path
):
    """
    Save trained model, labels, and metrics.
    
    Args:
        model: Trained classifier
        label_names: List of class names
        metrics: Training metrics
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {model_path}")
    
    # Save labels
    labels_path = output_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(label_names, f, indent=2)
    print(f"Saved labels to {labels_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train guitar chord classifier")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('ml/data/processed'),
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ml/models'),
        help='Path to save trained model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for test set'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Fraction of data for validation'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in RandomForest'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum depth of trees'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("=== Guitar Chord Classifier Training ===\n")
    
    # Load data
    print("=== Loading Training Data ===")
    features, labels, label_names = load_training_data(args.data_dir)
    
    # Train model
    model, metrics = train_model(
        features,
        labels,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Save outputs
    save_model(model, label_names, metrics, args.output_dir)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
