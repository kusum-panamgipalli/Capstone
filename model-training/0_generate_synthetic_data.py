"""
Create a synthetic ISL dataset for testing
This generates sample hand landmark data without needing to download images
"""

import os
import numpy as np
import json
import pickle

def generate_synthetic_landmarks(num_samples_per_class=100):
    """Generate synthetic hand landmark data"""
    
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC ISL DATASET")
    print("="*60)
    print("\nThis creates sample data for testing without downloading images")
    
    # Define classes (A-Z, 0-9)
    classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + list('0123456789')
    num_classes = len(classes)
    
    print(f"\nGenerating data for {num_classes} classes:")
    print(f"Classes: {', '.join(classes)}")
    print(f"Samples per class: {num_samples_per_class}")
    
    # Create label mappings
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Generate synthetic landmarks
    # Each sample: 21 landmarks x 3 coordinates = 63 features
    all_landmarks = []
    all_labels = []
    
    print("\nGenerating synthetic hand landmarks...")
    
    for class_idx, class_name in enumerate(classes):
        # Generate unique patterns for each class with some variation
        base_pattern = np.random.rand(63) * 0.5 + 0.25  # Base pattern in range 0.25-0.75
        
        for sample_idx in range(num_samples_per_class):
            # Add noise to create variations
            noise = np.random.randn(63) * 0.05
            sample = base_pattern + noise
            
            # Clip to valid range (0-1 for normalized coordinates)
            sample = np.clip(sample, 0, 1)
            
            all_landmarks.append(sample)
            all_labels.append(class_idx)
    
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    
    print(f"\n✓ Generated {len(X)} synthetic samples")
    print(f"  Feature shape: {X.shape}")
    print(f"  Label shape: {y.shape}")
    
    return X, y, label_to_idx, idx_to_label

def save_synthetic_data(X, y, label_to_idx, idx_to_label):
    """Save synthetic data in the same format as real processed data"""
    
    output_dir = './data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING SYNTHETIC DATA")
    print("="*60)
    
    # Save as numpy files
    np.save(os.path.join(output_dir, 'X_landmarks.npy'), X)
    np.save(os.path.join(output_dir, 'y_labels.npy'), y)
    print(f"\n✓ Saved: X_landmarks.npy")
    print(f"✓ Saved: y_labels.npy")
    
    # Save label mappings
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': len(label_to_idx)
        }, f, indent=2)
    print(f"✓ Saved: label_mapping.json")
    
    # Save as pickle for easy loading
    with open(os.path.join(output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f)
    print(f"✓ Saved: dataset.pkl")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    unique, counts = np.unique(y, return_counts=True)
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Number of classes: {len(unique)}")
    print(f"Samples per class: {counts[0]} (uniform)")
    print(f"\nClasses: {', '.join(sorted(label_to_idx.keys()))}")
    
    print("\n✓ Synthetic dataset ready for training!")

def main():
    print("\n" + "="*60)
    print("SYNTHETIC ISL DATASET GENERATOR")
    print("="*60)
    print("\nThis creates sample data for testing without downloading")
    print("the full Kaggle dataset.")
    
    # Generate synthetic data
    # Using 500 samples per class for proper training
    # 36 classes x 500 samples = 18,000 total samples
    print("\nGenerating substantial dataset for proper model training...")
    print("This will create 18,000 samples (500 per class)")
    
    X, y, label_to_idx, idx_to_label = generate_synthetic_landmarks(
        num_samples_per_class=500
    )
    
    # Save data
    save_synthetic_data(X, y, label_to_idx, idx_to_label)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\n⚠ NOTE: This is SYNTHETIC data for demonstration.")
    print("The model will train properly with this data.")
    print("For real ISL recognition accuracy, use the actual Kaggle dataset.")
    print("\n✓ Dataset size: 18,000 samples - sufficient for training")
    print("\nNext step: Run 3_train_model.py to train the model")
    print("="*60)

if __name__ == "__main__":
    main()
