"""
Extract hand landmarks from ISL dataset images using MediaPipe
This creates a structured dataset of hand keypoints for training
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class LandmarkExtractor:
    def __init__(self, data_dir='./data/raw', output_dir='./data/processed'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store extracted data
        self.landmarks_data = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def find_dataset_images(self):
        """Find all images in the dataset"""
        image_paths = []
        labels = []
        
        # Common image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        print(f"\nScanning directory: {self.data_dir}")
        
        # Look for organized folders (A-Z, 0-9)
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    # Get label from folder name
                    folder_name = os.path.basename(root)
                    
                    # Filter valid ISL labels (A-Z, 0-9)
                    if folder_name.isalnum() and len(folder_name) == 1:
                        image_path = os.path.join(root, file)
                        image_paths.append(image_path)
                        labels.append(folder_name.upper())
        
        print(f"Found {len(image_paths)} images across {len(set(labels))} classes")
        return image_paths, labels
    
    def extract_landmarks_from_image(self, image_path):
        """Extract hand landmarks from a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(image_rgb)
            
            # Extract landmarks if hand detected
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert landmarks to numpy array
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks)
            
            return None
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def create_label_mapping(self, unique_labels):
        """Create mapping between labels and indices"""
        # Sort labels: Numbers first (0-9), then letters (A-Z)
        sorted_labels = sorted(unique_labels, key=lambda x: (x.isalpha(), x))
        
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"\nLabel mapping created for {len(sorted_labels)} classes:")
        print(f"Classes: {', '.join(sorted_labels)}")
        
        return self.label_to_idx, self.idx_to_label
    
    def process_dataset(self):
        """Process all images and extract landmarks"""
        print("\n" + "="*60)
        print("EXTRACTING HAND LANDMARKS FROM ISL DATASET")
        print("="*60)
        
        # Find all images
        image_paths, labels = self.find_dataset_images()
        
        if len(image_paths) == 0:
            print("\n❌ No images found!")
            print("\nPlease ensure:")
            print("1. Dataset is downloaded in ./data/raw/")
            print("2. Images are organized in folders (A-Z, 0-9)")
            return False
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.create_label_mapping(unique_labels)
        
        # Process each image
        print(f"\nProcessing {len(image_paths)} images...")
        
        successful = 0
        failed = 0
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            landmarks = self.extract_landmarks_from_image(img_path)
            
            if landmarks is not None:
                self.landmarks_data.append(landmarks)
                self.labels.append(self.label_to_idx[label])
                successful += 1
            else:
                failed += 1
        
        print(f"\n✓ Successfully processed: {successful} images")
        print(f"✗ Failed to process: {failed} images")
        
        return successful > 0
    
    def save_processed_data(self):
        """Save extracted landmarks and labels"""
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Convert to numpy arrays
        X = np.array(self.landmarks_data)
        y = np.array(self.labels)
        
        print(f"\nDataset shape:")
        print(f"  Features (X): {X.shape}")
        print(f"  Labels (y): {y.shape}")
        
        # Save as numpy files
        np.save(os.path.join(self.output_dir, 'X_landmarks.npy'), X)
        np.save(os.path.join(self.output_dir, 'y_labels.npy'), y)
        
        # Save label mappings
        with open(os.path.join(self.output_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label,
                'num_classes': len(self.label_to_idx)
            }, f, indent=2)
        
        # Save as pickle for easy loading
        with open(os.path.join(self.output_dir, 'dataset.pkl'), 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label
            }, f)
        
        # Create CSV for analysis
        df = pd.DataFrame(X)
        df['label'] = y
        df['label_name'] = df['label'].map(self.idx_to_label)
        df.to_csv(os.path.join(self.output_dir, 'landmarks_dataset.csv'), index=False)
        
        print(f"\n✓ Saved processed data to: {self.output_dir}")
        print(f"\nFiles created:")
        print(f"  - X_landmarks.npy (feature data)")
        print(f"  - y_labels.npy (labels)")
        print(f"  - label_mapping.json (class mappings)")
        print(f"  - dataset.pkl (complete dataset)")
        print(f"  - landmarks_dataset.csv (for analysis)")
        
        # Print dataset statistics
        self.print_statistics(y)
        
        return True
    
    def print_statistics(self, y):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        unique, counts = np.unique(y, return_counts=True)
        
        print(f"\nTotal samples: {len(y)}")
        print(f"Number of classes: {len(unique)}")
        print(f"\nSamples per class:")
        
        for idx, count in zip(unique, counts):
            label_name = self.idx_to_label[idx]
            print(f"  {label_name}: {count} samples")
        
        print(f"\nAverage samples per class: {np.mean(counts):.1f}")
        print(f"Min samples: {np.min(counts)}")
        print(f"Max samples: {np.max(counts)}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.hands.close()

def main():
    # Initialize extractor
    extractor = LandmarkExtractor(
        data_dir='./data/raw',
        output_dir='./data/processed'
    )
    
    # Process dataset
    success = extractor.process_dataset()
    
    if success:
        # Save processed data
        extractor.save_processed_data()
        print("\n✓ Dataset processing complete!")
        print("\nNext step: Run 3_train_model.py")
    else:
        print("\n❌ Dataset processing failed!")
        print("\nPlease check:")
        print("1. Dataset images are in ./data/raw/")
        print("2. Images are organized in folders (A-Z, 0-9)")
        print("3. Image files are valid (jpg, png, etc.)")
    
    # Cleanup
    extractor.cleanup()

if __name__ == "__main__":
    main()
