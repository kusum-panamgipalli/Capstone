"""
Extract hand landmarks from all training images using MediaPipe
NOW SUPPORTS 2-HAND SIGNS!
Creates dataset with:
- 1 hand detected: 63 features (21 landmarks × 3 coordinates)
- 2 hands detected: 126 features (42 landmarks × 3 coordinates)
- Missing hand: padded with zeros
"""
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler

DATA_DIR = '../Indian'  # Using ORIGINAL dataset for better landmark detection!
OUTPUT_FILE = '../models/hand_landmarks_dataset_2hands.pkl'
LANDMARKS_JSON = '../models/hand_landmarks_dataset_2hands.json'

print("="*70)
print("EXTRACT HAND LANDMARKS FROM TRAINING IMAGES")
print("NOW SUPPORTS 2-HAND SIGNS!")
print("="*70)

# Initialize MediaPipe
print("\nInitializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Processing images, not video
    max_num_hands=2,  # Detect up to 2 hands for 2-handed signs!
    min_detection_confidence=0.5
)

print("✓ MediaPipe initialized (max 2 hands)")

# Get all classes
classes = sorted([d for d in os.listdir(DATA_DIR) 
                 if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"\nFound {len(classes)} classes: {classes}")

def extract_hand_landmarks(image_path):
    """
    Extract landmarks from up to 2 hands
    Returns 126-element array (2 hands × 21 landmarks × 3 coordinates)
    If only 1 hand detected, pads second hand with zeros
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, 0
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None, 0
    
    # Extract landmarks for all detected hands
    all_landmarks = []
    num_hands = len(results.multi_hand_landmarks)
    
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract x, y, z for each of 21 landmarks
        hand_coords = []
        for landmark in hand_landmarks.landmark:
            hand_coords.extend([landmark.x, landmark.y, landmark.z])
        all_landmarks.extend(hand_coords)
    
    # Pad with zeros if only 1 hand detected (to make it 126 features)
    if num_hands == 1:
        all_landmarks.extend([0.0] * 63)  # Pad second hand with zeros
    
    return np.array(all_landmarks, dtype=np.float32), num_hands

# Extract landmarks
print("\nExtracting landmarks from images...")
print("Detecting 1 or 2 hands per image...")

landmarks_data = []
labels_data = []
hand_counts = []  # Track how many hands were detected
skipped = 0
total_processed = 0
one_hand_count = 0
two_hand_count = 0

# Create progress bar
total_images = sum([len(os.listdir(os.path.join(DATA_DIR, c))) 
                   for c in classes if os.path.isdir(os.path.join(DATA_DIR, c))])

with tqdm(total=total_images, desc="Processing images") as pbar:
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(DATA_DIR, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        images = os.listdir(class_path)
        class_landmarks = []
        class_one_hand = 0
        class_two_hand = 0
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            total_processed += 1
            
            # Extract landmarks (up to 2 hands)
            landmarks, num_hands = extract_hand_landmarks(img_path)
            
            if landmarks is not None:
                landmarks_data.append(landmarks)
                labels_data.append(class_idx)
                hand_counts.append(num_hands)
                
                if num_hands == 1:
                    one_hand_count += 1
                    class_one_hand += 1
                elif num_hands == 2:
                    two_hand_count += 1
                    class_two_hand += 1
            else:
                skipped += 1
            
            pbar.update(1)
        
        print(f"\n  Class '{class_name}': {class_one_hand + class_two_hand}/{len(images)} images ({(class_one_hand + class_two_hand)/len(images)*100:.1f}%)")
        if class_two_hand > 0:
            print(f"    ⭐ 2-hand signs detected: {class_two_hand}/{class_one_hand + class_two_hand} ({class_two_hand/(class_one_hand + class_two_hand)*100:.1f}%)")

hands.close()

print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
print(f"\n✓ Total images processed: {total_processed}")
print(f"✓ Successful extractions: {len(landmarks_data)}")
print(f"  - 1-hand signs: {one_hand_count} ({one_hand_count/len(landmarks_data)*100:.1f}%)")
print(f"  - 2-hand signs: {two_hand_count} ({two_hand_count/len(landmarks_data)*100:.1f}%)")
print(f"✗ Skipped (no hand detected): {skipped}")
print(f"✓ Success rate: {len(landmarks_data)/total_processed*100:.1f}%")

# Convert to numpy arrays
X = np.array(landmarks_data, dtype=np.float32)
y = np.array(labels_data, dtype=np.int32)
hand_counts = np.array(hand_counts, dtype=np.int32)

print(f"\n✓ Dataset shape: {X.shape}")
print(f"  - Samples: {X.shape[0]}")
print(f"  - Features: {X.shape[1]} (up to 2 hands × 21 landmarks × 3 coordinates)")
print(f"  - Classes: {len(classes)}")

# Normalize features
print(f"\nNormalizing landmark features...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
print(f"✓ Features normalized (zero mean, unit variance)")

# Save dataset
print(f"\nSaving dataset to {OUTPUT_FILE}...")
dataset = {
    'X': X_normalized,
    'y': y,
    'hand_counts': hand_counts,  # Track 1-hand vs 2-hand signs
    'class_names': classes,
    'mean': scaler.mean_,
    'std': scaler.scale_,
    'feature_count': X.shape[1],
    'supports_2_hands': True
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(dataset, f)

print(f"✓ Dataset saved as {OUTPUT_FILE}")

# Save metadata
metadata = {
    'total_samples': len(X),
    'feature_count': int(X.shape[1]),
    'num_classes': len(classes),
    'class_names': classes,
    'success_rate': float(len(landmarks_data)/total_processed*100),
    'one_hand_samples': int(one_hand_count),
    'two_hand_samples': int(two_hand_count),
    'supports_2_hands': True,
    'mean': scaler.mean_.tolist(),
    'std': scaler.scale_.tolist()
}

with open(LANDMARKS_JSON, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved as {LANDMARKS_JSON}")

# Statistics
print(f"\n{'='*70}")
print("SAMPLE STATISTICS")
print("="*70)

print(f"\nFirst sample (class '{classes[y[0]]}'):")
if hand_counts[0] == 2:
    print("  🤚🤚 2-HAND SIGN DETECTED!")
    print(f"  Left hand - Landmark 0 (wrist): x={X[0][0]:.3f}, y={X[0][1]:.3f}, z={X[0][2]:.3f}")
    print(f"  Right hand - Landmark 0 (wrist): x={X[0][63]:.3f}, y={X[0][64]:.3f}, z={X[0][65]:.3f}")
else:
    print("  🤚 1-HAND SIGN")
    print(f"  Landmark 0 (wrist): x={X[0][0]:.3f}, y={X[0][1]:.3f}, z={X[0][2]:.3f}")

print(f"\nClass distribution:")
for i, class_name in enumerate(classes):
    count = np.sum(y == i)
    two_hand_in_class = np.sum((y == i) & (hand_counts == 2))
    if two_hand_in_class > 0:
        print(f"  {class_name}: {count} samples (⭐ {two_hand_in_class} are 2-hand signs)")
    else:
        print(f"  {class_name}: {count} samples")

print(f"\n{'='*70}")
print("READY FOR TRAINING!")
print("="*70)

print(f"\nDataset Summary:")
print(f"  - Input features: {X.shape[1]} (supports 1-hand and 2-hand signs)")
print(f"  - Training samples: {len(X)}")
print(f"  - Output classes: {len(classes)}")
print(f"  - 1-hand signs: {one_hand_count}")
print(f"  - 2-hand signs: {two_hand_count}")

print(f"\nNext step: python train_landmark_model_2hands.py")
print("="*70)
