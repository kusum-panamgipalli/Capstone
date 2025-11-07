"""
Detailed error analysis - find exact misclassifications
"""
import numpy as np
import pickle
import json
from tensorflow import keras
from sklearn.model_selection import train_test_split

print("="*70)
print("DETAILED ERROR ANALYSIS - VALIDATION SET")
print("="*70)

# Load dataset
with open('../models/hand_landmarks_dataset_2hands.pkl', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['X']
y = dataset['y']
hand_counts = dataset['hand_counts']
class_names = dataset['class_names']

# Split same way as training
X_train, X_val, y_train, y_val, hand_train, hand_val = train_test_split(
    X, y, hand_counts, test_size=0.2, random_state=42, stratify=y
)

# Load model
model = keras.models.load_model('../models/isl_landmark_model_2hands.h5')

# Get predictions on validation set only
print(f"\nAnalyzing validation set ({len(X_val)} samples)...")
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_confidence = np.max(y_pred, axis=1)

# Find all errors
errors = np.where(y_val != y_pred_classes)[0]

print(f"\n{'='*70}")
print(f"VALIDATION ERRORS: {len(errors)} out of {len(y_val)} samples")
print(f"Validation Accuracy: {(1 - len(errors)/len(y_val))*100:.2f}%")
print(f"{'='*70}\n")

if len(errors) > 0:
    print(f"{'Index':<8} {'True':<8} {'Predicted':<10} {'Confidence':<12} {'Hands':<8}")
    print("-"*60)
    
    error_details = []
    for idx in errors:
        true_sign = class_names[y_val[idx]]
        pred_sign = class_names[y_pred_classes[idx]]
        confidence = y_pred_confidence[idx] * 100
        hands = hand_val[idx]
        
        print(f"{idx:<8} {true_sign:<8} {pred_sign:<10} {confidence:<11.2f}% {hands:<8}")
        
        error_details.append({
            'index': int(idx),
            'true': true_sign,
            'predicted': pred_sign,
            'confidence': float(confidence),
            'hands': int(hands)
        })
    
    # Group errors by confusion pairs
    print(f"\n{'='*70}")
    print("ERROR PATTERNS - CONFUSION PAIRS")
    print(f"{'='*70}\n")
    
    confusion_pairs = {}
    for err in error_details:
        pair = f"{err['true']} → {err['predicted']}"
        if pair not in confusion_pairs:
            confusion_pairs[pair] = 0
        confusion_pairs[pair] += 1
    
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair}: {count} time(s)")
    
    # Analysis by hands
    print(f"\n{'='*70}")
    print("ERRORS BY HAND COUNT")
    print(f"{'='*70}\n")
    
    one_hand_errors = [e for e in error_details if e['hands'] == 1]
    two_hand_errors = [e for e in error_details if e['hands'] == 2]
    
    print(f"1-hand sign errors: {len(one_hand_errors)}")
    print(f"2-hand sign errors: {len(two_hand_errors)}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("SPECIFIC RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
        true_sign, pred_sign = pair.split(' → ')
        print(f"\n🔍 {pair} ({count} error(s))")
        print(f"   ✅ Review '{true_sign}' samples - ensure clear distinction from '{pred_sign}'")
        print(f"   ✅ Add more varied samples of '{true_sign}'")
        print(f"   ✅ Check if hand position/finger placement is ambiguous")
        print(f"   ✅ Remove any unclear/blurry images of '{true_sign}'")

else:
    print("🎉 PERFECT! No validation errors!")

print(f"\n{'='*70}")
print("SAMPLE DISTRIBUTION IN VALIDATION SET")
print(f"{'='*70}\n")

# Count samples per class in validation
val_distribution = {}
for i, name in enumerate(class_names):
    count = np.sum(y_val == i)
    val_distribution[name] = count

print(f"{'Sign':<6} {'Val Samples':<12}")
print("-"*20)
for name, count in sorted(val_distribution.items()):
    print(f"{name:<6} {count:<12}")

print(f"\n{'='*70}")
