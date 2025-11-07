"""
Analyze ISL dataset to identify alphabets needing improvement
"""
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

print("="*70)
print("ISL DATASET ANALYSIS - IDENTIFY IMPROVEMENT AREAS")
print("="*70)

# Load dataset and model
print("\nLoading data...")
with open('../models/hand_landmarks_dataset_2hands.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open('../models/isl_landmark_labels_2hands.json', 'r') as f:
    labels = json.load(f)

X = dataset['X']
y = dataset['y']
hand_counts = dataset['hand_counts']
class_names = dataset['class_names']

# Load model
model = keras.models.load_model('../models/isl_landmark_model_2hands.h5')

# Get predictions on full dataset
print("Running predictions on entire dataset...")
y_pred = model.predict(X, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred_classes)

print("\n" + "="*70)
print("ANALYSIS RESULTS")
print("="*70)

# Analyze each class
class_stats = []
for i, class_name in enumerate(class_names):
    class_mask = y == i
    class_samples = np.sum(class_mask)
    
    # Count 1-hand vs 2-hand for this class
    one_hand = np.sum(hand_counts[class_mask] == 1)
    two_hand = np.sum(hand_counts[class_mask] == 2)
    
    # Accuracy for this class
    correct = np.sum((y == i) & (y_pred_classes == i))
    accuracy = correct / class_samples * 100 if class_samples > 0 else 0
    
    # Find most common misclassifications
    misclassified_as = {}
    for j, other_class in enumerate(class_names):
        if i != j:
            count = cm[i][j]
            if count > 0:
                misclassified_as[other_class] = count
    
    class_stats.append({
        'name': class_name,
        'samples': class_samples,
        'one_hand': one_hand,
        'two_hand': two_hand,
        'accuracy': accuracy,
        'errors': class_samples - correct,
        'misclassified_as': misclassified_as
    })

# Sort by accuracy (lowest first)
class_stats.sort(key=lambda x: x['accuracy'])

print("\n📊 ALPHABETS/NUMBERS RANKED BY ACCURACY (Lowest = Needs Most Improvement)\n")
print(f"{'Sign':<6} {'Samples':<10} {'1-Hand':<10} {'2-Hand':<10} {'Accuracy':<12} {'Errors':<8} {'Confused With'}")
print("-"*100)

for stat in class_stats:
    confused = ", ".join([f"{k}({v})" for k, v in sorted(stat['misclassified_as'].items(), key=lambda x: x[1], reverse=True)[:3]])
    if not confused:
        confused = "None"
    
    # Color code based on accuracy
    if stat['accuracy'] < 99.0:
        marker = "🔴"  # Red - needs improvement
    elif stat['accuracy'] < 99.5:
        marker = "🟡"  # Yellow - minor issues
    else:
        marker = "🟢"  # Green - good
    
    print(f"{marker} {stat['name']:<6} {stat['samples']:<10} {stat['one_hand']:<10} {stat['two_hand']:<10} {stat['accuracy']:<11.2f}% {stat['errors']:<8} {confused}")

# Identify critical issues
print("\n" + "="*70)
print("🔍 CRITICAL FINDINGS - ALPHABETS NEEDING IMPROVEMENT")
print("="*70)

needs_improvement = [s for s in class_stats if s['accuracy'] < 99.5]

if needs_improvement:
    for i, stat in enumerate(needs_improvement, 1):
        print(f"\n{i}. {stat['name']} - {stat['accuracy']:.2f}% accuracy ({stat['errors']} errors)")
        print(f"   📈 Total samples: {stat['samples']} (1-hand: {stat['one_hand']}, 2-hand: {stat['two_hand']})")
        
        if stat['misclassified_as']:
            print(f"   ⚠️  Most confused with:")
            for confused_sign, count in sorted(stat['misclassified_as'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {confused_sign}: {count} times ({count/stat['samples']*100:.1f}%)")
        
        # Recommendations
        print(f"   💡 RECOMMENDATIONS:")
        
        # Check if samples are low
        avg_samples = np.mean([s['samples'] for s in class_stats])
        if stat['samples'] < avg_samples * 0.8:
            print(f"      ✅ ADD MORE SAMPLES - Currently {stat['samples']}, average is {avg_samples:.0f}")
        
        # Check if imbalance in hand counts
        if stat['two_hand'] > 0 and stat['one_hand'] > 0:
            ratio = max(stat['one_hand'], stat['two_hand']) / min(stat['one_hand'], stat['two_hand'])
            if ratio > 2:
                print(f"      ✅ BALANCE 1-HAND vs 2-HAND samples (current ratio: {ratio:.1f}:1)")
        
        # Suggest variety
        print(f"      ✅ Add MORE VARIETY in:")
        print(f"         - Hand positions/angles")
        print(f"         - Finger positioning clarity")
        print(f"         - Wrist rotation angles")
        
        # If confused with specific signs, suggest careful distinction
        if stat['misclassified_as']:
            top_confusion = list(stat['misclassified_as'].keys())[0]
            print(f"      ✅ CAREFULLY DIFFERENTIATE from '{top_confusion}'")
            print(f"         - Review samples to ensure clear distinction")
            print(f"         - Remove ambiguous/unclear images")
else:
    print("\n🎉 ALL ALPHABETS PERFORM WELL (>99.5% accuracy)!")

# Check sample balance across classes
print("\n" + "="*70)
print("📊 SAMPLE DISTRIBUTION ANALYSIS")
print("="*70)

samples_per_class = [s['samples'] for s in class_stats]
avg_samples = np.mean(samples_per_class)
std_samples = np.std(samples_per_class)
min_samples = min(samples_per_class)
max_samples = max(samples_per_class)

print(f"\nSample Statistics:")
print(f"  Average: {avg_samples:.0f} samples per class")
print(f"  Std Dev: {std_samples:.0f}")
print(f"  Min: {min_samples} samples ({[s['name'] for s in class_stats if s['samples'] == min_samples][0]})")
print(f"  Max: {max_samples} samples ({[s['name'] for s in class_stats if s['samples'] == max_samples][0]})")

# Find imbalanced classes
imbalanced = [s for s in class_stats if s['samples'] < avg_samples - std_samples]
if imbalanced:
    print(f"\n⚖️  CLASSES WITH FEWER SAMPLES (below avg-std = {avg_samples - std_samples:.0f}):")
    for stat in imbalanced:
        print(f"   - {stat['name']}: {stat['samples']} samples (need ~{int(avg_samples - stat['samples'])} more)")

# Overall summary
print("\n" + "="*70)
print("📝 OVERALL RECOMMENDATIONS")
print("="*70)

print(f"""
1. PRIORITY IMPROVEMENTS:
   Focus on signs with <99.5% accuracy (see list above)

2. SAMPLE BALANCE:
   - Aim for {int(avg_samples)} samples per class
   - Add samples to underrepresented classes

3. QUALITY OVER QUANTITY:
   - Ensure clear hand visibility
   - Dark background for good contrast
   - Consistent positioning
   - Clear finger distinctions

4. CONFUSION PAIRS:
   - Review confused pairs carefully
   - Ensure clear visual differences
   - Remove ambiguous samples

5. 2-HAND SIGNS:
   - Verify both hands clearly visible
   - Consistent positioning of both hands
   - Good separation between hands

Current Performance: {labels['val_accuracy']*100:.2f}% validation accuracy
Target: Achieve >99.8% by addressing the issues above
""")

print("="*70)
print("Analysis complete! Review recommendations above.")
print("="*70)
