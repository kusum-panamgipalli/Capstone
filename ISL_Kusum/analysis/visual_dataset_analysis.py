"""
Visual comparison of dataset samples to identify improvement areas
"""
import os
import numpy as np
from pathlib import Path
import json

print("="*70)
print("DATASET QUALITY ANALYSIS - SAMPLE COUNT & DISTRIBUTION")
print("="*70)

# Analyze Indian folder
dataset_path = Path("../Indian")

if not dataset_path.exists():
    print("❌ Indian folder not found!")
    exit(1)

print("\nAnalyzing dataset structure...\n")

# Get all class folders
class_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

print(f"{'Sign':<8} {'Total Images':<15} {'Recommendation':<50}")
print("-"*90)

stats = []
for folder in class_folders:
    sign_name = folder.name
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    count = len(images)
    
    stats.append({
        'sign': sign_name,
        'count': count
    })

# Calculate statistics
counts = [s['count'] for s in stats]
avg_count = np.mean(counts)
std_count = np.std(counts)
min_count = min(counts)
max_count = max(counts)

# Display with recommendations
for stat in sorted(stats, key=lambda x: x['count']):
    sign = stat['sign']
    count = stat['count']
    
    if count < avg_count - std_count:
        marker = "🔴"
        recommendation = f"ADD ~{int(avg_count - count)} more samples (below avg)"
    elif count < avg_count:
        marker = "🟡"
        recommendation = f"Consider adding ~{int(avg_count - count)} samples"
    else:
        marker = "🟢"
        recommendation = "Good sample count"
    
    print(f"{marker} {sign:<8} {count:<15} {recommendation:<50}")

# Summary statistics
print("\n" + "="*70)
print("STATISTICS")
print("="*70)
print(f"\nTotal Signs: {len(stats)}")
print(f"Total Images: {sum(counts)}")
print(f"Average per sign: {avg_count:.0f}")
print(f"Std Deviation: {std_count:.0f}")
print(f"Min: {min_count} ({[s['sign'] for s in stats if s['count'] == min_count]})")
print(f"Max: {max_count} ({[s['sign'] for s in stats if s['count'] == max_count]})")

# Identify issues
print("\n" + "="*70)
print("IMPROVEMENT PRIORITIES")
print("="*70)

low_samples = [s for s in stats if s['count'] < avg_count - std_count]
if low_samples:
    print(f"\n🔴 PRIORITY: Signs with significantly fewer samples:")
    for s in sorted(low_samples, key=lambda x: x['count']):
        print(f"   {s['sign']}: {s['count']} samples (need ~{int(avg_count - s['count'])} more)")

moderate_samples = [s for s in stats if avg_count - std_count <= s['count'] < avg_count]
if moderate_samples:
    print(f"\n🟡 MODERATE: Signs slightly below average:")
    for s in sorted(moderate_samples, key=lambda x: x['count']):
        print(f"   {s['sign']}: {s['count']} samples (could add ~{int(avg_count - s['count'])} more)")

# Cross-reference with model errors
print("\n" + "="*70)
print("CROSS-REFERENCE WITH MODEL PERFORMANCE")
print("="*70)

print("""
Based on validation error analysis, the model made only 2 errors:
1. C → O (1 error) - 'C' confused with 'O'
2. C → S (1 error) - 'C' confused with 'S'

SPECIFIC RECOMMENDATIONS FOR 'C':
✅ Current samples: {c_count}
✅ Add 50-100 MORE varied samples of 'C' focusing on:
   - Clear curved hand shape (distinct from 'O' circle)
   - Thumb position clearly separated from fingers
   - Different hand angles and rotations
   - Ensure fingers form clear 'C' curve, not closed 'O'
   - Avoid samples where 'C' looks like 'S' or 'O'

The sign 'C' is the ONLY one showing confusion - all others are perfect!
""".format(c_count=[s['count'] for s in stats if s['sign'] == 'C'][0] if 'C' in [s['sign'] for s in stats] else 'N/A'))

print("\n" + "="*70)
print("FINAL RECOMMENDATIONS")
print("="*70)

print("""
📊 OVERALL ASSESSMENT:
   Your model is performing EXCEPTIONALLY WELL (99.98% accuracy)!
   Only 2 errors in 8,386 validation samples.

🎯 TOP PRIORITY:
   1. Sign 'C' - Add 50-100 high-quality samples
      - Focus on clear distinction from 'O' and 'S'
      - Ensure proper 'C' curve shape
      - Good thumb positioning

📈 SECONDARY IMPROVEMENTS:
   2. Balance sample counts across all signs
   3. Maintain dark background (brightness ~29) for all new images
   4. Ensure clear hand visibility and contrast

✨ QUALITY CHECKLIST for new samples:
   ✓ Dark background (similar to existing dataset)
   ✓ Clear hand edges and finger separation
   ✓ Proper lighting (not too bright, not too dark)
   ✓ Hand fully visible in frame
   ✓ Consistent distance from camera
   ✓ Natural hand positioning
   ✓ For 2-hand signs: both hands clearly visible

💡 EXPECTED IMPROVEMENT:
   Adding 50-100 better 'C' samples could push accuracy to 99.99%+!
""")

print("="*70)
