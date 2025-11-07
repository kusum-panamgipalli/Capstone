"""
Train ISL model on hand landmarks - NOW SUPPORTS 2-HAND SIGNS!
Input: 126 features (2 hands × 21 landmarks × 3 coordinates)
- If only 1 hand: second hand padded with zeros
- If 2 hands: full 126 features used
"""
import numpy as np
import pickle
import json
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

def create_landmark_model_2hands(input_dim, num_classes):
    """
    Create neural network for 1-hand AND 2-hand landmark classification
    Uses same architecture but with 126 input features
    """
    model = keras.Sequential([
        # Input layer - now accepts 126 features (2 hands)
        keras.Input(shape=(input_dim,)),
        
        # First dense block - larger to handle more features
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second dense block
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Fourth dense block
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

print("="*70)
print("TRAIN LANDMARK-BASED ISL MODEL - 2-HAND SUPPORT!")
print("="*70)

# Load dataset
print("\nLoading landmark dataset...")
try:
    with open('../models/hand_landmarks_dataset_2hands.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    X = dataset['X']
    y = dataset['y']
    hand_counts = dataset['hand_counts']
    class_names = dataset['class_names']
    
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"✓ Classes: {len(class_names)}")
    print(f"✓ 1-hand samples: {np.sum(hand_counts == 1)}")
    print(f"✓ 2-hand samples: {np.sum(hand_counts == 2)}")
    
except FileNotFoundError:
    print("❌ Error: hand_landmarks_dataset_2hands.pkl not found!")
    print("   Please run extract_landmarks_2hands.py first")
    exit(1)

# Split data
print("\nSplitting dataset (80% train, 20% validation)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")

# Convert labels to categorical
num_classes = len(class_names)
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

# Create model
print("\nCreating neural network model...")
model = create_landmark_model_2hands(input_dim=X.shape[1], num_classes=num_classes)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print summary
print("\nModel Summary:")
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\n✓ Total trainable parameters: {total_params:,}")
print(f"  Compare to 1-hand model: 61,731 parameters")
print(f"  Compare to image CNN: 3,417,155 parameters")

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        '../models/isl_landmark_model_2hands.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("Training on 1-hand AND 2-hand landmark coordinates...")
print("="*70 + "\n")

start_time = datetime.now()

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_duration = (end_time - start_time).total_seconds()

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)

print(f"\n✓ Validation Accuracy: {val_acc*100:.2f}%")

# Detailed evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print(f"\nDetailed Classification Report:")
print(classification_report(y_val, y_pred_classes, target_names=class_names, zero_division=0))

# Find misclassified samples
misclassified = np.where(y_val != y_pred_classes)[0]
print(f"\nMisclassified samples: {len(misclassified)}/{len(y_val)}")

if len(misclassified) > 0:
    print(f"\nMisclassified examples:")
    for idx in misclassified[:10]:  # Show first 10
        true_class = class_names[y_val[idx]]
        pred_class = class_names[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]] * 100
        print(f"  Sample {idx}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.1f}%")

# Plot training history
print(f"\nGenerating training history plot...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy (2-Hand Support)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (2-Hand Support)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../models/training_history_landmark_2hands.png', dpi=150, bbox_inches='tight')
print(f"✓ Training history saved as 'training_history_landmark_2hands.png'")

# Save model info
model_info = {
    'class_names': class_names,
    'feature_count': int(X.shape[1]),
    'num_classes': len(class_names),
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'train_accuracy': float(train_acc),
    'val_accuracy': float(val_acc),
    'total_params': int(total_params),
    'training_duration_minutes': float(training_duration/60),
    'mean': dataset['mean'].tolist(),
    'std': dataset['std'].tolist(),
    'supports_2_hands': True,
    'trained_on': datetime.now().isoformat()
}

with open('../models/isl_landmark_labels_2hands.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"✓ Model info saved as 'isl_landmark_labels_2hands.json'")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

print(f"\n✓ Model saved as: isl_landmark_model_2hands.h5")
print(f"✓ Labels saved as: isl_landmark_labels_2hands.json")
print(f"✓ Final validation accuracy: {val_acc*100:.2f}%")
print(f"✓ Training duration: {training_duration/60:.1f} minutes")
print(f"✓ Model size: ~{total_params*4/1024/1024:.2f} MB")

print(f"\n" + "="*70)
print("ADVANTAGES OF 2-HAND LANDMARK MODEL:")
print("="*70)
print("✅ Supports both 1-hand and 2-hand signs")
print("✅ Automatically adapts to number of hands detected")
print("✅ Still very fast inference (~0.2ms vs 0.1ms for 1-hand)")
print("✅ Still much smaller than image CNN (~1-2MB vs 60MB)")
print("✅ Lighting and background independent")
print("✅ Ready for complex 2-hand ISL signs!")

print(f"\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Real-time inference: python realtime_inference_landmark_2hands.py")
print("2. This will detect and process up to 2 hands in real-time!")
print("="*70)
