"""
Train Neural Network model for ISL recognition
Uses extracted hand landmarks to train a classifier
"""

import os
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class ISLModelTrainer:
    def __init__(self, data_dir='./data/processed', model_dir='./models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        """Load processed landmark data"""
        print("\n" + "="*60)
        print("LOADING PROCESSED DATA")
        print("="*60)
        
        # Load data
        X = np.load(os.path.join(self.data_dir, 'X_landmarks.npy'))
        y = np.load(os.path.join(self.data_dir, 'y_labels.npy'))
        
        # Load label mappings
        with open(os.path.join(self.data_dir, 'label_mapping.json'), 'r') as f:
            label_info = json.load(f)
        
        self.label_to_idx = label_info['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in label_info['idx_to_label'].items()}
        self.num_classes = label_info['num_classes']
        
        print(f"\n✓ Loaded data successfully")
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {', '.join(sorted(self.label_to_idx.keys()))}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split and preprocess data"""
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {X_train.shape[0]} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {X_val.shape[0]} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Testing: {X_test.shape[0]} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Normalize features
        print("\nNormalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        print("✓ Preprocessing complete")
        
        return (X_train_scaled, y_train_cat, y_train), \
               (X_val_scaled, y_val_cat, y_val), \
               (X_test_scaled, y_test_cat, y_test)
    
    def build_model(self, input_shape):
        """Build neural network model"""
        print("\n" + "="*60)
        print("BUILDING MODEL ARCHITECTURE")
        print("="*60)
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # First dense block
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second dense block
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third dense block
            layers.Dense(64, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth dense block (optional, for better feature learning)
            layers.Dense(32, activation='relu', name='dense_4'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n✓ Model built successfully")
        print(f"\nModel Summary:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\nTraining parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        print("\nStarting training...\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test, y_test_original):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy*100:.2f}%")
        
        # Per-class accuracy
        predictions = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        print("\nPer-class Accuracy:")
        for class_idx in range(self.num_classes):
            class_mask = y_test_original == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred[class_mask] == y_test_original[class_mask])
                class_name = self.idx_to_label[class_idx]
                print(f"  {class_name}: {class_acc*100:.2f}%")
        
        return test_accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        print("\nGenerating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(plot_path)
        print(f"✓ Training plots saved to: {plot_path}")
        
    def save_model(self):
        """Save model and related files"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Save model in multiple formats
        
        # 1. Keras format (.h5)
        model_path_h5 = os.path.join(self.model_dir, 'isl_model.h5')
        self.model.save(model_path_h5)
        print(f"✓ Saved model (H5): {model_path_h5}")
        
        # 2. SavedModel format (for TensorFlow.js conversion)
        model_path_saved = os.path.join(self.model_dir, 'isl_model_saved')
        self.model.save(model_path_saved)
        print(f"✓ Saved model (SavedModel): {model_path_saved}")
        
        # 3. Save scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # 4. Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'input_shape': self.model.input_shape[1],
            'model_architecture': 'Dense Neural Network',
            'feature_type': 'MediaPipe Hand Landmarks (63 features: 21 landmarks x 3 coordinates)'
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
        
        print("\n✓ All model files saved successfully!")

def main():
    print("\n" + "="*60)
    print("ISL MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = ISLModelTrainer(
        data_dir='./data/processed',
        model_dir='./models'
    )
    
    # Load data
    X, y = trainer.load_data()
    
    # Preprocess data
    (X_train, y_train_cat, y_train), \
    (X_val, y_val_cat, y_val), \
    (X_test, y_test_cat, y_test) = trainer.preprocess_data(X, y)
    
    # Build model
    trainer.build_model(input_shape=X_train.shape[1])
    
    # Train model
    trainer.train_model(
        X_train, y_train_cat,
        X_val, y_val_cat,
        epochs=100,
        batch_size=32
    )
    
    # Evaluate model
    test_acc = trainer.evaluate_model(X_test, y_test_cat, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    print("\nNext step: Run 4_convert_to_tfjs.py to convert model for browser use")

if __name__ == "__main__":
    main()
