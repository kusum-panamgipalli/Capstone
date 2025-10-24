# ISL Model Training Guide

This guide will help you train the Indian Sign Language recognition model from scratch.

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Kaggle account for dataset download
- Internet connection

## Step-by-Step Instructions

### 1. Install Required Packages

```bash
pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn matplotlib pillow kaggle tensorflowjs tqdm
```

**Package Versions (tested):**
- tensorflow: 2.14.0
- opencv-python: 4.8.1
- mediapipe: 0.10.8
- numpy: 1.24.3
- scikit-learn: 1.3.2

### 2. Set Up Kaggle API

To download the dataset automatically, you need Kaggle API credentials:

1. Go to https://www.kaggle.com/account
2. Scroll to the "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Place it in:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
6. Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

### 3. Download ISL Dataset

Run the download script:

```bash
python 1_download_dataset.py
```

**What happens:**
- Downloads Indian Sign Language dataset from Kaggle
- Extracts images to `./data/raw/`
- Dataset contains images for A-Z and 0-9

**Expected output:**
```
Found X images across 36 classes
Downloaded to: ./data/raw/
```

**If automatic download fails:**
1. Go to: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
2. Click "Download" button
3. Extract ZIP file
4. Copy contents to `./data/raw/`

### 4. Extract Hand Landmarks

Run the landmark extraction script:

```bash
python 2_extract_landmarks.py
```

**What happens:**
- Processes each image using MediaPipe Hands
- Extracts 21 hand landmarks (x, y, z coordinates)
- Saves processed data to `./data/processed/`
- Creates:
  - `X_landmarks.npy` - Feature data (63 features per sample)
  - `y_labels.npy` - Label data
  - `label_mapping.json` - Class mappings
  - `landmarks_dataset.csv` - For analysis

**Processing time:** 5-15 minutes depending on dataset size

**Expected output:**
```
Successfully processed: XXXX images
Dataset shape: (XXXX, 63)
Classes: A, B, C, ..., Z, 0, 1, ..., 9
```

### 5. Train the Model

Run the training script:

```bash
python 3_train_model.py
```

**What happens:**
- Loads processed landmark data
- Splits into train/validation/test sets (70/10/20)
- Normalizes features using StandardScaler
- Trains a deep neural network
- Saves best model based on validation accuracy
- Generates training plots

**Training time:** 10-30 minutes depending on hardware

**Model architecture:**
```
Input (63 features)
├── Dense(256) + BatchNorm + Dropout(0.3)
├── Dense(128) + BatchNorm + Dropout(0.3)
├── Dense(64) + BatchNorm + Dropout(0.2)
├── Dense(32) + BatchNorm + Dropout(0.2)
└── Output(36) softmax
```

**Training hyperparameters:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch size: 32
- Max epochs: 100
- Early stopping: patience=15
- Learning rate reduction: factor=0.5, patience=5

**Expected output:**
```
Test Accuracy: XX.XX%
Training plots saved to: ./models/training_history.png
Model saved to: ./models/
```

**Monitoring training:**
- Watch validation accuracy (should increase)
- Watch validation loss (should decrease)
- Training will stop early if no improvement
- Best model is automatically saved

### 6. Convert Model to TensorFlow.js

Run the conversion script:

```bash
python 4_convert_to_tfjs.py
```

**What happens:**
- Converts TensorFlow model to TensorFlow.js format
- Quantizes to float16 to reduce size
- Copies model to extension folder
- Creates JavaScript configuration file

**Output files in `../isl-interpreter-extension/models/`:**
- `model.json` - Model architecture
- `group1-shard*.bin` - Model weights
- `model-config.js` - Label mappings and config
- `scaler.pkl` - Feature normalizer
- `test-model.html` - Browser test file

**Expected output:**
```
Model converted successfully!
Generated files:
  model.json (XX KB)
  group1-shard1of1.bin (XX KB)
  model-config.js (XX KB)
```

### 7. Test the Model

Open `test-model.html` in Chrome:

```bash
# Navigate to extension models folder
cd ../isl-interpreter-extension/models
# Open test-model.html in Chrome
```

Or simply open the file in Chrome browser.

**Test steps:**
1. Click "Load Model"
2. Should see: "✓ Model loaded successfully!"
3. Click "Test Prediction"
4. Should see prediction with confidence

## Troubleshooting

### Issue: Kaggle API not working
**Solution:**
- Verify kaggle.json is in correct location
- Check file permissions (should be 600 on Linux/Mac)
- Ensure you have accepted competition rules on Kaggle

### Issue: Out of memory during training
**Solution:**
- Reduce batch size in `3_train_model.py` (line ~150)
- Close other applications
- Use smaller model (reduce layer sizes)

### Issue: Low accuracy (<70%)
**Possible causes:**
- Insufficient training data
- Poor quality images
- Need more training epochs
- Model architecture too simple

**Solutions:**
- Collect more training data
- Increase epochs to 150-200
- Add data augmentation
- Try different model architectures

### Issue: TensorFlow.js conversion fails
**Solution:**
- Ensure tensorflowjs is installed: `pip install tensorflowjs`
- Check TensorFlow version compatibility
- Verify model trained successfully

### Issue: Model file too large
**Solution:**
- Model is already quantized to float16
- Consider pruning if still too large
- Use model compression techniques

## Expected Results

**Dataset:**
- ~40,000+ images across 36 classes
- ~1,100+ images per class (varies)

**Model Performance:**
- Training Accuracy: >95%
- Validation Accuracy: >90%
- Test Accuracy: >85%

**Model Size:**
- Total: ~500KB-2MB (after quantization)
- Fast inference: <50ms per prediction

## Tips for Better Results

1. **Data Quality:**
   - Ensure images are clear and well-lit
   - Consistent hand positioning
   - Remove noisy/incorrect labels

2. **Data Augmentation:**
   - Add rotation, scaling, translation
   - Adjust brightness/contrast
   - Simulate different lighting conditions

3. **Model Tuning:**
   - Experiment with layer sizes
   - Try different dropout rates
   - Adjust learning rate

4. **Training Optimization:**
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Implement learning rate scheduling

## Next Steps

After successful training:

1. **Test in Extension:**
   - Load extension in Chrome
   - Join a Google Meet call
   - Enable interpreter
   - Test with real hand signs

2. **Fine-tune:**
   - Collect problematic cases
   - Retrain with additional data
   - Adjust confidence thresholds

3. **Deploy:**
   - Package extension for distribution
   - Share with users
   - Collect feedback

## File Structure After Training

```
model-training/
├── data/
│   ├── raw/                    # Original images from Kaggle
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── processed/              # Processed landmark data
│       ├── X_landmarks.npy
│       ├── y_labels.npy
│       ├── label_mapping.json
│       └── landmarks_dataset.csv
│
└── models/                     # Trained models
    ├── best_model.h5           # Best model (H5 format)
    ├── isl_model.h5            # Final model (H5 format)
    ├── isl_model_saved/        # SavedModel format
    ├── scaler.pkl              # Feature scaler
    ├── model_metadata.json     # Model information
    └── training_history.png    # Training plots
```

## Support

If you encounter issues:
1. Check error messages carefully
2. Review troubleshooting section
3. Check Python/package versions
4. Open an issue on GitHub

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MediaPipe Hands Guide](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)
- [Kaggle ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
