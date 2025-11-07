# ISL Real-time Interpreter - Chrome Extension

A real-time Indian Sign Language (ISL) interpreter for video conferencing platforms like Google Meet and Zoom. The extension captures hand gestures through your camera, processes them using MediaPipe and a trained neural network, and displays translated text as captions in real-time.

## 🌟 Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands for accurate hand landmark detection
- **AI-Powered Recognition**: TensorFlow.js model trained on ISL dataset (A-Z, 0-9)
- **Platform Support**: Works on Google Meet and Zoom
- **Live Captions**: Displays translated sign language as overlay text
- **Lightweight**: Runs entirely in the browser with no server needed
- **Privacy-Focused**: All processing happens locally on your device

## 📋 Prerequisites

### For Using the Extension:
- Google Chrome browser (v88 or higher)
- Working webcam
- Internet connection (for loading MediaPipe and TensorFlow.js libraries)

### For Training the Model:
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- Kaggle account (for dataset download)

## 🚀 Quick Start

### Option 1: Use Pre-trained Model

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kusum-panamgipalli/Capstone.git
   cd Capstone
   ```

2. **Load the extension in Chrome**:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (toggle in top-right corner)
   - Click "Load unpacked"
   - Select the `isl-interpreter-extension` folder

3. **Use the extension**:
   - Join a Google Meet or Zoom call
   - Click the extension icon in Chrome toolbar
   - Click "Enable Interpreter"
   - Position your hand clearly in front of the camera
   - Sign language translations will appear as overlay text

### Option 2: Train Your Own Model

If you want to train the model with the ISL dataset:

1. **Set up Python environment**:
   ```bash
   cd model-training
   pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn matplotlib pillow kaggle tensorflowjs tqdm
   ```

2. **Download the dataset**:
   ```bash
   python 1_download_dataset.py
   ```
   
   *Note: You'll need to set up Kaggle API credentials first. Follow instructions in the script.*

3. **Extract hand landmarks**:
   ```bash
   python 2_extract_landmarks.py
   ```

4. **Train the model**:
   ```bash
   python 3_train_model.py
   ```

5. **Convert to TensorFlow.js**:
   ```bash
   python 4_convert_to_tfjs.py
   ```

6. **Load the extension** (follow steps from Option 1)

## 📁 Project Structure

```
Capstone/
├── isl-interpreter-extension/     # Chrome extension files
│   ├── manifest.json              # Extension configuration
│   ├── popup.html/js/css          # Extension popup UI
│   ├── content.js                 # Content script (injected into pages)
│   ├── background.js              # Background service worker
│   ├── video-processor.js         # MediaPipe + TensorFlow.js integration
│   ├── mediapipe-hands.js         # MediaPipe Hands configuration
│   ├── models/                    # Trained model files (TensorFlow.js)
│   │   ├── model.json
│   │   ├── model-config.js
│   │   └── *.bin (weight files)
│   └── icons/                     # Extension icons
│
├── model-training/                # Python training scripts
│   ├── 1_download_dataset.py      # Download ISL dataset from Kaggle
│   ├── 2_extract_landmarks.py     # Extract hand landmarks using MediaPipe
│   ├── 3_train_model.py           # Train neural network classifier
│   ├── 4_convert_to_tfjs.py       # Convert model to TensorFlow.js
│   ├── data/                      # Dataset storage
│   │   ├── raw/                   # Raw images from Kaggle
│   │   └── processed/             # Processed landmark data
│   └── models/                    # Trained model files
│
└── README.md                      # This file
```

## 🎯 How It Works

1. **Video Capture**: The extension accesses the video stream from Google Meet/Zoom
2. **Hand Detection**: MediaPipe Hands detects and tracks 21 hand landmarks in real-time
3. **Feature Extraction**: 3D coordinates (x, y, z) of all 21 landmarks are extracted (63 features)
4. **Classification**: TensorFlow.js model predicts the ISL sign (A-Z, 0-9)
5. **Display**: Recognized sign is displayed as overlay text on the video call

## 🧠 Model Architecture

- **Input**: 63 features (21 landmarks × 3 coordinates)
- **Architecture**: Dense Neural Network
  - Dense(256) + BatchNorm + Dropout(0.3)
  - Dense(128) + BatchNorm + Dropout(0.3)
  - Dense(64) + BatchNorm + Dropout(0.2)
  - Dense(32) + BatchNorm + Dropout(0.2)
  - Dense(36, softmax) # 26 letters + 10 digits
- **Training**: Adam optimizer, categorical crossentropy loss
- **Dataset**: Indian Sign Language (ISL) from Kaggle

## 📊 Dataset

The model is trained on the [Indian Sign Language (ISL) dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) which contains:
- Images for all English alphabets (A-Z)
- Images for digits (0-9)
- Multiple samples per class for robust training

## 🔧 Configuration

### Extension Settings (via popup):
- **Show Confidence Scores**: Toggle confidence display
- **Processing Speed**: Adjust FPS (15, 20, or 30)

### MediaPipe Settings (in `video-processor.js`):
```javascript
modelComplexity: 1,           // 0 (lite) or 1 (full)
minDetectionConfidence: 0.5,  // 0.0 to 1.0
minTrackingConfidence: 0.5,   // 0.0 to 1.0
maxNumHands: 1                // Number of hands to detect
```

## 🐛 Troubleshooting

### Extension not working:
- Check if camera permissions are granted
- Ensure you're on Google Meet or Zoom
- Check browser console for errors (F12)

### No hands detected:
- Ensure good lighting
- Keep hand clearly visible in frame
- Avoid cluttered background

### Low accuracy:
- Sign clearly with good hand positioning
- Ensure proper lighting conditions
- Try retraining model with more data

### Model not loading:
- Check internet connection (for CDN libraries)
- Verify model files exist in `models/` folder
- Check console for specific errors

## 🛠️ Development

### Testing locally:
1. Make changes to extension files
2. Go to `chrome://extensions/`
3. Click reload icon on the ISL Interpreter extension
4. Test on a Google Meet call

### Debugging:
- Open Chrome DevTools (F12) on the meeting page
- Check Console tab for logs
- Use `window.islDebug` object for debugging:
  ```javascript
  window.islDebug.debugMediaPipe()
  window.islDebug.toggleLandmarks()
  ```

## 📝 TODO / Future Enhancements

- [ ] Add support for more ISL signs (words, phrases)
- [ ] Implement smoothing/filtering for more stable predictions
- [ ] Add gesture history and sentence building
- [ ] Support for two-handed signs
- [ ] Add settings page with advanced configuration
- [ ] Implement user feedback mechanism for improving accuracy
- [ ] Add language translation support
- [ ] Create mobile app version

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for hand tracking
- [TensorFlow.js](https://www.tensorflow.org/js) for in-browser ML
- [ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) on Kaggle
- Indian Sign Language community

## 📧 Contact

Project Link: [https://github.com/kusum-panamgipalli/Capstone](https://github.com/kusum-panamgipalli/Capstone)

## 🎥 Demo

[Add demo video or GIF here]

---

**Note**: This is an educational project and may not achieve 100% accuracy. Professional sign language interpretation requires trained interpreters. This tool is meant to assist and supplement, not replace, human interpreters.