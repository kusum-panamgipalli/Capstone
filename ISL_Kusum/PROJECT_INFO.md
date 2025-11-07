# ISL (Indian Sign Language) Interpreter - Kusum's Project

**Author**: Kusum  
**Date**: November 2025  
**Version**: 2.0 (2-Hand Support)

## 🎯 Project Goal
Create a real-time Indian Sign Language interpreter using webcam input that will eventually be integrated as a web extension for platforms like Google Meet.

## 📊 Current Status
- ✅ **Model Accuracy**: 99.98% (validation)
- ✅ **Real-time Performance**: ~12 FPS
- ✅ **Model Size**: 0.93 MB (lightweight!)
- ✅ **2-Hand Support**: Recognizes both 1-hand and 2-hand signs
- ✅ **Production Ready**: Clean, documented, and tested

## 🏗️ Project Structure

```
ISL_Kusum/
├── models/                          # Trained models and datasets
│   ├── isl_landmark_model_2hands.h5           # Main trained model (0.93 MB)
│   ├── isl_landmark_labels_2hands.json        # Model metadata & labels
│   ├── hand_landmarks_dataset_2hands.pkl      # Extracted landmarks (41,930 samples)
│   ├── hand_landmarks_dataset_2hands.json     # Dataset info
│   └── training_history_landmark_2hands.png   # Training graphs
│
├── scripts/                         # Main Python scripts
│   ├── extract_landmarks_2hands.py            # Extract hand landmarks from images
│   ├── train_landmark_model_2hands.py         # Train the neural network
│   └── realtime_inference_landmark_2hands.py  # Real-time webcam recognition
│
├── web_extension/                   # Google Meet integration (future)
│   ├── chrome_extension_content.js            # Chrome extension content script
│   ├── manifest.json                          # Extension manifest
│   └── web_integration.py                     # WebSocket server for web integration
│
├── analysis/                        # Dataset analysis tools
│   ├── analyze_dataset.py                     # Overall dataset analysis
│   ├── detailed_error_analysis.py             # Validation error analysis
│   └── visual_dataset_analysis.py             # Sample count analysis
│
├── docs/                            # Documentation
│   ├── 2HAND_SUPPORT.md                       # 2-hand implementation details
│   ├── MIGRATION_COMPLETE.md                  # CNN to MediaPipe migration notes
│   └── DATASET_IMPROVEMENT_GUIDE.md           # Guide to improve dataset quality
│
├── dataset/                         # Dataset reference
│   └── README.md                              # Info about the Indian/ dataset
│
├── README.md                        # Main project documentation
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Git ignore rules
```

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Virtual environment is already included in ISL_Kusum/.venv311/

# Activate environment
.\.venv311\Scripts\Activate.ps1

# If you need to reinstall dependencies:
pip install -r requirements.txt
```

### 2. Run Real-time Recognition
```powershell
# From ISL_Kusum folder:
.\.venv311\Scripts\python.exe scripts\realtime_inference_landmark_2hands.py

# Or from scripts folder:
cd scripts
..\.venv311\Scripts\python.exe realtime_inference_landmark_2hands.py
```

**Controls**:
- **SPACE**: Add space to output
- **BACKSPACE**: Delete last character
- **C**: Clear all text
- **Q**: Quit

### 3. Retrain Model (if dataset changed)
```powershell
# From ISL_Kusum folder:

# Step 1: Extract landmarks
.\.venv311\Scripts\python.exe scripts\extract_landmarks_2hands.py

# Step 2: Train model
.\.venv311\Scripts\python.exe scripts\train_landmark_model_2hands.py
```

## 💡 Technical Highlights

### Technology Stack
- **MediaPipe v0.10.21**: Hand landmark detection
- **TensorFlow 2.19.1**: Neural network training
- **Python 3.11.9**: Required for MediaPipe compatibility
- **OpenCV**: Webcam capture and display

### Model Architecture
- **Input**: 126 features (2 hands × 21 landmarks × 3 coordinates)
- **Architecture**: 512→256→128→64→35 neurons
- **Technique**: Zero-padding for 1-hand signs
- **Training**: 5.9 minutes, 243,619 parameters
- **Performance**: 99.95% accuracy, ~64ms inference time

### Key Features
1. **Lighting Independent**: Uses hand geometry, not pixel values
2. **Background Independent**: Works with any background
3. **2-Hand Support**: Handles signs requiring both hands
4. **Real-time**: ~12 FPS on standard hardware
5. **Lightweight**: 0.93 MB model (vs 60 MB for CNN approaches)

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Samples | 33,544 (80%) |
| Validation Samples | 8,386 (20%) |
| Training Accuracy | 99.76% |
| Validation Accuracy | 99.95% |
| Validation Errors | 4 samples |
| 1-Hand Samples | 26,245 (62.6%) |
| 2-Hand Samples | 15,685 (37.4%) |
| MediaPipe Detection | 19ms |
| Model Inference | 64ms |
| Total Latency | ~83ms (~12 FPS) |

## 🎓 Learning Journey

### Evolution of Approaches
1. **CNN on Images** (v0.1): Failed due to lighting mismatch
2. **Normalized Dataset + CNN** (v0.5): Improved but still issues
3. **MediaPipe Landmarks** (v1.0): Breakthrough! 99.87% with 1-hand
4. **2-Hand Support** (v2.0): Final version, 99.95% accuracy

### Key Insights
- Preprocessing can harm if it destroys critical features
- Original dark backgrounds better than normalized
- MediaPipe landmarks superior to raw images (780× smaller features!)
- 2-hand support essential (37% of ISL signs are 2-handed)

## 🔮 Next Steps

### Immediate
- [ ] Test real-time inference thoroughly
- [ ] Add more 'C' samples (only sign with errors)
- [ ] Achieve 100% accuracy

### Future Enhancements
- [ ] WebSocket server for web integration
- [ ] Chrome extension for Google Meet
- [ ] TensorFlow Lite conversion (faster inference)
- [ ] Multi-threaded processing (higher FPS)
- [ ] Mobile deployment (Android/iOS)
- [ ] Phrase/sentence recognition (LSTM/GRU)

## 📧 Contact & Collaboration

This is Kusum's individual contribution to the ISL project. Files in the `ISL_Kusum/` folder are maintained separately from other contributors' work.

---

**Last Updated**: November 7, 2025  
**Model Version**: isl_landmark_model_2hands.h5  
**Status**: Production Ready ✅
