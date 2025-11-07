# ISL Interpreter - 2-Hand Support 🤚🤚# Indian Sign Language (ISL) Real-time Interpreter



Real-time Indian Sign Language interpreter using **MediaPipe hand landmarks** and neural networks.A state-of-the-art real-time Indian Sign Language interpreter using **MediaPipe hand landmarks** and deep learning for accurate sign recognition.



## ✨ Features## 🎯 Features



- **99.95% Accuracy** - Only 4 errors in 8,386 validation samples- **Real-time Recognition**: Instant translation of ISL signs from webcam

- **2-Hand Support** - Recognizes both 1-handed and 2-handed signs- **35 Classes**: Recognizes digits (1-9) and letters (A-Z)

- **Ultra Fast** - ~12 FPS real-time recognition- **MediaPipe Integration**: Uses Google's MediaPipe for precise hand landmark detection

- **Lighting Independent** - Works in any lighting condition- **Landmark-Based Learning**: Trains on hand geometry (21 landmarks × 3D coordinates) instead of raw images

- **Small Model** - Only 0.93 MB model size- **Ultra-Fast**: ~10ms total latency (10x faster than image-based models)

- **41,930 Training Samples** - Trained on comprehensive ISL dataset- **Lighting Independent**: Works in any lighting condition

- **Background Independent**: Ignores background clutter

## 🚀 Quick Start- **High Accuracy**: 95%+ accuracy with geometric feature learning

- **Optimized Model**: Quantized TFLite model (5MB) for web deployment

### Prerequisites- **User-friendly Interface**: Visual feedback with hand skeleton overlay

- Python 3.11 (MediaPipe requires 3.11 or 3.12)- **Translation Accumulation**: Builds sentences as you sign

- Webcam- **Web-ready**: Designed for Google Meet integration



### Installation## 📁 Project Structure



1. **Activate virtual environment:**```

```powershellISL/

.venv311\Scripts\Activate.ps1├── Indian_Normalized/       # Brightness-normalized dataset (42,745 images)

```│   ├── A/                   # ~1200 images per class

│   ├── B/

2. **Install dependencies (if needed):**│   ├── ...

```powershell│   └── Z/

pip install -r requirements.txt├── extract_landmarks.py     # Extract hand landmarks from images (MediaPipe)

```├── train_landmark_model.py  # Train on landmark coordinates

├── train_normalized_model.py # Train CNN on normalized images (alternative)

### Run the Interpreter├── realtime_inference_landmark.py  # Real-time with landmarks (FASTEST)

├── realtime_inference_quantized.py # Quantized CNN inference

```powershell├── quantize_model.py        # Convert model to TFLite

python realtime_inference_landmark_2hands.py├── web_integration.py       # WebSocket server for Google Meet

```├── requirements.txt         # Python dependencies

├── isl_landmark_model.h5    # Landmark-based model (generated)

**Controls:**├── isl_model_quantized.tflite # Quantized model (generated)

- `SPACE` - Add space to translation└── README.md               # This file

- `BACKSPACE` - Delete last character```

- `C` - Clear translation

- `Q` - Quit## 🚀 Quick Start



## 📁 Project Structure### ⚠️ Prerequisites



```**Python 3.11 or 3.12 Required** (MediaPipe doesn't support 3.13 yet)

ISL/

├── .venv311/                                   # Python 3.11 virtual environmentSee [SETUP_MEDIAPIPE.md](SETUP_MEDIAPIPE.md) for detailed Python setup instructions.

├── Indian/                                     # Original training dataset (42,745 images)

├── extract_landmarks_2hands.py                 # Extract hand landmarks from images### 1. Install Dependencies

├── train_landmark_model_2hands.py              # Train the 2-hand model

├── realtime_inference_landmark_2hands.py       # Real-time webcam inference```bash

├── isl_landmark_model_2hands.h5                # Trained model (0.93 MB, 99.95% accuracy)# Ensure you're using Python 3.11 or 3.12

├── isl_landmark_labels_2hands.json             # Model labels and normalization paramspython --version  # Should show 3.11.x or 3.12.x

├── hand_landmarks_dataset_2hands.pkl           # Extracted landmarks (41,930 samples)

├── hand_landmarks_dataset_2hands.json          # Dataset metadata# Install requirements (includes MediaPipe)

├── training_history_landmark_2hands.png        # Training accuracy/loss plotpip install -r requirements.txt

├── web_integration.py                          # WebSocket server for Google Meet```

├── chrome_extension_content.js                 # Chrome extension (future)

├── manifest.json                               # Extension manifest (future)### 2. Extract Hand Landmarks

└── requirements.txt                            # Python dependencies

``````bash

python extract_landmarks.py

## 🧠 How It Works```



### 1. Hand Detection (MediaPipe)This will:

- Detects up to **2 hands** simultaneously- Use MediaPipe to detect hands in ~42,000 images

- Extracts **21 landmarks** per hand (42 total)- Extract 21 hand landmarks (63 features: x, y, z coordinates)

- Each landmark has **x, y, z coordinates** (126 features total)- Normalize features for training

- Save as `hand_landmarks_dataset.pkl`

### 2. Landmark Processing- Expected time: 10-15 minutes

- Normalizes coordinates (zero mean, unit variance)

- Pads with zeros if only 1 hand detected### 3. Train the Landmark Model

- Works for both 1-hand signs (1-9, L, V...) and 2-hand signs (N, M, X...)

```bash

### 3. Neural Network Classificationpython train_landmark_model.py

- **Architecture:** 512→256→128→64→35 neurons```

- **Parameters:** 243,619 (only 0.93 MB!)

- **Accuracy:** 99.95% validation accuracyThis will:

- **Speed:** ~0.2ms inference time- Train a neural network on hand landmark coordinates

- Much faster than training on images (5-10 minutes vs 30-60 minutes)

### 4. Real-Time Recognition- Creates a smaller, faster model (~1-2MB vs 60MB)

- **MediaPipe extraction:** ~19ms- Save the best model as `isl_landmark_model.h5`

- **Model inference:** ~64ms- Expected accuracy: 95%+

- **Total:** ~83ms per frame (~12 FPS)

### 4. Run Real-time Inference

## 📊 Model Performance

```bash

### Training Resultspython realtime_inference_landmark.py

- **Training samples:** 33,544```

- **Validation samples:** 8,386

- **Classes:** 35 (1-9, A-Z)This provides:

- **Validation accuracy:** 99.95%- **Ultra-fast recognition**: ~10ms total latency

- **Misclassifications:** Only 4 errors- **Hand skeleton visualization**: See 21 landmarks in real-time

- **Training time:** 5.9 minutes- **High accuracy**: Geometric feature learning

- **Robust performance**: Works in any lighting/background

### Dataset Composition

- **Total samples:** 41,930## 🎮 Controls

- **1-hand signs:** 26,245 (62.6%)

- **2-hand signs:** 15,685 (37.4%)| Key | Action |

- **Success rate:** 98.1% (from 42,745 original images)|-----|--------|

| **Q** | Quit the application |

### Classes with 2-Hand Signs| **SPACE** | Add space to translation |

- **100% 2-hand:** A, D, E, X| **BACKSPACE** | Delete last character |

- **High 2-hand:** N (97.6%), M (99.3%), K (96.2%), T (98.2%), Y (80.7%)| **C** | Clear all translation text |

- **Mostly 1-hand:** 1-9, B, C, H, I, J, L, O, Q, R, S, U, V, W, Z| **S** | Save translation to file |



## 🔄 Re-training the Model## 🧠 Model Architecture



### Extract Landmarks### Landmark-Based Model (Recommended) ⭐

```powershell

python extract_landmarks_2hands.pyUses **MediaPipe Hand Landmarks** + **Dense Neural Network**:

```

- Processes all images in `Indian/` folder- **Input**: 63 features (21 landmarks × x, y, z coordinates)

- Detects up to 2 hands per image- **Architecture**: 256 → 128 → 64 → 32 → 35 neurons

- Creates `hand_landmarks_dataset_2hands.pkl`- **Batch Normalization**: Faster training and better generalization

- Takes ~35 minutes for 42,745 images- **Dropout Layers**: Prevents overfitting

- **Total Parameters**: ~100K (34x smaller than CNN!)

### Train Model- **Model Size**: ~1-2MB vs 60MB image-based

```powershell- **Inference Time**: ~0.1ms vs 5ms quantized CNN

python train_landmark_model_2hands.py

```### Why Landmark-Based is Superior:

- Trains on extracted landmarks

- 80/20 train/validation split✅ **Lighting Independent**: Coordinates don't change with brightness

- Creates `isl_landmark_model_2hands.h5`✅ **Background Independent**: Only hand geometry matters

- Takes ~6 minutes to train✅ **Scale/Rotation Invariant**: Relative positions stay consistent

✅ **10x Faster**: 63 features vs 49,152 pixels

## 🌐 Web Integration (Future)✅ **Better Generalization**: Learns hand shape, not appearance

✅ **Industry Standard**: Used by Google, Apple, Snapchat, TikTok

For Google Meet integration:

### Training Features:

1. **Start WebSocket server:**- MediaPipe hand detection (8-10ms per image)

```powershell- Feature normalization (zero mean, unit variance)

python web_integration.py- 80/20 train/validation split with stratification

```- Early stopping with best model restoration

- Learning rate reduction on plateau

2. **Install Chrome extension:**

- Load `manifest.json` in Chrome## 📊 Expected Performance

- Extension receives real-time translations

### Landmark-Based Model:

## 📈 Why MediaPipe Landmarks?- **Training Accuracy**: ~95-98%

- **Validation Accuracy**: ~95-97%

### vs Image-Based CNN- **Real-time FPS**: 60-100 FPS

- ✅ **780x smaller features:** 126 vs 49,152 (128×128×3)- **Total Latency**: ~10ms (MediaPipe 10ms + inference 0.1ms)

- ✅ **Lighting independent:** Coordinates, not pixels- **Model Size**: ~1-2MB

- ✅ **Background independent:** Only hand geometry- **Works in**: Any lighting, any background

- ✅ **Faster inference:** 0.2ms vs 5ms

- ✅ **Smaller model:** 0.93 MB vs 60 MB### Image-Based CNN (Quantized):

- **Training Accuracy**: ~98-99%

### vs Normalized Images- **Validation Accuracy**: ~96-98%

- ✅ **Better landmark quality:** Original images preserve finger separation- **Real-time FPS**: 30-60 FPS

- ✅ **54% more samples:** 41,930 vs 27,241- **Total Latency**: ~50ms (preprocessing + inference)

- ✅ **Better accuracy:** 99.95% vs 99.78%- **Model Size**: 5MB (quantized) or 60MB (original)

- ✅ **No feature loss:** Brightness normalization destroyed edges- **Requires**: Good lighting, plain background



## 🐛 Troubleshooting## 🌐 Web Integration (Future)



### "ModuleNotFoundError: No module named 'mediapipe'"The `web_integration.py` script provides a framework for integrating this model with web meeting platforms like Google Meet:

```powershell

.venv311\Scripts\Activate.ps1### Architecture:

pip install -r requirements.txt1. **Browser Extension**: Captures video stream

```2. **WebSocket Server**: Connects browser to Python backend

3. **ML Backend**: Processes frames and returns predictions

### "Python version 3.13 not supported"4. **Overlay UI**: Displays translations in the browser

- MediaPipe requires Python 3.11 or 3.12

- Use the included `.venv311` environment### Technologies:

- Chrome Extension API

### Webcam not detected- WebSocket for real-time communication

- Check if webcam is being used by another application- TensorFlow.js (optional) for in-browser inference

- Verify webcam index in `realtime_inference_landmark_2hands.py` (default: 0)- MediaPipe for hand detection (optional enhancement)



### Poor recognition accuracy## 🔧 Configuration

- Ensure good lighting (webcam can see hand clearly)

- Sign at comfortable distance from cameraEdit these variables in the scripts to customize behavior:

- Hold sign steady for stability threshold (5 frames)

### `train_model.py`

## 📝 Technical Details```python

IMG_SIZE = 128          # Image size for training

### Model ArchitectureBATCH_SIZE = 32         # Batch size

```pythonEPOCHS = 50             # Maximum epochs

Model: Sequential```

- Dense(512, activation='relu', input_dim=126)

- BatchNormalization()### `realtime_inference.py`

- Dropout(0.3)```python

- Dense(256, activation='relu')CONFIDENCE_THRESHOLD = 0.7     # Minimum confidence for display

- BatchNormalization()PREDICTION_SMOOTHING = 5        # Frames to average

- Dropout(0.3)```

- Dense(128, activation='relu')

- BatchNormalization()## 📈 Improving Performance

- Dropout(0.3)

- Dense(64, activation='relu')### Model Improvements:

- BatchNormalization()1. **Transfer Learning**: Use pre-trained models (ResNet, MobileNet)

- Dropout(0.3)2. **Hand Detection**: Add MediaPipe for hand tracking

- Dense(35, activation='softmax')3. **Temporal Models**: Use LSTM/GRU for gesture sequences

4. **Data Collection**: Add more diverse samples

Total params: 243,619 (0.93 MB)

```### Real-time Improvements:

1. **Model Quantization**: Reduce model size

### Feature Vector (126 dimensions)2. **TensorFlow Lite**: Faster inference

```3. **GPU Acceleration**: Enable CUDA support

Hand 1 (63 features):4. **Background Subtraction**: Isolate hand region

  - Wrist: x, y, z

  - Thumb (4 points): 4×3 = 12## 🐛 Troubleshooting

  - Index (4 points): 4×3 = 12

  - Middle (4 points): 4×3 = 12### Issue: Low FPS

  - Ring (4 points): 4×3 = 12- **Solution**: Reduce frame size, use GPU, decrease PREDICTION_SMOOTHING

  - Pinky (4 points): 4×3 = 12

  - Palm (1 point): 1×3 = 3### Issue: Poor accuracy in real-time

- **Solution**: Ensure good lighting, plain background, adjust CONFIDENCE_THRESHOLD

Hand 2 (63 features): Same structure or zeros if not detected

```### Issue: Webcam not detected

- **Solution**: Check camera permissions, try different camera index (0, 1, 2)

### Smoothing & Stability

- **Prediction history:** Last 3 predictions### Issue: CUDA/GPU errors

- **Stability threshold:** 5 consecutive frames- **Solution**: Install TensorFlow-GPU version matching your CUDA version

- **Confidence threshold:** 80%

- **Result:** Reduces flickering, improves accuracy## 📝 Notes



## 📚 Documentation- The model works best with good lighting and plain backgrounds

- Hold signs steady for 0.5-1 second for recognition

- **2HAND_SUPPORT.md** - Details on 2-hand implementation- The model is trained on static signs (no motion-based signs)

- **MIGRATION_COMPLETE.md** - MediaPipe migration journey- Future versions will support continuous sign language (sentences/phrases)



## 🎯 Future Enhancements## 🎯 Next Steps



- [ ] TensorFlow Lite conversion for faster inference1. ✅ Train and test basic model

- [ ] Multi-threaded processing for higher FPS2. ✅ Real-time inference with webcam

- [ ] Google Meet Chrome extension3. ⏳ Add hand detection with MediaPipe

- [ ] Support for ISL phrases and sentences4. ⏳ Implement web extension for Google Meet

- [ ] Mobile app deployment5. ⏳ Add support for dynamic/motion signs

- [ ] Real-time translation overlay6. ⏳ Build word/phrase prediction

7. ⏳ Deploy as cloud service

## 📄 License

## 📄 License

This project is for educational purposes.

This project is for educational purposes. Please ensure proper attribution if using the dataset or model.

---

## 🤝 Contributing

**Built with:** Python 3.11 • MediaPipe • TensorFlow/Keras • OpenCV • NumPy

Contributions are welcome! Areas for improvement:
- Dataset expansion
- Model optimization
- UI/UX enhancements
- Web integration
- Documentation

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**Built with ❤️ for the deaf and hard-of-hearing community**
