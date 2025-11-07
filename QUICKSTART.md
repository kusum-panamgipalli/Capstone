# ISL Interpreter - Quick Start Guide

Get your Indian Sign Language real-time interpreter up and running in minutes!

## 🎯 Two Options

### Option A: Use Without Training (Basic Mode)
The extension will work with basic gesture recognition even without a trained model.

### Option B: Full AI Mode (Recommended)
Train the model for accurate ISL recognition (A-Z, 0-9).

---

## Option A: Quick Setup (No Training)

### 1. Load the Extension

1. Open Google Chrome
2. Go to `chrome://extensions/`
3. Toggle ON "Developer mode" (top-right corner)
4. Click "Load unpacked"
5. Navigate to and select: `d:\Python\Capstone\isl-interpreter-extension`
6. Extension is now installed! ✓

### 2. Test It

1. Go to https://meet.google.com/ and start a test meeting
2. Click the ISL Interpreter icon in Chrome toolbar
3. Click "Enable Interpreter"
4. Position your hand in front of camera
5. You'll see basic gesture recognition (finger counting)

**Note:** Without training, accuracy is limited to basic hand gestures.

---

## Option B: Full Setup with AI Model

### Prerequisites
- Python 3.8+ installed
- Kaggle account
- 30-60 minutes for complete training

### Step 1: Set Up Kaggle API (5 minutes)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save downloaded `kaggle.json` to:
   ```
   C:\Users\<YourUsername>\.kaggle\kaggle.json
   ```

### Step 2: Install Python Packages (5 minutes)

Open PowerShell in the project directory:

```powershell
cd d:\Python\Capstone\model-training
D:/Python/.venv/Scripts/python.exe -m pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn matplotlib pillow kaggle tensorflowjs tqdm
```

### Step 3: Run Complete Training Pipeline (30-45 minutes)

**Easy Way - Run All at Once:**

```powershell
cd d:\Python\Capstone\model-training
.\train_all.bat
```

The script will:
- Download ISL dataset (~500MB)
- Extract hand landmarks
- Train neural network
- Convert to TensorFlow.js
- Copy to extension folder

**Manual Way - Step by Step:**

```powershell
# Step 1: Download dataset (5 mins)
D:/Python/.venv/Scripts/python.exe 1_download_dataset.py

# Step 2: Extract landmarks (10 mins)
D:/Python/.venv/Scripts/python.exe 2_extract_landmarks.py

# Step 3: Train model (20 mins)
D:/Python/.venv/Scripts/python.exe 3_train_model.py

# Step 4: Convert to JS (2 mins)
D:/Python/.venv/Scripts/python.exe 4_convert_to_tfjs.py
```

### Step 4: Load Extension (2 minutes)

1. Go to `chrome://extensions/` in Chrome
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select: `d:\Python\Capstone\isl-interpreter-extension`
5. Extension loaded with AI model! ✓

### Step 5: Test on Google Meet (5 minutes)

1. Join a Google Meet call
2. Click ISL Interpreter extension icon
3. Click "Enable Interpreter"
4. Show ISL signs (A-Z, 0-9) to camera
5. See real-time translation! 🎉

---

## 📊 What to Expect

### Without Training (Basic Mode):
- Finger counting recognition (0-5 fingers)
- ~70% accuracy for basic gestures
- Works immediately

### With Training (AI Mode):
- Full ISL alphabet (A-Z) and numbers (0-9)
- 85-95% accuracy (depends on training)
- Requires dataset download and training

---

## 🎮 Using the Extension

### On Google Meet:

1. **Join a meeting** (or start a test meeting)
2. **Click extension icon** in Chrome toolbar
3. **Click "Enable Interpreter"**
4. **Position hand** clearly in camera view
5. **Make ISL signs**
6. **See translation** appear as overlay text

### Controls:

- **Enable/Disable**: Toggle interpreter on/off
- **Show Confidence**: Toggle confidence scores
- **Processing Speed**: Adjust FPS (15, 20, 30)

### Tips for Best Results:

✅ **Good lighting** - Face a window or light source
✅ **Clear background** - Avoid cluttered backgrounds
✅ **Steady hand** - Hold sign for 1-2 seconds
✅ **Camera position** - Ensure full hand is visible
✅ **Correct form** - Make signs clearly and accurately

❌ **Avoid:**
- Dark rooms
- Moving too fast
- Partial hand visibility
- Multiple hands (unless intended)

---

## 🐛 Troubleshooting

### Extension not appearing?
- Check `chrome://extensions/` - ensure it's enabled
- Look for errors in the extension details
- Try reloading the extension

### No video detected?
- Ensure camera permissions are granted
- Check if camera is being used by another app
- Try refreshing the Google Meet page

### Low accuracy?
- Ensure model is trained (check models folder)
- Improve lighting conditions
- Make signs more clearly
- Hold signs steady for 1-2 seconds

### Performance issues?
- Lower processing speed to 15 FPS
- Close other Chrome tabs
- Ensure good internet connection

### Model not loading?
- Check browser console (F12) for errors
- Verify model files exist in `models/` folder
- Ensure TensorFlow.js can load (check CSP settings)

---

## 📁 Project Structure

```
Capstone/
├── isl-interpreter-extension/    # Chrome extension
│   ├── manifest.json             # Extension config
│   ├── popup.html/js/css         # Extension UI
│   ├── content.js                # Meeting page integration
│   ├── video-processor.js        # AI processing
│   ├── models/                   # Trained model (after training)
│   └── icons/                    # Extension icons
│
└── model-training/               # Training scripts
    ├── 1_download_dataset.py     # Download ISL data
    ├── 2_extract_landmarks.py    # Extract features
    ├── 3_train_model.py          # Train neural network
    ├── 4_convert_to_tfjs.py      # Convert for browser
    ├── train_all.bat             # Run all steps
    └── TRAINING_GUIDE.md         # Detailed training info
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Extension appears in `chrome://extensions/`
- [ ] Extension icon visible in Chrome toolbar
- [ ] Popup opens when clicking icon
- [ ] Model files exist in `models/` folder (if trained)
- [ ] Console shows no errors (F12 on Meet page)
- [ ] Hand detection works (landmarks visible)
- [ ] Text overlay appears
- [ ] Signs are recognized (test A, B, C, 1, 2, 3)

---

## 🎓 Learning ISL Signs

To use the interpreter effectively, learn ISL signs:

### Resources:
- [ISL Dictionary](https://www.islrtc.nic.in/)
- [YouTube ISL Tutorials](https://www.youtube.com/results?search_query=indian+sign+language+alphabet)
- Practice with the Kaggle dataset images

### Quick Reference:
The extension recognizes:
- **Letters**: A through Z (ISL alphabet)
- **Numbers**: 0 through 9 (ISL numerals)

---

## 📞 Need Help?

1. **Check README.md** for detailed documentation
2. **Check TRAINING_GUIDE.md** for training issues
3. **Browser Console** (F12) for error messages
4. **GitHub Issues** for community support

---

## 🎉 You're Ready!

The ISL Interpreter is now set up and ready to use. Join a Google Meet call and start translating sign language in real-time!

**Happy Signing! 🤟**

---

## Next Steps

1. **Practice**: Test with different ISL signs
2. **Share**: Let others try your extension
3. **Improve**: Collect feedback and retrain
4. **Contribute**: Add more features or signs
5. **Deploy**: Package for Chrome Web Store

For advanced topics, see README.md and TRAINING_GUIDE.md.
