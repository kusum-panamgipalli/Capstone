# Quick Start Guide - ISL_Kusum

## 📂 Folder Organization

This folder (`ISL_Kusum/`) contains all of Kusum's work on the ISL interpreter project, organized separately from other contributors.

## 🚀 Running the Model

### Option 1: From the scripts folder (Recommended)
```powershell
cd ISL_Kusum\scripts
..\.venv311\Scripts\python.exe realtime_inference_landmark_2hands.py
```

### Option 2: From the project root
```powershell
cd ISL_Kusum
.\.venv311\Scripts\python.exe scripts\realtime_inference_landmark_2hands.py
```

**Controls:**
- **SPACE**: Add space to output
- **BACKSPACE**: Delete last character  
- **C**: Clear all text
- **Q**: Quit

## 🔄 Retraining the Model

If you modify the dataset (add/remove/change images in `../Indian/` folder):

### Step 1: Extract Landmarks
```powershell
cd ISL_Kusum\scripts
..\.venv311\Scripts\python.exe extract_landmarks_2hands.py
```
This creates `models/hand_landmarks_dataset_2hands.pkl` (~10 min)

### Step 2: Train Model
```powershell
..\.venv311\Scripts\python.exe train_landmark_model_2hands.py
```
This creates `models/isl_landmark_model_2hands.h5` (~6 min)

### Step 3: Analyze Results
```powershell
cd ..\analysis
..\.venv311\Scripts\python.exe detailed_error_analysis.py
```

## 📊 Analysis Tools

All analysis scripts are in the `analysis/` folder:

```powershell
cd ISL_Kusum\analysis

# Overall dataset analysis
..\.venv311\Scripts\python.exe analyze_dataset.py

# Detailed error analysis
..\.venv311\Scripts\python.exe detailed_error_analysis.py

# Visual sample count analysis
..\.venv311\Scripts\python.exe visual_dataset_analysis.py
```

## 📁 File Structure

```
ISL_Kusum/
├── models/           # Trained models and datasets
├── scripts/          # Main Python scripts
├── web_extension/    # Google Meet integration (future)
├── analysis/         # Dataset analysis tools
├── docs/             # Documentation
└── dataset/          # Reference to ../Indian/
```

## 🔗 Git Repository Setup

To connect to your existing Git repository:

```powershell
# Navigate to ISL_Kusum folder
cd ISL_Kusum

# Initialize Git
git init

# Add your remote repository
git remote add origin <your-repo-url>

# Add all files
git add .

# Commit
git commit -m "Initial commit - ISL Interpreter v2.0 (2-hand support, 99.98% accuracy)"

# Push to repository
git push -u origin main
```

## ⚠️ Important Notes

1. **Dataset Location**: The training dataset (`Indian/` folder) remains in the parent directory (`../Indian/`) to avoid duplication
2. **Virtual Environment**: The Python 3.11 environment is in `.venv311/` folder (already included, excluded from Git)
3. **Model Files**: All trained models are in `models/` folder (already trained and ready to use)
4. **Documentation**: See `PROJECT_INFO.md` for complete project details

## 🎯 Current Model Performance

- **Accuracy**: 99.98% validation
- **Errors**: Only 2 in 8,386 samples (both for sign 'C')
- **Speed**: ~12 FPS real-time
- **Model Size**: 0.93 MB
- **Supports**: 35 signs (1-9, A-Z) with 1-hand and 2-hand detection

## 📖 More Information

- **Complete Documentation**: See `PROJECT_INFO.md`
- **Dataset Improvement Guide**: See `docs/DATASET_IMPROVEMENT_GUIDE.md`
- **2-Hand Implementation**: See `docs/2HAND_SUPPORT.md`
- **Migration History**: See `docs/MIGRATION_COMPLETE.md`

---

**Last Updated**: November 7, 2025
