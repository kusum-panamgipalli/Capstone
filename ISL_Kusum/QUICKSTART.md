# Quick Start Guide - ISL_Kusum

##  Running the Model

```powershell
cd ISL_Kusum
.\.venv311\Scripts\python.exe scripts\realtime_inference_landmark_2hands.py
```

**Controls:**
- **SPACE**: Add space
- **BACKSPACE**: Delete character  
- **C**: Clear text
- **Q**: Quit

---

## � Setup (First Time Only)

If you just cloned this repository:

```powershell
# 1. Create virtual environment
cd ISL_Kusum
python -m venv .venv311

# 2. Install dependencies
.\.venv311\Scripts\pip install -r requirements.txt

# 3. Download the Indian/ dataset folder separately (not in Git)
# Place it in the parent directory: ISL/Indian/

# 4. Run the model
.\.venv311\Scripts\python.exe scripts\realtime_inference_landmark_2hands.py
```

---

## 🔄 Retrain Model (If Dataset Changed)

```powershell
cd ISL_Kusum

# Extract landmarks (~10 min)
.\.venv311\Scripts\python.exe scripts\extract_landmarks_2hands.py

# Train model (~6 min)
.\.venv311\Scripts\python.exe scripts\train_landmark_model_2hands.py
```

---

**Model**: 99.98% accuracy | 0.93 MB | ~12 FPS  
**Supports**: 35 signs (1-9, A-Z) with 1-hand and 2-hand detection
