@echo off
REM Complete Training Pipeline for ISL Model
REM This script runs all training steps sequentially

echo ============================================================
echo ISL MODEL TRAINING PIPELINE
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo [1/6] Checking Python packages...
echo.

REM Check if required packages are installed
python -c "import tensorflow, cv2, mediapipe, numpy, pandas, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Some required packages are missing. Installing...
    pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn matplotlib pillow kaggle tensorflowjs tqdm
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
) else (
    echo All required packages are installed
)

echo.
echo ============================================================
echo [2/6] STEP 1: Download Dataset
echo ============================================================
echo.
echo This will download the ISL dataset from Kaggle.
echo Make sure you have set up Kaggle API credentials.
echo.
pause

python 1_download_dataset.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Dataset download failed!
    echo Please check Kaggle credentials and try again.
    echo Or download manually from: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [3/6] STEP 2: Extract Hand Landmarks
echo ============================================================
echo.
echo This will process all images and extract hand landmarks.
echo This may take 5-15 minutes...
echo.
pause

python 2_extract_landmarks.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Landmark extraction failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [4/6] STEP 3: Train Model
echo ============================================================
echo.
echo This will train the neural network model.
echo This may take 10-30 minutes depending on your hardware...
echo.
pause

python 3_train_model.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Model training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [5/6] STEP 4: Convert to TensorFlow.js
echo ============================================================
echo.
echo This will convert the model for browser use...
echo.
pause

python 4_convert_to_tfjs.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Model conversion failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [6/6] STEP 5: Create Extension Icons
echo ============================================================
echo.

python create_icons.py
if %errorlevel% neq 0 (
    echo WARNING: Icon creation failed (not critical)
)

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
echo All steps completed successfully!
echo.
echo Next steps:
echo 1. Go to chrome://extensions/ in Google Chrome
echo 2. Enable "Developer mode"
echo 3. Click "Load unpacked"
echo 4. Select the "isl-interpreter-extension" folder
echo 5. Test the extension on Google Meet or Zoom
echo.
echo Model files are in: ..\isl-interpreter-extension\models\
echo.
pause
