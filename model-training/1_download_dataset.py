"""
Script to download Indian Sign Language dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
Modified to download only a subset of images for quick training
"""

import os
import zipfile
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """
    Instructions to set up Kaggle API credentials:
    1. Go to https://www.kaggle.com/account
    2. Scroll to API section
    3. Click "Create New API Token"
    4. Save kaggle.json to: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
    """
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("\n" + "="*60)
        print("KAGGLE API SETUP REQUIRED")
        print("="*60)
        print("\n1. Go to: https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print(f"4. Save kaggle.json to: {kaggle_dir}")
        print("\n" + "="*60)
        return False
    
    # Set permissions (Unix-like systems)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    
    return True

def limit_images_per_class(download_path, max_images_per_class=50):
    """
    Keep only a subset of images per class to reduce dataset size
    """
    print(f"\n" + "="*60)
    print(f"LIMITING TO {max_images_per_class} IMAGES PER CLASS")
    print("="*60)
    
    classes_processed = 0
    total_images_kept = 0
    total_images_removed = 0
    
    # Find all class folders (A-Z, 0-9)
    for class_folder in Path(download_path).iterdir():
        if class_folder.is_dir() and (class_folder.name.isalpha() or class_folder.name.isdigit()):
            if len(class_folder.name) == 1:  # Only single character folders
                images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')) + list(class_folder.glob('*.jpeg'))
                
                if len(images) > max_images_per_class:
                    # Keep only first max_images_per_class images
                    images_to_remove = images[max_images_per_class:]
                    
                    for img in images_to_remove:
                        img.unlink()
                        total_images_removed += 1
                    
                    total_images_kept += max_images_per_class
                    print(f"  {class_folder.name}: Kept {max_images_per_class}, removed {len(images_to_remove)}")
                else:
                    total_images_kept += len(images)
                    print(f"  {class_folder.name}: Kept all {len(images)} images")
                
                classes_processed += 1
    
    print(f"\n✓ Processed {classes_processed} classes")
    print(f"✓ Total images kept: {total_images_kept}")
    print(f"✓ Total images removed: {total_images_removed}")
    print(f"\nReduced dataset size significantly!")

def download_dataset(max_images_per_class=50):
    """Download ISL dataset from Kaggle and limit images per class"""
    
    # Check Kaggle credentials
    if not setup_kaggle_credentials():
        print("\nPlease set up Kaggle credentials and run this script again.")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("\n" + "="*60)
        print("DOWNLOADING ISL DATASET FROM KAGGLE")
        print("="*60)
        print(f"\nNote: Will limit to {max_images_per_class} images per class")
        
        # Dataset identifier
        dataset = 'prathumarikeri/indian-sign-language-isl'
        
        # Download path
        download_path = './data/raw'
        os.makedirs(download_path, exist_ok=True)
        
        print(f"\nDownloading dataset: {dataset}")
        print(f"To: {download_path}")
        print("This will download the full dataset first, then reduce it...")
        print("(~500MB download, will be reduced to ~50MB)\n")
        
        # Download dataset
        api.dataset_download_files(dataset, path=download_path, unzip=True)
        
        print("\n✓ Dataset downloaded successfully!")
        
        # Limit images per class to reduce size
        limit_images_per_class(download_path, max_images_per_class)
        
        print(f"\n✓ Dataset ready at: {os.path.abspath(download_path)}")
        
        # Show final structure
        print("\nDataset structure (sample):")
        count = 0
        for root, dirs, files in os.walk(download_path):
            if count < 10:  # Show first 10 folders
                level = root.replace(download_path, '').count(os.sep)
                indent = ' ' * 2 * level
                folder_name = os.path.basename(root)
                if folder_name:
                    print(f'{indent}{folder_name}/ ({len(files)} images)')
                    count += 1
        
        return True
        
    except ImportError:
        print("\nError: kaggle package not installed")
        print("Install it with: pip install kaggle")
        return False
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Kaggle API credentials are set up correctly")
        print("2. Check internet connection")
        print("3. Verify dataset name is correct")
        return False

def manual_download_instructions():
    """Provide manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nIf automatic download fails, download manually:")
    print("\n1. Visit: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl")
    print("2. Click 'Download' button (you may need to sign in)")
    print("3. Extract the ZIP file")
    print("4. Copy extracted folder to: ./data/raw/")
    print("5. (Optional) Manually limit to ~50 images per folder to reduce size")
    print("\n" + "="*60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ISL DATASET DOWNLOADER (REDUCED SIZE)")
    print("="*60)
    print("\nThis will download the dataset and keep only 50 images per class")
    print("Total size: ~50MB instead of full 500MB")
    
    # Ask user for confirmation
    print("\nYou can also use synthetic data instead:")
    print("  python 0_generate_synthetic_data.py")
    print("\nOr continue with limited real dataset download.")
    
    success = download_dataset(max_images_per_class=50)
    
    if not success:
        manual_download_instructions()
    else:
        print("\n✓ Reduced dataset ready for processing!")
        print("\nNext step: Run 2_extract_landmarks.py")
