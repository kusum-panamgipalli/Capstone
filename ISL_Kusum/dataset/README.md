# Dataset Information

**Note**: The training dataset (`Indian/` folder) remains in the parent directory due to its large size (~6 GB with 42,745 images).

**Location**: `../Indian/`

The dataset is referenced by the scripts in the `scripts/` folder but kept separate to:
- Avoid duplication
- Keep this folder size manageable for Git
- Allow sharing with other contributors

## Dataset Structure
```
Indian/
  ├── 1/ (1,200 images)
  ├── 2/ (1,200 images)
  ...
  ├── C/ (1,447 images)
  ...
  └── Z/ (1,200 images)
```

Total: 35 classes (1-9, A-Z)
Total Images: 42,745
Format: JPG with dark backgrounds
