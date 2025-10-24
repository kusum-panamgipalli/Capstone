# Icons Directory

This directory should contain the extension icons.

Required files:
- icon16.png (16x16 pixels)
- icon48.png (48x48 pixels)  
- icon128.png (128x128 pixels)

## How to create icons:

### Option 1: Use an online tool
1. Go to https://www.favicon-generator.org/
2. Upload your logo/image
3. Generate icons
4. Download and rename to icon16.png, icon48.png, icon128.png

### Option 2: Use image editing software
- Use Photoshop, GIMP, or any image editor
- Create images with green background (#4CAF50)
- Add "ISL" text in white
- Export as PNG in sizes 16x16, 48x48, 128x128

### Option 3: Use the Python script
If you have PIL/Pillow installed:
```
pip install pillow
python create_icons.py
```

## Temporary Icons
If icons are missing, Chrome will use default placeholder icons.
The extension will still work without custom icons.
