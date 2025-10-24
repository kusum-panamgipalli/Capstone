"""
Generate placeholder icons for the Chrome extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple icon with ISL text"""
    # Create image with gradient background
    img = Image.new('RGB', (size, size), color='#4CAF50')
    draw = ImageDraw.Draw(img)
    
    # Draw circle background
    margin = size // 10
    draw.ellipse([margin, margin, size-margin, size-margin], fill='#2E7D32', outline='#FFFFFF', width=size//20)
    
    # Add text
    try:
        # Try to use a nice font
        font_size = size // 3
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw "ISL" text
    text = "ISL"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # Draw text with shadow
    draw.text((x+2, y+2), text, fill='#000000', font=font)  # Shadow
    draw.text((x, y), text, fill='#FFFFFF', font=font)  # Main text
    
    # Save
    img.save(output_path)
    print(f"✓ Created icon: {output_path} ({size}x{size})")

def main():
    print("\n" + "="*60)
    print("CREATING EXTENSION ICONS")
    print("="*60 + "\n")
    
    # Create icons directory
    icons_dir = '../isl-interpreter-extension/icons'
    os.makedirs(icons_dir, exist_ok=True)
    
    # Generate icons in different sizes
    sizes = [16, 48, 128]
    
    for size in sizes:
        icon_path = os.path.join(icons_dir, f'icon{size}.png')
        create_icon(size, icon_path)
    
    print("\n✓ All icons created successfully!")
    print(f"Location: {os.path.abspath(icons_dir)}")

if __name__ == "__main__":
    main()
