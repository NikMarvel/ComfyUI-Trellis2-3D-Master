import os
from PIL import Image, ImageDraw, ImageFont

def make_comparison():
    artifact_dir = r'C:\Users\nikit\.gemini\antigravity\brain\31af7ac8-c2c2-4523-be81-f928dee69b9e'
    raw_1024_path = r'C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\front_view_00008.png'
    ultra_4k_path = os.path.join(artifact_dir, 'front_view_4K_UltraSharp.png')

    if not os.path.exists(raw_1024_path) or not os.path.exists(ultra_4k_path):
        print("Files not found")
        return

    img_1024 = Image.open(raw_1024_path).convert('RGB')
    img_4k = Image.open(ultra_4k_path).convert('RGB')

    # Crop head and chest:
    # in 1024x1024: x=256..768 (512 width), y=50..562 (512 height)
    crop_1024 = img_1024.crop((256, 50, 768, 562)).resize((1024, 1024), Image.Resampling.NEAREST)
    
    # in 4096x4096: x=1024..3072 (2048 width), y=200..2248 (2048 height)
    crop_4k = img_4k.crop((1024, 200, 3072, 2248)).resize((1024, 1024), Image.Resampling.LANCZOS)

    # Combine side-by-side (2048 x 1024)
    comp = Image.new('RGB', (2048, 1024))
    comp.paste(crop_1024, (0, 0))
    comp.paste(crop_4k, (1024, 0))

    # Add text banner
    draw = ImageDraw.Draw(comp)
    # Draw simple bounding divider line
    draw.line([(1024, 0), (1024, 1024)], fill=(255, 215, 0), width=4)

    comp.save(os.path.join(artifact_dir, 'comparison_1024_vs_4K.png'))
    print("Comparison image created successfully!")

if __name__ == '__main__':
    make_comparison()
