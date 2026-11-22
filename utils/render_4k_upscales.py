import os
import torch
import numpy as np
from PIL import Image
import sys

sys.path.append('D:/ComfyLLM/3D/ComfyUI')
import comfy.utils
import comfy.model_management
from spandrel import ModelLoader

def upscale_views():
    input_dir = r'C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output'
    output_dir = r'C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\upscale_4k_views'
    artifact_dir = r'C:\Users\nikit\.gemini\antigravity\brain\31af7ac8-c2c2-4523-be81-f928dee69b9e'
    os.makedirs(output_dir, exist_ok=True)

    model_path = r'D:\ComfyLLM\3D\ComfyUI\models\upscale_models\4x-UltraSharp.pth'
    print(f"Loading upscale model: {model_path}")
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    model = ModelLoader().load_from_state_dict(sd).eval()

    device = comfy.model_management.get_torch_device()
    model = model.to(device)

    views = [
        ('front_view_00008.png', 'front_view_4K_UltraSharp.png'),
        ('left_view_00008.png', 'left_view_4K_UltraSharp.png'),
        ('back_view_00008.png', 'back_view_4K_UltraSharp.png'),
        ('right_view_00008.png', 'right_view_4K_UltraSharp.png'),
    ]

    for in_name, out_name in views:
        in_path = os.path.join(input_dir, in_name)
        if not os.path.exists(in_path):
            print(f"File not found: {in_path}")
            continue

        print(f"Upscaling {in_name}...")
        img = Image.open(in_path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        # [1, 3, H, W]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        tile = 512
        overlap = 32
        with torch.no_grad():
            s = comfy.utils.tiled_scale(
                t, lambda a: model(a),
                tile_x=tile, tile_y=tile, overlap=overlap,
                upscale_amount=4, output_device=torch.device('cpu')
            )
        # [1, 3, H*4, W*4] -> [H*4, W*4, 3]
        out_arr = (torch.clamp(s[0].permute(1, 2, 0), 0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        out_img = Image.fromarray(out_arr)
        
        # Save to output folder
        out_path1 = os.path.join(output_dir, out_name)
        out_img.save(out_path1, quality=95)
        print(f"Saved: {out_path1} (Size: {out_img.size})")

        # Save to artifact folder
        out_path2 = os.path.join(artifact_dir, out_name)
        out_img.save(out_path2, quality=95)
        print(f"Saved artifact: {out_path2}")

        # Also create a 1024 preview crop of the face/chest from front for quick side-by-side comparison
        if 'front' in out_name:
            # Crop 1024x1024 around head & chest (from 4096: x: 1024..3072, y: 0..2048)
            head_crop = out_img.crop((1024, 200, 3072, 2248))
            head_crop.save(os.path.join(artifact_dir, 'front_4K_Head_Chest_Detail.png'))
            print("Saved head/chest detail crop")

    print("All upscales finished successfully!")

if __name__ == '__main__':
    upscale_views()
