"""Build the quad cage, rig, STL and the three preview images."""
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI")
sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI\custom_nodes")
import folder_paths
from garage_kit_topology import prepare_topology

folder_paths.set_output_directory(r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
SRC = r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\3d\ComfyUI_00001.glb"
geom = next(iter(trimesh.load(SRC, force="scene").geometry.values()))


class Mesh:
    pass


mesh = Mesh()
mesh.vertices = torch.from_numpy(np.asarray(geom.vertices, dtype=np.float32)).unsqueeze(0)
mesh.faces = torch.from_numpy(np.asarray(geom.faces, dtype=np.int64)).unsqueeze(0)
mesh.uvs = torch.from_numpy(np.asarray(geom.visual.uv, dtype=np.float32)).unsqueeze(0)
result, files = prepare_topology(mesh, "quads", 16000, 768, "3d/2B_Anim")
print("files", files)
out = r"D:\ComfyLLM\3D\ComfyUI\output\3d\preview"
os.makedirs(out, exist_ok=True)
for name, image in zip(("topo_uv.png", "topo_rig.png", "topo_wire.png"), result[1:]):
    arr = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out, name))
    print("wrote", name)
