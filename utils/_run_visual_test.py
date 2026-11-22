"""Прогон глаз и топологии на ComfyUI_00001 и запись кадров."""
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI")
sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI\custom_nodes")
import folder_paths
from comfy_extras.nodes_mesh_postprocess import _build_triangle_bvh, _render_view
from garage_kit_eyes import repair_eyes
from garage_kit_topology import prepare_topology

folder_paths.set_output_directory(r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
SRC = r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\3d\ComfyUI_00001.glb"
OUT = r"D:\ComfyLLM\3D\ComfyUI\output\3d\preview"
os.makedirs(OUT, exist_ok=True)

geom = next(iter(trimesh.load(SRC, force="scene").geometry.values()))
verts = np.asarray(geom.vertices, dtype=np.float32)
faces = np.asarray(geom.faces, dtype=np.int64)
uvs = np.asarray(geom.visual.uv, dtype=np.float32)
image = np.asarray(geom.visual.material.baseColorTexture.convert("RGB"))


def render_face(mesh_v, mesh_f, mesh_uv, tex, path):
    y = mesh_v[:, 1]
    band = (y > 0.30) & (y < 0.50) & (np.abs(mesh_v[:, 0]) < 0.12)
    use = mesh_f[band[mesh_f].all(1)]
    used = np.unique(use)
    remap = np.full(len(mesh_v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vv = np.ascontiguousarray(mesh_v[used])
    ff = np.ascontiguousarray(remap[use])
    uuv = np.ascontiguousarray(mesh_uv[used].copy())
    uuv[:, 1] = 1.0 - uuv[:, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = torch.from_numpy(vv).to(device)
    lf = torch.from_numpy(ff).to(device)
    tri = lv[lf]
    bvh = _build_triangle_bvh(tri)
    center = lv.mean(0)
    dist = float((lv - center).norm(dim=1).max()) * 2.15
    eye = center + torch.tensor([0.0, 0.02, dist], device=device)
    forward = torch.nn.functional.normalize(center - eye, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    up = torch.linalg.cross(right, forward)
    img, hit, _ = _render_view(
        tri, bvh,
        torch.from_numpy(uuv).to(device),
        lf,
        torch.from_numpy(np.ascontiguousarray(tex.astype(np.float32) / 255.0)).to(device),
        eye, forward, right, up, 0.38, 900, 900,
    )
    rgb = np.clip(img.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    rgb[~hit.detach().cpu().numpy()] = 18
    Image.fromarray(rgb).save(path)
    print("wrote", path)
    if device.type == "cuda":
        torch.cuda.empty_cache()


render_face(verts, faces, uvs, image, os.path.join(OUT, "test_face_before.png"))
nv, nf, nuv, nimg, report = repair_eyes(verts.astype(np.float64), faces, uvs, image)
print(report)
render_face(nv, nf, nuv, nimg, os.path.join(OUT, "test_face_after.png"))


class Mesh:
    pass


mesh = Mesh()
mesh.vertices = torch.from_numpy(nv).unsqueeze(0)
mesh.faces = torch.from_numpy(nf).unsqueeze(0)
mesh.uvs = torch.from_numpy(nuv).unsqueeze(0)
result, files = prepare_topology(mesh, "quads", 18000, 900, "3d/2B_Anim")
print("files", files)
for name, image_t in zip(("test_uv.png", "test_rig.png", "test_quads.png"), result[1:]):
    arr = (image_t[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, name))
    print("wrote", name)
