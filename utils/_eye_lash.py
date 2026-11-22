"""Strip only the eyelash artifact and render the eyes."""
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI")
sys.path.insert(0, r"D:\ComfyLLM\3D\ComfyUI\custom_nodes")
from comfy_extras.nodes_mesh_postprocess import _build_triangle_bvh, _render_view
from garage_kit_eyes import repair_eyes

SRC = r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\3d\ComfyUI_00015.glb"
OUT = r"D:\ComfyLLM\3D\ComfyUI\output\3d\preview"
os.makedirs(OUT, exist_ok=True)
geom = next(iter(trimesh.load(SRC, force="scene").geometry.values()))
v = np.asarray(geom.vertices, dtype=np.float32)
f = np.asarray(geom.faces, dtype=np.int64)
uv = np.asarray(geom.visual.uv, dtype=np.float32)
image = np.asarray(geom.visual.material.baseColorTexture.convert("RGB"))
nv, nf, nuv, nimg, report = repair_eyes(v.astype(np.float64), f, uv, image)
print(report)


def render(verts, faces, uvs, tex, path):
    band = (verts[:, 1] > 0.39) & (verts[:, 1] < 0.47) & (np.abs(verts[:, 0]) < 0.11) & (verts[:, 2] > 0.02)
    faces = faces[band[faces].all(1)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vv, ff = verts[used], remap[faces]
    uuv = uvs[used].copy()
    uuv[:, 1] = 1.0 - uuv[:, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = torch.from_numpy(np.ascontiguousarray(vv)).to(device)
    lf = torch.from_numpy(np.ascontiguousarray(ff)).to(device)
    uv_t = torch.from_numpy(np.ascontiguousarray(uuv)).to(device)
    tex_t = torch.from_numpy(np.ascontiguousarray(tex.astype(np.float32) / 255.0)).to(device)
    tri = lv[lf]
    bvh = _build_triangle_bvh(tri)
    center = lv.mean(0)
    dist = float((lv - center).norm(dim=1).max()) * 2.1
    eye = center + torch.tensor([0.0, 0.0, dist], device=device)
    forward = torch.nn.functional.normalize(center - eye, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    up = torch.linalg.cross(right, forward)
    img, hit, _ = _render_view(tri, bvh, uv_t, lf, tex_t, eye, forward, right, up, 0.42, 900, 900)
    rgb = np.clip(img.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    rgb[~hit.detach().cpu().numpy()] = 16
    Image.fromarray(rgb).save(path)
    print("wrote", path)

render(nv.astype(np.float32), nf, nuv, nimg, os.path.join(OUT, "lash_after.png"))
fixed = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
fixed.visual = trimesh.visual.TextureVisuals(
    uv=nuv,
    material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(nimg), metallicFactor=0.0, roughnessFactor=0.55,
    ),
)
dest = os.path.splitext(SRC)[0] + "_eyes.glb"
fixed.export(dest)
print("saved", dest)
