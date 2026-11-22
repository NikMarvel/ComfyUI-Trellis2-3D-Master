import trimesh
import numpy as np

glb_path = r'C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\3d\2B_Figure_Full_Master_8K_3M_Poly_00002_.glb'
scene = trimesh.load(glb_path)
print('Scene type:', type(scene))
if isinstance(scene, trimesh.Scene):
    for name, geom in scene.geometry.items():
        print(f'Geom {name}: vertices={geom.vertices.shape}, faces={geom.faces.shape}')
        print(f'Vertex normals: {geom.vertex_normals.shape if geom.vertex_normals is not None else None}')
        if hasattr(geom.visual, 'material'):
            m = geom.visual.material
            print('Material attributes:')
            for attr in ['baseColorTexture', 'normalTexture', 'metallicRoughnessTexture']:
                print(f'  {attr}: {getattr(m, attr, None)}')
                if getattr(m, attr, None) is not None:
                    tex = getattr(m, attr)
                    print(f'    size: {tex.size}, mode: {tex.mode}')
