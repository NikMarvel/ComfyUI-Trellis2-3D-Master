"""
Утилита экспорта 3D-моделей: Конвертер GLB в высокоточный бинарный STL.

Архитектурная роль:
- Извлекает триангулированную сетку (позиции и индексы) из glTF/GLB без потерь точности.
- Рассчитывает геометрические нормали граней.
- Записывает монолитный бинарный STL файл для прямой загрузки в слайсеры 3D-печати
  (Chitubox, Lychee Slicer, PrusaSlicer, Bambu Studio, Cura) в исходной ориентации без наклона.
"""

import struct
import json
import os
import sys
import numpy as np


def convert_glb_to_stl(glb_path: str, stl_path: str = None) -> str:
    """
    Конвертирует мастер-файл GLB в бинарный STL с сохранением абсолютных координат.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Файл не найден: {glb_path}")

    if stl_path is None:
        stl_path = os.path.splitext(glb_path)[0] + ".stl"

    with open(glb_path, "rb") as f:
        magic, version, length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError("Файл не является валидным glTF/GLB.")

        chunk_len, chunk_type = struct.unpack("<I4s", f.read(8))
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes.decode("utf-8"))

        bin_len, bin_type = struct.unpack("<I4s", f.read(8))
        bin_offset = f.tell()

        mesh = gltf["meshes"][0]
        prim = mesh["primitives"][0]
        pos_accessor = gltf["accessors"][prim["attributes"]["POSITION"]]
        idx_accessor = gltf["accessors"][prim["indices"]]

        pos_bv = gltf["bufferViews"][pos_accessor["bufferView"]]
        f.seek(bin_offset + pos_bv.get("byteOffset", 0) + pos_accessor.get("byteOffset", 0))
        pos_bytes = f.read(pos_accessor["count"] * 12)
        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)

        idx_bv = gltf["bufferViews"][idx_accessor["bufferView"]]
        f.seek(bin_offset + idx_bv.get("byteOffset", 0) + idx_accessor.get("byteOffset", 0))
        idx_dtype = np.uint32 if idx_accessor["componentType"] == 5125 else np.uint16
        idx_bytes = f.read(idx_accessor["count"] * (4 if idx_dtype == np.uint32 else 2))
        indices = np.frombuffer(idx_bytes, dtype=idx_dtype).reshape(-1, 3)

    num_triangles = len(indices)

    header = b"Antigravity 3D Print Master - High Detail Binary STL".ljust(80, b" ")
    with open(stl_path, "wb") as f_out:
        f_out.write(header)
        f_out.write(struct.pack("<I", num_triangles))

        v0 = positions[indices[:, 0]]
        v1 = positions[indices[:, 1]]
        v2 = positions[indices[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normals = normals / norm

        record_dtype = np.dtype([
            ("normal", np.float32, (3,)),
            ("v0", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("attr", np.uint16),
        ])

        records = np.zeros(num_triangles, dtype=record_dtype)
        records["normal"] = normals
        records["v0"] = v0
        records["v1"] = v1
        records["v2"] = v2
        records["attr"] = 0

        f_out.write(records.tobytes())

    return stl_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\3d\2B_Figure_Full_Master_4K_1.8M_00001_.glb"
    output = convert_glb_to_stl(target)
    print(f"Готово! STL сохранен: {output}")
