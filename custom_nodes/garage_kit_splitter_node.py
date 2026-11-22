"""
ComfyUI Custom Node: Garage Kit 3D Splitter

Архитектурная роль:
- Принимает единую монолитную 3D-сетку (MESH).
- Анатомически сегментирует фигурку на 5 сборочных компонентов под 3D-печать:
  1. Head & Hair (Голова и прическа)
  2. Torso & Dress (Торс и платье)
  3. Right Arm (Правая рука)
  4. Left Arm (Левая рука)
  5. Legs & Boots (Ноги в высоких сапогах)
- Запечатывает (cap) плоскости среза для идеального сопряжения деталей (нулевой допуск).
- Автоматически сохраняет готовые бинарные STL и GLB в директорию output/3d/.
- Передает отдельные MESH в граф ComfyUI для дальнейшей обработки или раздельного рендера.
"""

import copy
import os
import torch
import numpy as np
import trimesh
import folder_paths


class GarageKitSplitterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Входная монолитная сетка персонажа"}),
                "neck_height": (
                    "FLOAT",
                    {
                        "default": 0.33,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Высота среза шеи / воротника по оси Y",
                    },
                ),
                "waist_height": (
                    "FLOAT",
                    {
                        "default": -0.05,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Высота среза подола платья / пояса по оси Y",
                    },
                ),
                "arm_offset": (
                    "FLOAT",
                    {
                        "default": 0.14,
                        "min": 0.05,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Смещение плечевых суставов по оси X",
                    },
                ),
                "export_stl": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Автоматически экспортировать бинарные STL файлы",
                    },
                ),
                "export_glb": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Автоматически экспортировать GLB файлы",
                    },
                ),
                "folder_name": (
                    "STRING",
                    {
                        "default": "GarageKit_Parts",
                        "tooltip": "Имя подпапки в output/3d для сохранения деталей",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MESH", "MESH", "MESH", "MESH", "MESH")
    RETURN_NAMES = ("head_mesh", "torso_mesh", "right_arm", "left_arm", "legs_mesh")
    FUNCTION = "split_mesh"
    CATEGORY = "3d/garage_kit"

    def split_mesh(
        self,
        mesh,
        neck_height,
        waist_height,
        arm_offset,
        export_stl,
        export_glb,
        folder_name,
    ):
        v = mesh.vertices[0].detach().cpu().numpy()
        f = mesh.faces[0].detach().cpu().numpy()

        t_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

        # 1. Срез головы (Y = neck_height)
        head_t = trimesh.intersections.slice_mesh_plane(
            t_mesh, plane_normal=[0, 1, 0], plane_origin=[0, neck_height, 0], cap=True
        )
        body_t = trimesh.intersections.slice_mesh_plane(
            t_mesh, plane_normal=[0, -1, 0], plane_origin=[0, neck_height, 0], cap=True
        )

        # 2. Срез ног (Y = waist_height)
        torso_arms_t = trimesh.intersections.slice_mesh_plane(
            body_t, plane_normal=[0, 1, 0], plane_origin=[0, waist_height, 0], cap=True
        )
        legs_t = trimesh.intersections.slice_mesh_plane(
            body_t, plane_normal=[0, -1, 0], plane_origin=[0, waist_height, 0], cap=True
        )

        # 3. Срез рук (X = +/- arm_offset)
        r_arm_t = trimesh.intersections.slice_mesh_plane(
            torso_arms_t, plane_normal=[1, 0, 0], plane_origin=[arm_offset, 0, 0], cap=True
        )
        t_left_t = trimesh.intersections.slice_mesh_plane(
            torso_arms_t, plane_normal=[-1, 0, 0], plane_origin=[arm_offset, 0, 0], cap=True
        )

        l_arm_t = trimesh.intersections.slice_mesh_plane(
            t_left_t, plane_normal=[-1, 0, 0], plane_origin=[-arm_offset, 0, 0], cap=True
        )
        torso_t = trimesh.intersections.slice_mesh_plane(
            t_left_t, plane_normal=[1, 0, 0], plane_origin=[-arm_offset, 0, 0], cap=True
        )

        parts = {
            "Part_01_Head_Hair": head_t,
            "Part_02_Torso_Dress": torso_t,
            "Part_03_Right_Arm": r_arm_t,
            "Part_04_Left_Arm": l_arm_t,
            "Part_05_Legs_Boots": legs_t,
        }

        # Экспорт на диск
        if export_stl or export_glb:
            output_dir = os.path.join(folder_paths.get_output_directory(), "3d", folder_name)
            os.makedirs(output_dir, exist_ok=True)
            for name, p in parts.items():
                if export_stl:
                    p.export(os.path.join(output_dir, f"{name}.stl"))
                if export_glb:
                    p.export(os.path.join(output_dir, f"{name}.glb"))

        # Преобразование обратно в объекты ComfyUI Mesh
        def to_comfy_mesh(trimesh_obj):
            m = copy.copy(mesh)
            nv = torch.from_numpy(trimesh_obj.vertices.astype(np.float32)).unsqueeze(0)
            nf = torch.from_numpy(trimesh_obj.faces.astype(np.int64)).unsqueeze(0)
            m.vertices = nv
            m.faces = nf
            m.uvs = None
            m.normals = None
            return m

        return (
            to_comfy_mesh(head_t),
            to_comfy_mesh(torso_t),
            to_comfy_mesh(r_arm_t),
            to_comfy_mesh(l_arm_t),
            to_comfy_mesh(legs_t),
        )


NODE_CLASS_MAPPINGS = {
    "GarageKitSplitter": GarageKitSplitterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GarageKitSplitter": "🧩 Garage Kit 3D Splitter (5 Parts)",
}
