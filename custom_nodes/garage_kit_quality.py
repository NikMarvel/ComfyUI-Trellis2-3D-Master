"""Тумблер качества: авто по видеопамяти или ручные числа.

Авто подбирает потолок, который такая карта реально тянет. Ручной режим
отдаёт числа с этой ноды, и связанные ноды берут их вместо своих виджетов.
"""

from __future__ import annotations

import torch


# Потолок, уже проверенный на RTX 3060 6 ГБ. Авто на этой карте возвращает его же.
_PROFILE_6 = dict(
    structure_steps=28,
    shape_steps=22,
    detail_steps=18,
    texture_steps=24,
    upsample=1536,
    remesh=1536,
    faces=4_500_000,
    atlas=4096,
    normal_res=4096,
    project_res=4096,
    ao_res=2048,
    ao_samples=512,
    view_res=2048,
    quads=18_000,
)

_KEYS = (
    "structure_steps",
    "shape_steps",
    "detail_steps",
    "texture_steps",
    "upsample",
    "remesh",
    "faces",
    "atlas",
    "normal_res",
    "project_res",
    "ao_res",
    "ao_samples",
    "view_res",
    "quads",
)


def _snap(value, step, lo, hi):
    snapped = int(round(float(value) / step) * step)
    return int(min(hi, max(lo, snapped)))


def detect_vram():
    if not torch.cuda.is_available():
        return 0.0, "CPU"
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None)
    if total is None:
        total = getattr(props, "total_mem", 0)
    name = torch.cuda.get_device_name(0)
    return float(total) / (1024 ** 3), name


def profile_for_vram(gb):
    """Чем больше память, тем выше разрешение сетки и карт. Шаги растут медленнее."""
    p = dict(_PROFILE_6)
    if gb <= 7.5:
        tier = "6 ГБ"
    elif gb <= 10:
        tier = "8 ГБ"
        p.update(shape_steps=24, detail_steps=20, texture_steps=28, faces=6_000_000, ao_samples=768, view_res=2560, quads=22_000)
    elif gb <= 14:
        tier = "12 ГБ"
        p.update(
            structure_steps=32, shape_steps=28, detail_steps=22, texture_steps=28,
            upsample=1792, remesh=1792, faces=8_000_000,
            ao_res=4096, ao_samples=768, view_res=3072, quads=30_000,
        )
    elif gb <= 20:
        tier = "16 ГБ"
        p.update(
            structure_steps=32, shape_steps=28, detail_steps=24, texture_steps=32,
            upsample=2048, remesh=2048, faces=12_000_000,
            ao_res=4096, ao_samples=1024, view_res=4096, quads=40_000,
        )
    else:
        tier = "24 ГБ и выше"
        p.update(
            structure_steps=36, shape_steps=32, detail_steps=24, texture_steps=32,
            upsample=2048, remesh=2048, faces=16_000_000,
            atlas=8192, normal_res=8192, project_res=8192,
            ao_res=4096, ao_samples=1024, view_res=4096, quads=50_000,
        )
    p["upsample"] = _snap(p["upsample"], 128, 1024, 2048)
    p["remesh"] = _snap(p["remesh"], 64, 512, 2048)
    p["atlas"] = _snap(p["atlas"], 64, 1024, 8192)
    p["normal_res"] = _snap(p["normal_res"], 64, 1024, 8192)
    p["project_res"] = _snap(p["project_res"], 64, 1024, 8192)
    p["ao_res"] = _snap(p["ao_res"], 64, 512, 8192)
    p["ao_samples"] = _snap(p["ao_samples"], 4, 64, 1024)
    p["view_res"] = _snap(p["view_res"], 64, 1024, 8192)
    p["faces"] = _snap(p["faces"], 100_000, 1_000_000, 20_000_000)
    p["quads"] = _snap(p["quads"], 1000, 8_000, 80_000)
    return p, tier


def _report(mode, gb, name, tier, profile):
    head = f"{mode}: {name}, {gb:.1f} ГБ, профиль {tier}" if mode == "авто" else f"ручные настройки ({name}, {gb:.1f} ГБ)"
    body = (
        f"шаги {profile['structure_steps']}/{profile['shape_steps']}/{profile['detail_steps']}/{profile['texture_steps']}, "
        f"форма {profile['upsample']}, ремеш {profile['remesh']}, грани {profile['faces']}, "
        f"атлас {profile['atlas']}, нормали {profile['normal_res']}, проекция {profile['project_res']}, "
        f"AO {profile['ao_res']}×{profile['ao_samples']}, кадры {profile['view_res']}, квады {profile['quads']}"
    )
    return head + "\n" + body


class GarageKitQualityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto считает видеопамять и ставит потолок этой карты. manual берёт числа ниже.",
                }),
                "structure_steps": ("INT", {"default": 28, "min": 8, "max": 60, "step": 1}),
                "shape_steps": ("INT", {"default": 22, "min": 8, "max": 60, "step": 1}),
                "detail_steps": ("INT", {"default": 18, "min": 8, "max": 60, "step": 1}),
                "texture_steps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "upsample": ("INT", {"default": 1536, "min": 1024, "max": 2048, "step": 128}),
                "remesh": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 64}),
                "faces": ("INT", {"default": 4_500_000, "min": 1_000_000, "max": 20_000_000, "step": 100_000}),
                "atlas": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 64}),
                "normal_res": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 64}),
                "project_res": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 64}),
                "ao_res": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "ao_samples": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 4}),
                "view_res": ("INT", {"default": 2048, "min": 1024, "max": 8192, "step": 64}),
                "quads": ("INT", {"default": 18_000, "min": 8_000, "max": 80_000, "step": 1000}),
            }
        }

    RETURN_TYPES = ("INT",) * len(_KEYS)
    RETURN_NAMES = _KEYS
    FUNCTION = "run"
    CATEGORY = "3d/garage_kit"
    OUTPUT_NODE = True

    def run(self, mode, **manual):
        gb, name = detect_vram()
        if mode == "auto" and gb > 0:
            profile, tier = profile_for_vram(gb)
            label = "авто"
        else:
            profile = {key: int(manual[key]) for key in _KEYS}
            tier = "ручной"
            label = "ручные"
        text = _report(label, gb, name, tier, profile)
        print("[quality]", text.replace("\n", " | "))
        values = tuple(int(profile[key]) for key in _KEYS)
        return {"ui": {"text": [text]}, "result": values}


NODE_CLASS_MAPPINGS = {"GarageKitQuality": GarageKitQualityNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GarageKitQuality": "Качество · авто или вручную"}
