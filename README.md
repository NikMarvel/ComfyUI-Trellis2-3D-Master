# ComfyUI 3D Garage Kit Master Pipeline (8K / 4.5M Poly + Quad Retopology)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Supported-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Trellis 2](https://img.shields.io/badge/Trellis.2-Pixal3D-green.svg)](https://github.com/microsoft/TRELLIS)
[![VRAM](https://img.shields.io/badge/VRAM-6GB%20Optimized-orange.svg)](#hardware-requirements)
[![3D Print](https://img.shields.io/badge/3D%20Print-Watertight%20STL-purple.svg)](#3d-printing-export)

Профессиональный конвейер (пайплайн) генерации высокодетализированных коллекционных фигурок (Garage Kit), аниме-персонажей и мастер-моделей для 3D-печати на базе **ComfyUI**, **Microsoft TRELLIS.2**, **Pixal3D Multi-View**, генеративного апскейлера **SUPIR**, процедурного восстановления глаз, квад-авторетопологии и гибридного фототекстурирования **4K PBR**.

Оптимизирован для запуска на видеокартах среднего уровня с жестким лимитом памяти (**NVIDIA RTX 3060 Laptop 6 ГБ VRAM**) с автоматическим управлением кэшем.

---

## 🌟 Ключевые возможности

1. **Генеративный апскейл ракурсов (SUPIR + 4x-UltraSharp)**:
   - Синтез микродеталей (текстура ткани, кружево, анатомические линии, кожа, волосы) через латентную диффузию SDXL Base (denoise 0.30, 24 шага, CFG 3.5, масштаб 2048).
2. **Мультиракурсное 3D-кондиционирование (Pixal3D + DINOv3 ViT-L)**:
   - 4 ортогональных ракурса (Front 0°, Left 90°, Back 180°, Right 270°) с чистым 1024 кропом и извлечением признаков DINOv3.
3. **Ультра-детализированная геометрия (до 4 500 000 полигонов)**:
   - `RemeshMesh`: плотность воксельной сетки 1536, алгоритм CelloCut Solid Manifold (`fill=28.0`, `band=0.8`, `project_back=0.78`). Идеально сохраняет крупный вырез на спине, перья, бант и тонкие каблуки.
4. **Процедурная реставрация глаз (`GarageKitEyes`)**:
   - Автоматически выявляет миндалевидные зоны глаз на лице, удаляет рваную диффузионную корку век и встраивает чистое глазное яблоко с анатомическими веками. Сплошную повязку 2B оставляет нетронутой.
5. **Квадратная топология и авто-риг (`GarageKitTopology`)**:
   - Переключение между треугольной печатной сеткой (Tris) и чистыми четырехугольниками (Quads через QuadriFlow / Blender).
   - Экспорт готового рига с костями и тепловыми весами вершин (`2B_Anim_rig.glb`) для анимации.
6. **Гибридный двухпроходный PBR-пак**:
   - Объединяет объемную воксельную текстуру скрытых зон с фотопроекцией камер 4096×4096 (`blend_exponent=12.0`) по непрерывной маске уверенности.
   - Запекание 4K карты нормалей поверхности (`BakeNormalMapFromMesh`) и Ambient Occlusion.
7. **Модульная генерация (Garage Kit Layer Decomposer)**:
   - Автоматическая поочередная генерация отдельных частей (Тело, Волосы, Повязка, Платье, Перчатки, Сапоги) с полным сбросом видеопамяти (`/free`) между этапами.
8. **Прямой экспорт в STL**:
   - Автоматическая конвертация каждого меша из GLB в монолитный бинарный STL, готовый к слайсингу (Chitubox, Bambu Studio, Lychee, Cura, PrusaSlicer).

---

## 🛠️ Системные требования

- **GPU**: NVIDIA RTX 3060 (6 ГБ VRAM) или мощнее.
- **RAM**: 16–32 ГБ.
- **Окружение**: ComfyUI (версия 0.46.2+) с установленным PyTorch 2.x CUDA.

---

## 📦 Необходимые модели (Weights)

Поместите веса моделей в соответствующие папки `ComfyUI/models/`:

| Модель | Файл | Папка размещения |
|---|---|---|
| **SDXL Base 1.0** | `sd_xl_base_1.0_0.9vae.safetensors` | `models/checkpoints/` |
| **SUPIR Patch** | `SUPIR-v0Q.ckpt` | `models/model_patches/` |
| **SDXL VAE** | `sdxl_vae.safetensors` | `models/vae/` |
| **DINOv3 ViT-L** | `dino_v3_L_naf_fp32.safetensors` | `models/clip_vision/` |
| **Trellis 2 Shape VAE** | `trellis_2_shape_vae_bf16.safetensors` | `models/vae/` |
| **Trellis 2 Texture VAE** | `trellis_2_texture_vae_bf16.safetensors` | `models/vae/` |
| **Pixal3D DiT** | `pixal3d_multiview_int8_convrot.safetensors` | `models/unet/` |
| **4x-UltraSharp** | `4x-UltraSharp.pth` | `models/upscale_models/` |

---

## 🚀 Быстрый старт

### Способ 1: Через веб-интерфейс ComfyUI
1. Откройте ComfyUI.
2. Перетащите файл `Antigravity_Edit_8K_Ultimate.json` в окно браузера.
3. В нодах **LoadImage** (Front, Left, Back, Right) выберите ваши 4 ракурса персонажа.
4. Нажмите **Queue Prompt** (`Ctrl + Enter`).
5. Результат сохранится в `output/3d/` как GLB мастер-модель.

### Способ 2: Консольный модульный генератор (Garage Kit)
Генерация всех деталей по отдельности с автоматическим контролем памяти:
```bash
python utils/generate_modular_garage_kit.py --mode modular
```

Конвертация любого GLB в бинарный STL:
```bash
python utils/glb_to_stl.py path/to/model.glb
```

Проверка целостности нод воркфлоу:
```bash
python utils/verify_nodes.py
```

---

## 📁 Структура проекта

Подробная карта модулей и утилит представлена в [STRUCTURE.md](STRUCTURE.md).

---

## 📄 Лицензия

MIT License. Свободно для личного и коммерческого использования.
