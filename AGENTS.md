# Контекст проекта: 3D Garage Kit Master Pipeline (ComfyLLM 3D)

## 1. Текущий статус проекта
- **Репозиторий GitHub**: `https://github.com/NikMarvel/ComfyUI-Trellis2-3D-Master.git`
- **Пайплайн воркфлоу**: `Antigravity_Edit_8K_Ultimate.json` и `ComfyUI/user/default/workflows/Antigravity Edit.json`.
- **Всего нод**: 112 нод, 181 связь.
- **Статус аудита**: 0 критических ошибок, граф 100% валиден и верифицирован `verify_nodes.py`.
- **Интеграция SUPIR (Generative Prior)**: SUPIR только для фотопроекции (denoise 0.30, 24 шага, CFG 3.5), затем UltraSharp и масштаб 2048. В Pixal3D идёт чистый кроп 1024. Карта — RTX 3060 Laptop 6 ГБ, динамический VRAM, `--reserve-vram 0.8`.
- **Модульная генерация**: Создан оркестратор `utils/generate_modular_garage_kit.py` и конвертер `utils/wf_to_prompt.py`. Поддерживается поочередная генерация деталей (Body, Hair, Blindfold, Dress, Gloves, Boots) с контролем памяти GPU через `/free` (Unload Models + Flush Memory) для предотвращения OOM на 6 ГБ VRAM (RTX 3060).
- **Экспорт 3D-печати**: Все меши автоматически конвертируются из GLB в монолитный бинарный STL (`utils/glb_to_stl.py`).

## 2. Ключевые параметры геометрии и запекания
- **Сэмплеры диффузии (ноды 3, 18, 23, 12)**: `euler` (flow-модель Pixal3D). Структура 28 / `normal`, форма 22 / `normal`, детализация 18 / `simple`, текстура 24 / `normal`, CFG текстуры 1.5. Апскейл формы — 1536. На форме (нода 279) `CFGOverride.start_percent=1.0`: привязка к референсу держится до последнего шага, иначе повязка на глазах рвётся.
- **Гибридное двухпроходное текстурирование (нода 453, ImageCompositeMasked)**: объединяет воксельный базовый цвет (`BakeTextureFromVoxel`, нода 147) со сверхрезкой 4K фотопроекцией камер (`BakeTextureFromViews`, нода 417, blend_exponent=12.0) по маске уверенности `confidence`.
- **RemeshMesh (нода 241)**: `resolution=1536`, `sign_mode='solid'`, `fill=28.0`, `band=0.8`, `project_back=0.78`, `fix_poles=True`, `smooth_iters=1`, `drop_small_components=0.002`. Выше fill меньше затягивает крупный вырез на спине. project_back 0.78 возвращает перья, бант и каблук.
- **DecimateMesh (нода 186)**: `target_face_count=4,500,000`, `placement_mode=midpoint`.
- **Глаза (нода 468, `GarageKitEyes`)**: если на лице два миндаля, а не повязка, рваная корка века снимается и вместо неё ставятся глазное яблоко и веки. Ресниц нет. Сплошная повязка 2B не трогается.
- **Топология (нода 464, `GarageKitTopology`)**: выбор `quads` / `triangles`. Квады — QuadriFlow, а если сетка не многообразие (перья, вырез), Blender склеивает треугольники в квады и оставляет отверстия. UV со швами по острым рёбрам. Файлы: `output/3d/2B_Anim_quad.obj` и `output/3d/2B_Anim_rig.glb` (тепловые веса, до 4 костей на вершину). Экраны 465–467 без текстуры: UV, риг, полигональная сетка. Печатный GLB ноды 430 не подменяется.
- **Фотопроекция (нода 417)**: `blend_exponent=12.0`. Камера FOV 20°, дистанция 3.12, вход `mesh` от развёртки. Кроп: pad 1.1, grow_mask 0 (без белого ореола студии). UV padding 8.
- **BakeTextureFromVoxel (нода 147)**: `reference_mesh=None` (отключен для устранения белой маски).
- **BakeNormalMapFromMesh (нода 224)**: `resolution=4096`.
- **DINOv3 CLIP Vision (нода 15)**: `dino_v3_L_naf_fp32.safetensors`.

## 3. Файлы и утилиты
- `utils/generate_modular_garage_kit.py` — послойный конвейер и мастер-генератор.
- `utils/wf_to_prompt.py` — конвертер UI воркфлоу в API Prompt ComfyUI.
- `utils/glb_to_stl.py` — бинарный конвертер GLB в STL.
- `utils/verify_nodes.py` — автоматический аудитор целостности графа.
- `utils/auto_detect_layers.py` — декомпозитор ракурсов по слоям.
- `STRUCTURE.md` — актуализированная карта репозитория.
