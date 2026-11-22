# Структура проекта (ComfyLLM 3D)

Данный репозиторий / рабочее пространство представляет собой среду ComfyUI для 3D-генерации высокодетализированных мастер-моделей фигурок (Garage Kit), персонажей и ассетов для 3D-печати на базе TRELLIS 2, Pixal3D и Hunyuan3D-2 с поддержкой 4K/8K UV-развертки, полигональной плотностью 2.5M+ полигонов, нейросети метрической глубины MoGe 2, генеративного апскейлера SUPIR и модульной послойной сборки с автоматическим распознаванием деталей.

```text
3D/
├── custom_nodes/                          # Пользовательские ноды Garage Kit для ComfyUI
│   ├── garage_kit_eyes.py                 # Процедурная замена поврежденных век/глаз
│   ├── garage_kit_quality.py              # Анализатор метрик качества и полигональной сетки
│   ├── garage_kit_splitter_node.py        # Модульная нарезка фигурки на слои
│   └── garage_kit_topology.py             # Квадратная авторетопология (QuadriFlow / Blender) и риггинг
├── .launcher/                             # Скрипты и конфигурации запуска среды
├── Antigravity_Edit_8K_Ultimate.json       # Ультимативный 4-ракурсный пайплайн (Front, Left, Back, Right) без нарезки
├── Antigravity_SingleImage_MoGe_3D.json    # Пайплайн генерации из 1 картинки с нейросетью глубины MoGe 2
├── ComfyUI/                               # Ядро ComfyUI с поддержкой 3D-нод
│   ├── comfy/                             # Базовые модули диффузии, сэмплеры, VAE, SUPIR patcher
│   ├── comfy_extras/                      # Дополнительные ноды (SUPIRApply, SAM3_Detect, mesh3d)
│   ├── custom_nodes/                      # Пользовательские расширения
│   │   ├── garage_kit_splitter_node.py    # Опциональная нода 3D-сегментации
│   │   └── websocket_image_save.py        # Стриминг превью по вебсокету
│   ├── models/                            # Директории весов моделей
│   │   ├── checkpoints/                   # Базовые модели диффузии (SDXL Base 1.0, SAM 3.1)
│   │   ├── model_patches/                 # Патчи моделей (SUPIR-v0Q.ckpt)
│   │   ├── upscale_models/                # Модели супер-разрешения (4x-UltraSharp.pth)
│   │   └── vae/                           # Энкодеры/декодеры VAE (sdxl_vae.safetensors)
│   └── user/default/workflows/            # Пользовательские воркфлоу в интерфейсе ComfyUI Desktop
│       ├── Antigravity Edit.json          # Рабочий 4-ракурсный воркфлоу (2.5M Poly Master-Model)
│       └── Antigravity SingleImage MoGe 3D.json # Воркфлоу: 1 картинка -> MoGe 2 глубина -> 3D STL
├── logs/                                  # Системные журналы выполнения (comfyui.log)
├── standalone-env/                        # Изолированное окружение Python и зависимости
├── utils/                                 # Вспомогательные утилиты и экспортеры
│   ├── apply_pipeline_fixes.py            # Автоматическая калибровка связей и геометрии воркфлоу
│   ├── auto_detect_layers.py              # Автономный семантический декомпозитор элементов фигурки
│   ├── download_supir_models.py           # Фоновый менеджер загрузки весов SUPIR и SDXL
│   ├── generate_modular_garage_kit.py     # Послойный генератор STL деталей с контролем памяти
│   ├── glb_to_stl.py                      # Высокоточный бинарный экспортер GLB -> STL
│   ├── inspect_mesh.py                    # Анализатор топологии, нормалей и текстур меша
│   ├── inspect_wf.py                      # Инспектор графа и параметров нод ComfyUI
│   ├── make_comparison.py                 # Генератор макро-сравнений 1024 vs 4K
│   ├── render_4k_upscales.py              # Автономный рендер 4K ракурсов с тайлингом
│   ├── split_garage_kit.py                # Standalone CLI утилита нарезки на детали
│   ├── verify_nodes.py                    # Автоматический аудитор целостности нод и связей (Node Integrity Auditor)
│   └── wf_to_prompt.py                    # Конвертер UI воркфлоу в исполняемый API Prompt ComfyUI
├── manifest.json                          # Манифест окружения
├── requirements-nvidia.txt                # Зависимости для GPU NVIDIA
└── STRUCTURE.md                           # Карта структуры проекта
```
