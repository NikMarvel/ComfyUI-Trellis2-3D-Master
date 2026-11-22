"""
Модульный конвейер послойной 3D-генерации персонажа (Modular Garage Kit Generator).

Архитектурная роль:
- Автоматически оркестрирует поочередную генерацию изолированных частей фигурки
  (Тело, Волосы/Бант, Повязка, Платье/Корсет, Перчатки, Сапоги) через ComfyUI API.
- Поддерживает генерацию единой ультимативной мастер-модели (Master 8K 2.5M Poly).
- Гарантирует изоляцию видеопамяти (VRAM Flush) между слоями через endpoint /free,
  предотвращая OOM на GPU с 6 ГБ VRAM (RTX 3060 Laptop).
- Автоматически конвертирует все выходные GLB-меши в бинарные STL-файлы для 3D-печати.
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import urllib.error
from datetime import datetime

# Подключаем локальные модули
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from wf_to_prompt import workflow_to_prompt
from glb_to_stl import convert_glb_to_stl

COMFY_HOST = "http://127.0.0.1:8188"
INPUT_DIR = r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"
OUTPUT_DIR = r"C:\Users\nikit\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
MASTER_WF = "D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json"

LAYERS_DEF = [
    {
        "id": "01_body_base",
        "name": "Анатомическая База Тела (Body Base)",
        "output_prefix": "3d/GarageKit_Modular/01_Body_Base",
    },
    {
        "id": "02_hair",
        "name": "Прическа и Бант (Hair & Bow)",
        "output_prefix": "3d/GarageKit_Modular/02_Hair_Bow",
    },
    {
        "id": "03_blindfold",
        "name": "Тактическая Повязка (Blindfold)",
        "output_prefix": "3d/GarageKit_Modular/03_Blindfold",
    },
    {
        "id": "04_dress_corset",
        "name": "Готическое Платье (Gothic Dress & Corset)",
        "output_prefix": "3d/GarageKit_Modular/04_Dress_Corset",
    },
    {
        "id": "05_gloves",
        "name": "Перчатки с Раструбами (Feathered Gloves)",
        "output_prefix": "3d/GarageKit_Modular/05_Gloves",
    },
    {
        "id": "06_boots",
        "name": "Ботфорты на Шпильке (High Boots)",
        "output_prefix": "3d/GarageKit_Modular/06_Boots",
    },
]


def free_vram():
    """Сбрасывает видеопамять и кэш моделей в ComfyUI."""
    try:
        req = urllib.request.Request(
            f"{COMFY_HOST}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                print("  [VRAM] Память GPU успешно очищена (Unload Models + Flush Memory).")
    except Exception as e:
        print(f"  [VRAM Предупреждение] Не удалось вызвать /free: {e}")


def wait_for_execution(prompt_id, timeout_sec=1800):
    """Ожидает завершения генерации промпта в ComfyUI."""
    print(f"  Ожидание завершения генерации (Prompt ID: {prompt_id})...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_sec:
        time.sleep(4)
        try:
            with urllib.request.urlopen(f"{COMFY_HOST}/history/{prompt_id}", timeout=5) as r:
                data = json.loads(r.read())
                if prompt_id in data:
                    item = data[prompt_id]
                    status = item.get("status", {})
                    if status.get("completed", False):
                        print(f"  [OK] Генерация успешно завершена за {int(time.time() - start_time)} сек.")
                        return item
                    if status.get("status_str") == "error":
                        print(f"  [ОШИБКА] Ошибка генерации: {status.get('messages')}")
                        return None
        except Exception:
            pass

    print(f"  [ТАЙМАУТ] Превышено время ожидания ({timeout_sec} сек).")
    return None


def post_prompt(prompt_dict):
    """Отправляет граф промпта на сервер ComfyUI."""
    payload = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        res = json.loads(r.read())
        return res.get("prompt_id")


def run_master_generation(dry_run=False):
    """Генерация полной мастер-модели 8K (2.5M Poly)."""
    print("\n=======================================================")
    print("  ЗАПУСК ГЕНЕРАЦИИ ПОЛНОЙ МАСТЕР-МОДЕЛИ (FULL MASTER 8K)")
    print("=======================================================")
    
    prompt = workflow_to_prompt(MASTER_WF)
    if not prompt:
        print("Ошибка: не удалось преобразовать воркфлоу в промпт.")
        return False

    prompt["430"]["inputs"]["filename_prefix"] = "3d/2B_Figure_Full_Master_8K_Ultra_Detail"
    
    if dry_run:
        print("  [DRY-RUN] Промпт валиден. Всего нод для исполнения:", len(prompt))
        return True

    free_vram()
    prompt_id = post_prompt(prompt)
    print(f"  Задание поставлено в очередь. ID: {prompt_id}")
    res = wait_for_execution(prompt_id)
    free_vram()
    return res is not None


def run_modular_generation(dry_run=False):
    """Поочередная генерация всех 6 слоев персонажа."""
    print("\n=======================================================")
    print("  ЗАПУСК ПОСЛОЙНОЙ МОДУЛЬНОЙ ГЕНЕРАЦИИ (GARAGE KIT)")
    print("=======================================================")
    
    base_prompt = workflow_to_prompt(MASTER_WF)
    if not base_prompt:
        print("Ошибка загрузки базового воркфлоу.")
        return False

    seg_dir = os.path.join(OUTPUT_DIR, "segmented_layers")
    if not os.path.exists(seg_dir):
        print(f"Папка сегментированных слоев не найдена: {seg_dir}")
        print("Запуск auto_detect_layers.py...")
        from auto_detect_layers import analyze_character_layers
        analyze_character_layers()

    # Убеждаемся, что выходная папка существует
    os.makedirs(os.path.join(OUTPUT_DIR, "3d", "GarageKit_Modular"), exist_ok=True)

    for idx, layer in enumerate(LAYERS_DEF, 1):
        layer_id = layer["id"]
        layer_name = layer["name"]
        prefix = layer["output_prefix"]
        
        print(f"\n--- [{idx}/{len(LAYERS_DEF)}] Генерация слоя: {layer_name} ---")
        
        layer_folder = os.path.join(seg_dir, layer_id)
        # Копируем 4 ракурса слоя во входную директорию ComfyUI
        vnames = {"front": "338", "left": "341", "back": "342", "right": "345"}
        layer_images = {}
        for vn, nid in vnames.items():
            src_img = os.path.join(layer_folder, f"{vn}.png")
            dst_name = f"layer_{layer_id}_{vn}.png"
            dst_path = os.path.join(INPUT_DIR, dst_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_path)
                layer_images[nid] = dst_name

        # Клонируем промпт и подменяем ракурсы и имя экспорта
        prompt = json.loads(json.dumps(base_prompt))
        for nid, img_name in layer_images.items():
            prompt[nid]["inputs"]["image"] = img_name

        # Настраиваем имя выходного файла
        prompt["430"]["inputs"]["filename_prefix"] = prefix
        
        if dry_run:
            print(f"  [DRY-RUN] Слой {layer_name} готов к отправке. Файл: {prefix}")
            continue

        # Освобождаем память перед каждым слоем
        free_vram()
        
        prompt_id = post_prompt(prompt)
        print(f"  Промпт отправлен (ID: {prompt_id}). Обработка...")
        res = wait_for_execution(prompt_id)
        
        if res:
            # Ищем сгенерированный GLB и конвертируем в STL
            target_glb = os.path.join(OUTPUT_DIR, f"{prefix}_00001_.glb")
            if not os.path.exists(target_glb):
                # Поиск по шаблону
                for f in os.listdir(os.path.join(OUTPUT_DIR, "3d", "GarageKit_Modular")):
                    if f.startswith(os.path.basename(prefix)) and f.endswith(".glb"):
                        target_glb = os.path.join(OUTPUT_DIR, "3d", "GarageKit_Modular", f)
                        break

            if os.path.exists(target_glb):
                stl_file = convert_glb_to_stl(target_glb)
                print(f"  ✅ Деталь успешно сохранена в STL: {stl_file}")

        # Финальная очистка памяти после слоя
        free_vram()
        time.sleep(2)

    print("\n🎉 ВСЕ МОДУЛЬНЫЕ ДЕТАЛИ УСПЕШНО СГЕНЕРИРОВАНЫ!")
    return True


if __name__ == "__main__":
    mode = "all"
    dry = False
    
    if len(sys.argv) > 1:
        if "--dry-run" in sys.argv:
            dry = True
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]

    print(f"Режим запуска: {mode} (Dry-Run: {dry})")

    if mode in ["master", "all"]:
        run_master_generation(dry_run=dry)
    if mode in ["modular", "all"]:
        run_modular_generation(dry_run=dry)
