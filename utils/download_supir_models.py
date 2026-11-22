"""
Фоновый загрузчик моделей SUPIR и SDXL (SUPIR Model Downloader)

Архитектурная роль:
- Скачивает веса с HuggingFace с поддержкой докачки (Range/chunk streaming):
  1. SUPIR-v0Q.ckpt (5.32 GB) -> models/model_patches/
  2. sd_xl_base_1.0_0.9vae.safetensors (6.93 GB) -> models/checkpoints/
  3. sdxl_vae.safetensors (334 MB) -> models/vae/
"""

import os
import sys
import time
import urllib.request

DOWNLOADS = [
    {
        "name": "sdxl_vae.safetensors",
        "url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
        "target_dir": r"D:\ComfyLLM\3D\ComfyUI\models\vae",
        "expected_size": 334641162,
    },
    {
        "name": "SUPIR-v0Q.ckpt",
        "url": "https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt",
        "target_dir": r"D:\ComfyLLM\3D\ComfyUI\models\model_patches",
        "expected_size": 5329810432,
    },
    {
        "name": "sd_xl_base_1.0_0.9vae.safetensors",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
        "target_dir": r"D:\ComfyLLM\3D\ComfyUI\models\checkpoints",
        "expected_size": 6938078334,
    },
]

def download_file(url, target_path, expected_size):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = target_path + ".part"

    downloaded = 0
    if os.path.exists(temp_path):
        downloaded = os.path.getsize(temp_path)
    elif os.path.exists(target_path):
        current_size = os.path.getsize(target_path)
        if current_size == expected_size:
            print(f"[OK] Файл уже полностью загружен: {target_path}")
            return
        downloaded = current_size
        os.rename(target_path, temp_path)

    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"Возобновление загрузки с {downloaded / (1024*1024):.1f} МБ...")

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_print = t0

    mode = "ab" if downloaded > 0 else "wb"
    with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, mode) as f:
        chunk_size = 1024 * 1024 # 1 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 5.0: # print every 5 sec
                elapsed = now - t0
                speed = (len(chunk) / (1024*1024)) / max(now - last_print, 0.001)
                percent = (downloaded / expected_size) * 100 if expected_size else 0
                print(f"  Прогресс: {downloaded/(1024*1024):.1f} / {expected_size/(1024*1024):.1f} МБ ({percent:.1f}%) | Скорость: ~{speed:.2f} МБ/с")
                last_print = now

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"✅ Успешно загружен: {target_path}")

def main():
    print("=== ЗАПУСК ЗАГРУЗКИ ВЕСОВ ДЛЯ SUPIR И SDXL ===")
    for item in DOWNLOADS:
        t_path = os.path.join(item["target_dir"], item["name"])
        print(f"\nНачало загрузки: {item['name']} ({item['expected_size']/(1024*1024):.1f} МБ)")
        try:
            download_file(item["url"], t_path, item["expected_size"])
        except Exception as e:
            print(f"❌ Ошибка при загрузке {item['name']}: {e}")
            sys.exit(1)
    print("\n🎉 Все веса SUPIR и SDXL успешно установлены в соответствующие директории!")

if __name__ == '__main__':
    main()
