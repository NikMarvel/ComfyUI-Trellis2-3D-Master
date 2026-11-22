"""Resume-capable download of the Pixal3D weights this workflow runs."""
import os
import sys
import time
import urllib.request

DOWNLOADS = [
    (
        "birefnet.safetensors",
        "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors",
        r"D:\ComfyLLM\3D\ComfyUI\models\background_removal",
    ),
    (
        "trellis_2_texture_vae_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors",
        r"D:\ComfyLLM\3D\ComfyUI\models\vae",
    ),
    (
        "trellis_2_shape_vae_bf16.safetensors",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors",
        r"D:\ComfyLLM\3D\ComfyUI\models\vae",
    ),
    (
        "dino_v3_L_naf_fp32.safetensors",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors",
        r"D:\ComfyLLM\3D\ComfyUI\models\clip_vision",
    ),
    (
        "pixal3d_multiview_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_multiview_int8_convrot.safetensors",
        r"D:\ComfyLLM\3D\ComfyUI\models\diffusion_models",
    ),
]


def download(name, url, folder):
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, name)
    part = target + ".part"
    downloaded = os.path.getsize(part) if os.path.exists(part) else 0
    if os.path.exists(target) and os.path.getsize(target) > 1_000_000 and not os.path.exists(part):
        print(f"[OK] {name} already {os.path.getsize(target)/1e6:.1f} MB", flush=True)
        return
    for attempt in range(1, 8):
        downloaded = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        req = urllib.request.Request(url, headers=headers)
        print(f"[GET] {name} from {downloaded/1e6:.1f} MB (try {attempt})", flush=True)
        t0 = time.time()
        last = t0
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(part, "ab" if downloaded else "wb") as f:
                if downloaded and getattr(resp, "status", 200) == 200:
                    f.seek(0)
                    f.truncate()
                    downloaded = 0
                total_hdr = resp.headers.get("Content-Length")
                total = int(total_hdr) + downloaded if total_hdr else 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last >= 8:
                        speed = downloaded / max(now - t0, 0.1) / 1e6
                        pct = f"{100 * downloaded / total:.1f}%" if total else "?"
                        print(f"  {name}: {downloaded/1e6:.1f} MB {pct} {speed:.2f} MB/s", flush=True)
                        last = now
            if total and downloaded < total - 4096:
                raise TimeoutError(f"short read {downloaded} < {total}")
            break
        except Exception as exc:
            print(f"[RETRY] {name}: {exc}", flush=True)
            time.sleep(3)
    else:
        raise SystemExit(f"failed: {name}")
    os.replace(part, target)
    print(f"[DONE] {name} {os.path.getsize(target)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:] 
    for name, url, folder in DOWNLOADS:
        if only and name not in only:
            continue
        download(name, url, folder)
    print("ALL_DONE", flush=True)
