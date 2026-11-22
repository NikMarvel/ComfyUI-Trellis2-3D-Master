import json

PATHS = [
    r"D:\ComfyLLM\3D\Antigravity_Edit_8K_Ultimate.json",
    r"D:\ComfyLLM\3D\ComfyUI\user\default\workflows\Antigravity Edit.json",
]
POSITIVE = (
    "2B from NieR Automata, white bob hair with a black bow, black blindfold, "
    "black dress, white feather cuffs, same pose and same costume as the reference, "
    "faithful fabric pattern, clean edges, no new accessories"
)
NEGATIVE = (
    "blurry, jpeg artifacts, blocking, extra fingers, extra limbs, new patterns, "
    "changed costume, changed face, changed hair, asymmetric redesign, "
    "hallucinated details, oversharpened, plastic skin, deformed, watermark, text"
)


def patch(path):
    wf = json.load(open(path, encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}

    for nid in (312, 349, 350, 352):
        n = nodes[nid]
        n["widgets_values"] = [1024, 1024, 1.1, 8, "#000000"]
        n["widgets_values_named"] = {
            "width": 1024, "height": 1024, "pad_factor": 1.1,
            "grow_mask": 8, "background": "#000000",
        }
        n["title"] = "Кроп 1024 · pad 1.1 · grow 8"

    n = nodes[241]
    n["widgets_values"] = [1280, "solid", 28.0, False, 1.0, 0.62, True, 2, 0.01, 20000000]
    n["widgets_values_named"] = {
        "resolution": 1280, "sign_mode": "solid", "sign_mode.fill": 28.0,
        "sign_mode.qef": False, "band": 1.0, "project_back": 0.62,
        "fix_poles": True, "smooth_iters": 2, "drop_small_components": 0.01,
        "precluster_max_verts": 20000000,
    }
    n["title"] = "Remesh 1280 · solid · project 0.62 · fill 28"

    n = nodes[417]
    n["widgets_values"] = [4096, 5.0, False]
    n["widgets_values_named"] = {"resolution": 4096, "blend_exponent": 5.0, "fill_unseen": False}
    n["title"] = "Фотопроекция 4096 · blend 5.0"

    supir = [42, "fixed", 24, 3.5, "dpmpp_2m", "karras", 0.30]
    named = {
        "seed": 42, "control_after_generate": "fixed", "steps": 24, "cfg": 3.5,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.30,
    }
    for nid, view in ((438, "Front"), (442, "Left"), (446, "Back"), (450, "Right")):
        nodes[nid]["widgets_values"] = supir
        nodes[nid]["widgets_values_named"] = dict(named)
        nodes[nid]["title"] = f"{view} · SUPIR 24 · denoise 0.30"

    nodes[434]["widgets_values"] = [POSITIVE]
    nodes[434]["widgets_values_named"] = {"text": POSITIVE}
    nodes[435]["widgets_values"] = [NEGATIVE]
    nodes[435]["widgets_values_named"] = {"text": NEGATIVE}

    json.dump(wf, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("patched", path)

for p in PATHS:
    patch(p)
