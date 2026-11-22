"""
Конвертер UI Workflow в API Prompt.

Виджет control_after_generate (fixed/increment/...) есть только в UI и стоит
между seed и steps. Пропущенный advanced-виджет берётся из default схемы.
"""
import json
import urllib.request

SKIP_TYPES = {"Note", "MarkdownNote", "Preview3DAdvanced", "Save3DAdvanced"}
CONTROL_AFTER_GENERATE = {"fixed", "increment", "decrement", "randomize"}
NUMBER_KINDS = {"INT", "FLOAT"}
WIDGET_KINDS = NUMBER_KINDS | {"STRING", "BOOLEAN", "COMBO", "COLOR", "COMFY_DYNAMICCOMBO_V3"}


def _kind(spec):
    if isinstance(spec, list) and spec:
        return spec[0]
    return None


def _default(spec):
    if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("default", None)
    return None


def _take_widget(widgets, w_idx, spec):
    kind = _kind(spec)
    if isinstance(kind, str) and kind in NUMBER_KINDS:
        while w_idx < len(widgets) and widgets[w_idx] in CONTROL_AFTER_GENERATE:
            w_idx += 1
    if w_idx < len(widgets):
        return widgets[w_idx], w_idx + 1
    return _default(spec), w_idx


def parse_node_inputs(n, links, object_info):
    ntype = n.get("type")
    if ntype not in object_info:
        return None

    schema = object_info[ntype]
    req_inputs = schema.get("input", {}).get("required", {})
    opt_inputs = schema.get("input", {}).get("optional", {})

    node_inputs = {}

    for inp in n.get("inputs", []):
        name = inp.get("name")
        lid = inp.get("link")
        if lid is not None and lid in links:
            link = links[lid]
            node_inputs[name] = [str(link[1]), link[2]]

    named = n.get("widgets_values_named") or {}
    if named:
        for key, value in named.items():
            if key == "control_after_generate" or value is None:
                continue
            root = key.split(".", 1)[0]
            parent = node_inputs.get(root)
            if key in node_inputs or isinstance(parent, list):
                continue
            node_inputs[key] = value
        return node_inputs

    widgets = n.get("widgets_values", [])
    if not isinstance(widgets, list):
        widgets = []
    linked_widgets = {
        inp.get("name")
        for inp in n.get("inputs", [])
        if inp.get("link") is not None and inp.get("widget")
    }
    w_idx = 0
    all_keys = list(req_inputs.keys()) + list(opt_inputs.keys())
    for key in all_keys:
        spec = req_inputs.get(key) or opt_inputs.get(key)
        if spec is None:
            continue
        kind = _kind(spec)
        is_widget = isinstance(kind, list) or (isinstance(kind, str) and kind in WIDGET_KINDS)
        if not is_widget:
            continue
        if key in node_inputs or key in linked_widgets:
            if key in linked_widgets:
                _, w_idx = _take_widget(widgets, w_idx, spec)
            continue

        if kind == "COMFY_DYNAMICCOMBO_V3":
            combo_key, w_idx = _take_widget(widgets, w_idx, spec)
            if combo_key is None:
                continue
            node_inputs[key] = combo_key
            for opt in spec[1].get("options", []):
                if opt.get("key") != combo_key:
                    continue
                sub = opt.get("inputs", {})
                for sub_key in list(sub.get("required", {})) + list(sub.get("optional", {})):
                    sub_spec = sub.get("required", {}).get(sub_key) or sub.get("optional", {}).get(sub_key)
                    value, w_idx = _take_widget(widgets, w_idx, sub_spec)
                    if value is not None:
                        node_inputs[f"{key}.{sub_key}"] = value
                break
            continue

        value, w_idx = _take_widget(widgets, w_idx, spec)
        if value is None and key in opt_inputs:
            continue
        if value is not None:
            node_inputs[key] = value

    return node_inputs

def workflow_to_prompt(wf_path):
    with open(wf_path, encoding="utf-8") as f:
        wf = json.load(f)

    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60) as r:
            object_info = json.loads(r.read())
    except Exception as e:
        print(f"Не удалось подключиться к ComfyUI: {e}")
        return None

    nodes = wf.get("nodes", [])
    links = {l[0]: l for l in wf.get("links", [])}

    prompt = {}

    for n in nodes:
        nid = str(n["id"])
        ntype = n.get("type")
        if ntype in SKIP_TYPES:
            continue
        if ntype not in object_info:
            continue

        inputs = parse_node_inputs(n, links, object_info)
        if inputs is not None:
            prompt[nid] = {
                "class_type": ntype,
                "inputs": inputs,
                "_meta": {"title": n.get("title", ntype)}
            }

    return prompt

if __name__ == "__main__":
    p = workflow_to_prompt("D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json")
    print(f"Сконвертировано нод: {len(p)}")
    print("Node 241 (RemeshMesh):", json.dumps(p["241"]["inputs"], indent=2))
