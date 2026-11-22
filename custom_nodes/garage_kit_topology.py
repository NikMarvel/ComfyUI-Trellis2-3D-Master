"""
Топология для анимации: треугольники или квады, UV, авто-риг и три экрана.

Квады собирает QuadriFlow в Blender (границы выреза и разреза сохраняются),
затем швы по острым рёбрам и unwrap, который открывается в Blender, Maya и других.
Риг — гуманоидный скелет по медиальной оси фигуры и тепловые веса (heat diffusion),
не больше четырёх костей на вершину, как в glTF. Экраны без текстуры: UV-сетка,
риг и полигональная сетка.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw

import folder_paths


BLENDER_CANDIDATES = (
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
)

BLENDER_SCRIPT = textwrap.dedent(
    r"""
    import math
    import sys
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1:]
    src, dst, target_s, margin_s = argv
    target = max(2000, int(target_s))
    margin = float(margin_s)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.wm.obj_import(filepath=src, up_axis="Y", forward_axis="NEGATIVE_Z")
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.dissolve_degenerate(threshold=1e-6)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    area = sum(p.area for p in obj.data.polygons)
    voxel = math.sqrt(max(area, 1e-8) / float(target)) * 1.15
    print("VOXEL", round(voxel, 6), "area", round(area, 5))

    bpy.ops.object.duplicate()
    high = bpy.context.selected_objects[0]
    high.name = "HighTarget"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.modifier_add(type="REMESH")
    rem = obj.modifiers[-1]
    rem.mode = "VOXEL"
    rem.voxel_size = voxel
    rem.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier=rem.name)
    print("VOXEL_FACES", len(obj.data.polygons))

    bpy.ops.object.modifier_add(type="SHRINKWRAP")
    sw = obj.modifiers[-1]
    sw.target = high
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.wrap_mode = "ON_SURFACE"
    bpy.ops.object.modifier_apply(modifier=sw.name)

    bpy.ops.object.modifier_add(type="SMOOTH")
    sm = obj.modifiers[-1]
    sm.factor = 0.55
    sm.iterations = 8
    bpy.ops.object.modifier_apply(modifier=sm.name)

    bpy.data.objects.remove(high, do_unlink=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=margin, scale_to_bounds=True)
    except Exception as exc:
        print("UNWRAP_FAIL", exc)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=margin)
    try:
        bpy.ops.uv.average_islands_scale()
        bpy.ops.uv.pack_islands(margin=margin)
    except Exception as exc:
        print("PACK_FAIL", exc)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=dst,
        export_selected_objects=True,
        export_uv=True,
        export_normals=False,
        export_materials=False,
        export_triangulated_mesh=False,
        up_axis="Y",
        forward_axis="NEGATIVE_Z",
    )
    me = obj.data
    quads = sum(1 for p in me.polygons if len(p.vertices) == 4)
    tris = sum(1 for p in me.polygons if len(p.vertices) == 3)
    print(f"QUAD_STATS {quads} {tris} {len(me.vertices)}")
    """
).strip() + "\n"


def find_blender():
    for path in BLENDER_CANDIDATES:
        if os.path.isfile(path):
            return path
    root = r"C:\Program Files\Blender Foundation"
    if os.path.isdir(root):
        found = []
        for dirpath, _, files in os.walk(root):
            if "blender.exe" in files:
                found.append(os.path.join(dirpath, "blender.exe"))
        if found:
            return sorted(found)[-1]
    return None


def mesh_arrays(mesh):
    verts = mesh.vertices
    faces = mesh.faces
    uvs = getattr(mesh, "uvs", None)
    if isinstance(verts, (list, tuple)):
        v = verts[0]
        f = faces[0]
        uv = None if uvs is None else uvs[0]
    elif verts.ndim == 3:
        vc = getattr(mesh, "vertex_counts", None)
        fc = getattr(mesh, "face_counts", None)
        nv = int(vc[0]) if vc is not None else verts.shape[1]
        nf = int(fc[0]) if fc is not None else faces.shape[1]
        v = verts[0, :nv]
        f = faces[0, :nf]
        uv = None if uvs is None else uvs[0, :nv]
    else:
        v, f, uv = verts, faces, uvs
    v = v.detach().float().cpu().numpy()
    f = f.detach().long().cpu().numpy()
    if uv is not None:
        uv = uv.detach().float().cpu().numpy()
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    return np.ascontiguousarray(v), np.ascontiguousarray(f[keep]), uv


def cluster_decimate(verts, faces, target_verts):
    if verts.shape[0] <= target_verts:
        return verts, faces
    lo = verts.min(axis=0)
    extent = float((verts.max(axis=0) - lo).max())
    # Поверхность занимает около res^2 клеток, а не объём res^3.
    cells = max(16.0, math.sqrt(float(target_verts)))
    scale = cells / max(extent, 1e-8)
    q = np.floor((verts - lo) * scale).astype(np.int64)
    q -= q.min(axis=0)
    span = q.max(axis=0) + 1
    key = (q[:, 0] * int(span[1]) + q[:, 1]) * int(span[2]) + q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    n = int(inv.max()) + 1
    acc = np.zeros((n, 3), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    np.add.at(acc, inv, verts)
    np.add.at(cnt, inv, 1.0)
    nv = (acc / np.maximum(cnt, 1.0)[:, None]).astype(np.float32)
    nf = inv[faces]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 2] != nf[:, 0])
    nf = nf[ok]
    used = np.zeros(n, dtype=bool)
    used[nf.reshape(-1)] = True
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return nv[used], remap[nf]


def write_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for p in verts:
            fh.write(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n")
        for a, b, c in faces:
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


def parse_obj(path):
    verts, uvs, polys = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                u, v = line.split()[1:3]
                uvs.append((float(u), float(v)))
            elif line.startswith("f "):
                corners = []
                for tok in line.split()[1:]:
                    bits = tok.split("/")
                    vi = int(bits[0])
                    vi = vi - 1 if vi > 0 else len(verts) + vi
                    if len(bits) > 1 and bits[1]:
                        ti = int(bits[1])
                        ti = ti - 1 if ti > 0 else len(uvs) + ti
                    else:
                        ti = -1
                    corners.append((vi, ti))
                if len(corners) >= 3:
                    polys.append(corners)
    if not verts or not polys:
        raise RuntimeError("Blender вернул пустой OBJ.")
    pos = np.asarray(verts, dtype=np.float32)
    uv_src = np.asarray(uvs, dtype=np.float32) if uvs else np.zeros((1, 2), dtype=np.float32)
    corner_key = {}
    out_v, out_uv = [], []
    quads, tris = [], []

    def corner(vi, ti):
        key = (vi, ti)
        slot = corner_key.get(key)
        if slot is None:
            slot = len(out_v)
            corner_key[key] = slot
            out_v.append(pos[vi])
            out_uv.append(uv_src[ti] if ti >= 0 else np.zeros(2, dtype=np.float32))
        return slot

    for poly in polys:
        idx = [corner(vi, ti) for vi, ti in poly[:4]]
        if len(poly) == 4:
            quads.append(idx)
            tris.append([idx[0], idx[1], idx[2]])
            tris.append([idx[0], idx[2], idx[3]])
        else:
            fan = [corner(vi, ti) for vi, ti in poly]
            tris_i = fan[:3]
            tris.append(tris_i)
            for k in range(3, len(fan)):
                tris.append([fan[0], fan[k - 1], fan[k]])
    return (
        np.asarray(out_v, dtype=np.float32),
        np.asarray(out_uv, dtype=np.float32),
        np.asarray(tris, dtype=np.int64),
        np.asarray(quads, dtype=np.int64) if quads else np.zeros((0, 4), dtype=np.int64),
    )


def simplify_for_quads(verts, faces, target_faces):
    """QEM сохраняет многообразие. Кластеризация его рвёт, и QuadriFlow тогда отказывается работать."""
    from comfy_extras.mesh3d.postprocess.qem_decimate import qem_simplify

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.from_numpy(np.ascontiguousarray(verts, dtype=np.float32))
    f = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64))
    try:
        rv, rf, _ = qem_simplify(v.to(device), f.to(device), int(target_faces))
    except RuntimeError as exc:
        print(f"[topology] QEM on {device} failed: {exc}")
        if device.type != "cuda":
            raise
        torch.cuda.empty_cache()
        rv, rf, _ = qem_simplify(v, f, int(target_faces))
    out_v = rv.detach().float().cpu().numpy()
    out_f = rf.detach().long().cpu().numpy()
    print(f"[topology] QEM {len(faces)} -> {len(out_f)} faces on {rv.device}")
    return out_v, out_f


def run_blender_quads(verts, faces, target_quads, margin):
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender не найден. QuadriFlow ставится вместе с Blender 4.0 в Program Files.")
    guide = simplify_for_quads(verts, faces, max(int(target_quads) * 4, 60000))
    tmp = tempfile.mkdtemp(prefix="gk_quad_")
    src = os.path.join(tmp, "in.obj")
    dst = os.path.join(tmp, "out.obj")
    script = os.path.join(tmp, "quad.py")
    write_obj(src, guide[0], guide[1])
    with open(script, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(BLENDER_SCRIPT)
    proc = subprocess.run(
        [blender, "--background", "--python", script, "--", src, dst, str(int(target_quads)), f"{margin:.5f}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0 or not os.path.isfile(dst):
        tail = log[-2000:]
        raise RuntimeError(f"QuadriFlow в Blender не собрал сетку.\n{tail}")
    for line in log.splitlines():
        if "QF_RESULT" in line or "QF_FALLBACK" in line or line.startswith("Error") or "Traceback" in line:
            print("[topology]", line)
    stats = ""
    for line in log.splitlines():
        if line.startswith("QUAD_STATS"):
            stats = line
    out_v, out_uv, tris, quads = parse_obj(dst)
    print(f"[topology] {stats or 'QUAD_STATS ?'}  verts {len(out_v)}")
    if len(quads) < max(100, int(target_quads) // 5):
        raise RuntimeError(f"QuadriFlow оставил слишком мало квадов ({len(quads)}). {stats}")
    return out_v, out_uv, tris, quads


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _centroid(pts, fallback):
    if pts.shape[0] < 8:
        return fallback.astype(np.float64)
    return np.median(pts, axis=0)


def _body_axes(v, y, y0, height, origin, up):
    """Тонкая ось торса — вперёд. Знак берётся по чёлке, не по одной вершине банта."""
    torso = v[(y > y0 + 0.55 * height) & (y < y0 + 0.78 * height)]
    if torso.shape[0] < 50:
        torso = v
    xz = torso[:, [0, 2]] - origin[[0, 2]]
    _, evec = np.linalg.eigh(xz.T @ xz)
    axis = _unit(np.array([evec[0, 0], 0.0, evec[1, 0]]))
    if float(np.linalg.norm(axis)) < 0.5:
        axis = np.array([0.0, 0.0, 1.0])
    head = v[y > y0 + 0.84 * height]
    proj = (head - origin) @ axis if head.shape[0] else np.array([1.0])
    pos = proj[proj > 0]
    neg = -proj[proj < 0]
    pos_score = float(np.percentile(pos, 90)) if pos.size > 30 else 0.0
    neg_score = float(np.percentile(neg, 90)) if neg.size > 30 else 0.0
    facing = _unit(axis if pos_score >= neg_score else -axis)
    right = _unit(np.cross(up, facing))
    return facing, right, (v - origin) @ right, (v - origin) @ facing


def fit_humanoid(verts, frame=None):
    """Гуманоидный скелет по срезам фигуры. Y вверх, кости внутри объёма."""
    v = verts.astype(np.float64)
    y = v[:, 1]
    y0 = float(y.min())
    y1 = float(y.max())
    height = max(y1 - y0, 1e-6)
    up = np.array([0.0, 1.0, 0.0])
    origin = np.array([float(np.median(v[:, 0])), 0.0, float(np.median(v[:, 2]))])

    if frame is not None:
        facing = np.asarray(frame["facing"], dtype=np.float64)
        right = np.asarray(frame["right"], dtype=np.float64)
        side = (v - origin) @ right
        depth = (v - origin) @ facing
    else:
        facing, right, side, depth = _body_axes(v, y, y0, height, origin, up)

    def band(y_a, y_b, side_max=None, side_sign=0):
        m = (y >= y0 + y_a * height) & (y <= y0 + y_b * height)
        if side_max is not None:
            m &= np.abs(side) <= side_max * height
        if side_sign != 0:
            m &= side * side_sign > 0.0
        return v[m]

    core = band(0.58, 0.76, 0.12)
    chest_depth = float(np.median(depth[(y > y0 + 0.60 * height) & (y < y0 + 0.74 * height) & (np.abs(side) < 0.10 * height)])) if np.any(np.abs(side) < 0.10 * height) else 0.0
    torso_radius = float(np.percentile(np.abs(side[(y > y0 + 0.62 * height) & (y < y0 + 0.74 * height)]), 62)) if np.any((y > y0 + 0.62 * height) & (y < y0 + 0.74 * height)) else 0.08 * height

    def at_height(frac, side_value, depth_value):
        return origin + up * (y0 + frac * height) + right * side_value + facing * depth_value

    neck_band = band(0.78, 0.90, 0.16)
    neck_y = y0 + 0.835 * height
    if neck_band.shape[0] > 30:
        rel = (neck_band[:, 1] - y0) / height
        radii = []
        best = None
        for frac in np.linspace(0.79, 0.90, 12):
            sl = neck_band[np.abs(rel - frac) < 0.012]
            if sl.shape[0] < 15:
                continue
            rad = float(np.percentile(np.linalg.norm((sl - origin)[:, [0, 2]], axis=1), 80))
            radii.append((rad, frac, sl))
        if radii:
            rad, frac, sl = min(radii, key=lambda t: t[0])
            neck_y = y0 + frac * height
            neck = _centroid(sl, at_height(frac, 0.0, chest_depth * 0.2))
        else:
            neck = at_height(0.835, 0.0, chest_depth * 0.15)
    else:
        neck = at_height(0.835, 0.0, chest_depth * 0.15)

    head_pts = v[y > neck_y]
    head_c = _centroid(head_pts, at_height(0.92, 0.0, chest_depth * 0.1))
    head_top = head_c.copy()
    head_top[1] = y1 - 0.01 * height
    head_joint = head_c.copy()
    head_joint[1] = neck_y + 0.45 * (y1 - neck_y)

    hip_y = y0 + 0.50 * height
    pelvis = at_height(0.50, 0.0, chest_depth * 0.25)
    hip_pts = band(0.46, 0.54, 0.14)
    if hip_pts.shape[0] > 20:
        pelvis = _centroid(hip_pts, pelvis)
        pelvis[1] = hip_y
    chest = at_height(0.70, 0.0, chest_depth * 0.35)
    chest_pts = band(0.66, 0.74, 0.12)
    if chest_pts.shape[0] > 20:
        chest = _centroid(chest_pts, chest)
    spine = (pelvis + chest) * 0.5

    def limb(sign, kind):
        if kind == "leg":
            mask = (y < y0 + 0.52 * height) & (side * sign > torso_radius * 0.15)
            foot = v[mask & (y < y0 + 0.14 * height)]
            ankle = _centroid(foot, at_height(0.07, sign * torso_radius * 0.85, 0.0))
            ankle[1] = float(np.percentile(foot[:, 1], 35)) if foot.shape[0] > 10 else y0 + 0.07 * height
            toe_pool = foot if foot.shape[0] else v[mask]
            if toe_pool.shape[0]:
                toe = toe_pool[np.argmax((toe_pool - origin) @ facing)].copy()
            else:
                toe = ankle + facing * (0.06 * height)
            hip = at_height(0.50, sign * torso_radius * 0.7, chest_depth * 0.2)
            near = v[mask & (np.abs(y - hip_y) < 0.04 * height)]
            hip = _centroid(near, hip)
            hip[1] = hip_y
            knee = (hip + ankle) * 0.5
            knee_pts = v[mask & (np.abs(y - knee[1]) < 0.035 * height)]
            knee = _centroid(knee_pts, knee)
            return hip, knee, ankle, toe
        reach = (y > y0 + 0.38 * height) & (y < y0 + 0.82 * height) & (side * sign > torso_radius * 1.05)
        arm_pts = v[reach]
        shoulder = at_height(0.78, sign * torso_radius * 0.95, chest_depth * 0.25)
        if arm_pts.shape[0] < 30:
            hand = at_height(0.55, sign * torso_radius * 2.4, chest_depth * 0.1)
            elbow = (shoulder + hand) * 0.5
            return shoulder, elbow, hand, hand + _unit(hand - shoulder) * (0.04 * height)
        order = side[reach] * sign
        limit = np.percentile(order, 88)
        palm = arm_pts[order >= limit * 0.92]
        hand = _centroid(palm, arm_pts[np.argmax(order)])
        elbow_guess = shoulder * 0.42 + hand * 0.58
        elbow_pts = arm_pts[np.linalg.norm(arm_pts - elbow_guess, axis=1) < 0.08 * height]
        elbow = _centroid(elbow_pts, elbow_guess)
        # Локоть — точка цепочки с наибольшим изломом.
        samples = []
        for t in np.linspace(0.25, 0.75, 7):
            guess = shoulder * (1.0 - t) + hand * t
            near = arm_pts[np.linalg.norm(arm_pts - guess, axis=1) < 0.06 * height]
            samples.append(_centroid(near, guess))
        line = hand - shoulder
        line_n = _unit(line)
        bend = 0.0
        bent = samples[len(samples) // 2]
        for p in samples:
            dist = float(np.linalg.norm(np.cross(p - shoulder, line_n)))
            if dist > bend:
                bend = dist
                bent = p
        span = float(np.dot(line, line)) + 1e-12
        along = float(np.dot(bent - shoulder, line)) / span
        if bend > 0.02 * height and 0.32 <= along <= 0.5:
            elbow = bent
        else:
            elbow = shoulder * 0.64 + hand * 0.36
        wrist = elbow * 0.22 + hand * 0.78
        return shoulder, elbow, wrist, hand

    bones = []

    def add(name, parent, head, tail):
        bones.append({
            "name": name,
            "parent": parent,
            "head": np.asarray(head, dtype=np.float64),
            "tail": np.asarray(tail, dtype=np.float64),
        })
        return len(bones) - 1

    pelvis_i = add("Pelvis", -1, pelvis, spine)
    spine_i = add("Spine", pelvis_i, spine, chest)
    chest_i = add("Chest", spine_i, chest, neck)
    neck_i = add("Neck", chest_i, neck, head_joint)
    add("Head", neck_i, head_joint, head_top)

    for sign, tag in ((-1.0, "L"), (1.0, "R")):
        shoulder, elbow, wrist, hand = limb(sign, "arm")
        clav = chest + (shoulder - chest) * 0.35
        clav_i = add(f"Clavicle_{tag}", chest_i, clav, shoulder)
        arm_i = add(f"UpperArm_{tag}", clav_i, shoulder, elbow)
        fore_i = add(f"ForeArm_{tag}", arm_i, elbow, wrist)
        add(f"Hand_{tag}", fore_i, wrist, hand)
        hip, knee, ankle, toe = limb(sign, "leg")
        thigh_i = add(f"Thigh_{tag}", pelvis_i, hip, knee)
        shin_i = add(f"Shin_{tag}", thigh_i, knee, ankle)
        foot_i = add(f"Foot_{tag}", shin_i, ankle, toe)
        toe_tail = toe + facing * (0.035 * height)
        add(f"Toe_{tag}", foot_i, toe, toe_tail)

    meta = {"facing": facing, "right": right, "up": up, "origin": origin, "height": height}
    return bones, meta


def bone_segments(bones):
    return np.stack([np.stack([b["head"], b["tail"]], axis=0) for b in bones], axis=0)


def heat_weights(verts, faces, bones, influences=4):
    """Тепловые веса: Лапласиан сетки разносит привязку ближайшей кости."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    v = verts.astype(np.float64)
    n = v.shape[0]
    extent = float((v.max(0) - v.min(0)).max())
    q = np.round(v / max(extent * 1e-5, 1e-9)).astype(np.int64)
    _, weld, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    wn = int(weld.max()) + 1
    acc = np.zeros((wn, 3), dtype=np.float64)
    np.add.at(acc, weld, v)
    centers = acc / np.maximum(np.bincount(weld), 1)[:, None]

    wf = weld[faces]
    edges = np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = coo_matrix((np.ones(rows.shape[0], dtype=np.float64), (rows, cols)), shape=(wn, wn)).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    lap = diags(deg + 1e-4) - adj

    segs = bone_segments(bones)
    dist = np.empty((wn, len(bones)), dtype=np.float64)
    for i, (a, b) in enumerate(segs):
        ab = b - a
        denom = float(ab @ ab) + 1e-12
        t = np.clip(((centers - a) @ ab) / denom, 0.0, 1.0)
        dist[:, i] = np.linalg.norm(centers - (a + t[:, None] * ab), axis=1)
    nearest = np.argmin(dist, axis=1)
    nearest_d = dist[np.arange(wn), nearest]

    solved = np.zeros((wn, len(bones)), dtype=np.float64)
    for b in range(len(bones)):
        h = np.zeros(wn, dtype=np.float64)
        attach = nearest == b
        if int(attach.sum()) < 4:
            continue
        h[attach] = 1.0 / (nearest_d[attach] ** 2 + (extent * 1e-4) ** 2)
        system = lap + diags(h)
        try:
            w, info = cg(system, h, rtol=1e-3, maxiter=250)
        except TypeError:
            w, info = cg(system, h, tol=1e-3, maxiter=250)
        if not np.isfinite(w).all():
            continue
        solved[:, b] = np.clip(w, 0.0, None)
        _ = info

    empty = solved.sum(axis=1) <= 1e-8
    solved[empty, nearest[empty]] = 1.0
    if edges.shape[0]:
        smooth = adj.dot(solved) / np.maximum(deg, 1e-8)[:, None]
        solved = 0.7 * solved + 0.3 * smooth
    solved = np.maximum(solved, 0.0)

    k = min(influences, solved.shape[1])
    pick = np.argpartition(solved, -k, axis=1)[:, -k:]
    ww = np.take_along_axis(solved, pick, axis=1)
    ww /= np.maximum(ww.sum(axis=1, keepdims=True), 1e-8)
    full_i = np.zeros((wn, influences), dtype=np.uint16)
    full_w = np.zeros((wn, influences), dtype=np.float32)
    full_i[:, :k] = pick.astype(np.uint16)
    full_w[:, :k] = ww.astype(np.float32)
    return full_i[weld], full_w[weld]


def _pad4(blob):
    return blob + b"\x00" * ((4 - len(blob) % 4) % 4)


def write_skinned_glb(path, verts, faces, uvs, normals, bones, joints, weights):
    """glTF со скелетом. В покое сетка не смещается: IBM — обратный перенос головы кости."""
    arrays = []

    def push(arr):
        raw = _pad4(np.ascontiguousarray(arr).tobytes())
        arrays.append(raw)
        return len(arrays) - 1, len(raw)

    v = np.ascontiguousarray(verts, dtype=np.float32)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    uv = np.ascontiguousarray(uvs, dtype=np.float32)
    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    j = np.ascontiguousarray(joints, dtype=np.uint16)
    w = np.ascontiguousarray(weights, dtype=np.float32)

    blobs = {
        "pos": push(v),
        "nrm": push(nrm),
        "uv": push(uv),
        "idx": push(f),
        "joints": push(j),
        "weights": push(w),
    }
    heads = np.stack([b["head"] for b in bones]).astype(np.float32)
    ibms = np.zeros((len(bones), 4, 4), dtype=np.float32)
    for i, h in enumerate(heads):
        ibms[i] = np.eye(4, dtype=np.float32)
        ibms[i][:3, 3] = -h
    # glTF stores matrices column-major.
    ibms = np.transpose(ibms, (0, 2, 1)).copy()
    blobs["ibm"] = push(ibms)

    bin_blob = b"".join(arrays[i] for i in range(len(arrays)))
    view_off = []
    cursor = 0
    for raw in arrays:
        view_off.append(cursor)
        cursor += len(raw)

    def view(key, target):
        idx, length = blobs[key]
        return {"buffer": 0, "byteOffset": view_off[idx], "byteLength": length, "target": target}

    # buffer views inserted in blob order
    buffer_views = [view(name, 34962) for name in ("pos", "nrm", "uv", "idx", "joints", "weights", "ibm")]
    buffer_views[3]["target"] = 34963

    def acc(view_i, comp, ctype, count, type_, mins=None, maxs=None):
        item = {
            "bufferView": view_i,
            "componentType": comp,
            "count": count,
            "type": type_,
        }
        if ctype is not None:
            pass
        if mins is not None:
            item["min"] = mins
            item["max"] = maxs
        return item

    accessors = [
        acc(0, 5126, None, len(v), "VEC3", v.min(0).tolist(), v.max(0).tolist()),
        acc(1, 5126, None, len(nrm), "VEC3"),
        acc(2, 5126, None, len(uv), "VEC2"),
        acc(3, 5125, None, int(f.shape[0] * 3), "SCALAR"),
        acc(4, 5123, None, len(j), "VEC4"),
        acc(5, 5126, None, len(w), "VEC4"),
        acc(6, 5126, None, len(bones), "MAT4"),
    ]
    # Index accessor count is the number of indices, not faces. Fix:
    accessors[3]["count"] = int(f.size)

    nodes = []
    # 0 is the mesh. Joints follow, then leaf tips.
    mesh_node = {"name": "Mesh", "mesh": 0, "skin": 0}
    nodes.append(mesh_node)
    joint_ids = []
    for b in bones:
        node = {"name": b["name"], "translation": [float(x) for x in b["head"]]}
        if b["parent"] >= 0:
            node["translation"] = [float(x) for x in (b["head"] - bones[b["parent"]]["head"])]
        nodes.append(node)
        joint_ids.append(len(nodes) - 1)
    for i, b in enumerate(bones):
        children = [joint_ids[k] for k, other in enumerate(bones) if other["parent"] == i]
        if not children:
            tip = {"name": b["name"] + "_end", "translation": [float(x) for x in (b["tail"] - b["head"])]}
            nodes.append(tip)
            children = [len(nodes) - 1]
        nodes[joint_ids[i]]["children"] = children
    for i, b in enumerate(bones):
        if b["parent"] < 0:
            continue
        # child link is stored on the parent; nothing else
        _ = i

    roots = [0] + [joint_ids[i] for i, b in enumerate(bones) if b["parent"] < 0]
    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyLLM garage kit rig"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "skins": [{"name": "Armature", "joints": joint_ids, "inverseBindMatrices": 6, "skeleton": joint_ids[0]}],
        "meshes": [{
            "name": "Body",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2,
                    "JOINTS_0": 4,
                    "WEIGHTS_0": 5,
                },
                "indices": 3,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [{
            "name": "Clay",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.78, 0.78, 0.78, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65,
            },
        }],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    import json
    payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(payload) + 8 + len(bin_blob))
    out += struct.pack("<I", len(payload)) + b"JSON" + payload
    out += struct.pack("<I", len(bin_blob)) + b"BIN\x00" + bin_blob
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(out)


def write_quad_obj(path, verts, uvs, tris, quads):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# quads + UV, Y-up, editable in Blender\n")
        for p in verts:
            fh.write(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n")
        for u, t in uvs:
            fh.write(f"vt {u:.6g} {t:.6g}\n")
        for face in quads:
            ids = " ".join(f"{i + 1}/{i + 1}" for i in face)
            fh.write(f"f {ids}\n")
        used = set()
        if len(quads):
            for face in quads:
                a, b, c, d = (int(x) for x in face)
                used.add((a, b, c))
                used.add((a, c, d))
        for a, b, c in tris:
            key = (int(a), int(b), int(c))
            if key in used:
                continue
            fh.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")


def _camera(verts, meta, device):
    from comfy_extras.nodes_mesh_postprocess import _mesh_view_camera

    lo, hi = verts.min(0), verts.max(0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    dist = radius / math.tan(math.radians(18.0)) * 1.08
    facing = meta["facing"]
    right = meta["right"]
    eye = center + facing * dist * 0.82 + np.array([0.0, 1.0, 0.0]) * dist * 0.16 + right * dist * 0.38
    cam = {
        "position": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "target": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "fov": 34.0,
        "zoom": 1.0,
        "cameraType": "perspective",
        "up": {"y": 1.0},
    }
    eye_t, forward, right_t, up_t, fov, ortho = _mesh_view_camera(cam, device)
    return eye_t, forward, right_t, up_t, fov, ortho, center


def _render(verts, faces, uvs, texture, eye, forward, right, up, fov, size, device):
    from comfy_extras.nodes_mesh_postprocess import _build_triangle_bvh, _render_view

    lv = torch.from_numpy(np.ascontiguousarray(verts, dtype=np.float32)).to(device)
    lf = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).to(device)
    tri = lv[lf]
    bvh = _build_triangle_bvh(tri)
    uv_t = None
    tex_t = None
    if texture is not None and uvs is not None:
        uv_t = torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).to(device)
        tex_t = torch.from_numpy(np.ascontiguousarray(texture, dtype=np.float32)).to(device)
    img, hit, depth = _render_view(tri, bvh, uv_t, lf, tex_t, eye, forward, right, up, fov, size, size)
    return img.detach().float().cpu().numpy(), hit.detach().cpu().numpy(), depth.detach().float().cpu().numpy()


def _checker(cells=32, size=512):
    img = np.empty((size, size, 3), dtype=np.float32)
    g = np.indices((size, size))
    cell = size // cells
    a = ((g[0] // cell) + (g[1] // cell)) % 2 == 0
    major = ((g[0] // cell) % 4 == 0) | ((g[1] // cell) % 4 == 0)
    img[:] = np.array([0.22, 0.22, 0.24], dtype=np.float32)
    img[a] = np.array([0.86, 0.86, 0.84], dtype=np.float32)
    edge = (g[0] % cell < 2) | (g[1] % cell < 2)
    img[edge] = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    img[major & edge] = np.array([0.75, 0.28, 0.18], dtype=np.float32)
    return img


def _project(points, eye, forward, right, up, fov, size):
    eye_n = eye.detach().float().cpu().numpy()
    f = forward.detach().float().cpu().numpy()
    r = right.detach().float().cpu().numpy()
    u = up.detach().float().cpu().numpy()
    rel = points - eye_n
    z = rel @ f
    tn = math.tan(0.5 * float(fov))
    gx = (rel @ r) / (z * tn + 1e-8)
    gy = (rel @ u) / (z * tn + 1e-8)
    px = (gx + 1.0) * 0.5 * (size - 1)
    py = (1.0 - gy) * 0.5 * (size - 1)
    return px, py, z, np.linalg.norm(rel, axis=1)


def _edges_of(tris, quads):
    chunks = [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]
    if quads is not None and len(quads):
        chunks = [quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]
    e = np.concatenate(chunks, axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _draw_edges(image, depth, verts, edges, eye, forward, right, up, fov, size):
    if edges.shape[0] == 0:
        return image
    px, py, z, dist = _project(verts, eye, forward, right, up, fov, size)
    a, b = edges[:, 0], edges[:, 1]
    ax, ay, bx, by = px[a], py[a], px[b], py[b]
    length = np.hypot(ax - bx, ay - by)
    keep = (z[a] > 0.0) & (z[b] > 0.0) & (length > 1.1) & (length < size * 0.45)
    if keep.sum() > 180000:
        idx = np.flatnonzero(keep)
        idx = idx[np.argpartition(length[idx], -180000)[-180000:]]
        keep = np.zeros_like(keep)
        keep[idx] = True
    mx = (ax + bx) * 0.5
    my = (ay + by) * 0.5
    md = (dist[a] + dist[b]) * 0.5
    xi = np.clip(mx.astype(np.int32), 0, size - 1)
    yi = np.clip(my.astype(np.int32), 0, size - 1)
    bias = float(np.linalg.norm(verts.max(0) - verts.min(0))) * 0.004
    visible = keep & np.isfinite(depth[yi, xi]) & (md <= depth[yi, xi] + bias)
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for x0, y0, x1, y1 in zip(ax[visible], ay[visible], bx[visible], by[visible]):
        pen.line((float(x0), float(y0), float(x1), float(y1)), fill=(18, 20, 24), width=1)
    return np.asarray(canvas)


def _draw_bones(image, depth, bones, eye, forward, right, up, fov, size):
    heads = np.stack([b["head"] for b in bones])
    tails = np.stack([b["tail"] for b in bones])
    pts = np.concatenate([heads, tails], axis=0)
    px, py, z, dist = _project(pts, eye, forward, right, up, fov, size)
    n = len(bones)
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    bias = 0.0
    span = 0.0
    for i, bone in enumerate(bones):
        span = max(span, float(np.linalg.norm(bone["tail"] - bone["head"])))
    bias = span * 0.35
    for i, bone in enumerate(bones):
        if z[i] <= 0 or z[n + i] <= 0:
            continue
        mx = int(np.clip((px[i] + px[n + i]) * 0.5, 0, size - 1))
        my = int(np.clip((py[i] + py[n + i]) * 0.5, 0, size - 1))
        md = (dist[i] + dist[n + i]) * 0.5
        if np.isfinite(depth[my, mx]) and md > depth[my, mx] + bias:
            color = (120, 72, 40)
        else:
            color = (235, 122, 36)
        pen.line((float(px[i]), float(py[i]), float(px[n + i]), float(py[n + i])), fill=color, width=3)
        r = 4
        pen.ellipse((px[i] - r, py[i] - r, px[i] + r, py[i] + r), fill=(255, 214, 120))
    return np.asarray(canvas)


def _caption(image, text):
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except OSError:
        font = None
    box = pen.textbbox((20, 16), text, font=font)
    pen.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
    pen.text((20, 16), text, fill=(240, 240, 240), font=font)
    return np.asarray(canvas)


def _to_u8(img, hit, bg=(18, 18, 20)):
    out = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    rgb = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    mask = hit.astype(bool)
    out[mask] = rgb[mask]
    out[~mask] = np.array(bg, dtype=np.uint8)
    return out


def build_previews(verts, faces, uvs, quads, bones, meta, size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eye, forward, right, up, fov, _ortho, _center = _camera(verts, meta, device)
    clay, hit, depth = _render(verts, faces, None, None, eye, forward, right, up, fov, size, device)
    clay_u8 = _to_u8(clay, hit)
    if uvs is None:
        uv_u8 = clay_u8.copy()
        uv_u8 = _caption(uv_u8, "UV: на треугольниках нет развёртки")
    else:
        checker, _, _ = _render(verts, faces, uvs, _checker(), eye, forward, right, up, fov, size, device)
        luma = clay.mean(axis=2, keepdims=True)
        shaded = checker * (0.42 + 0.75 * luma)
        uv_u8 = _caption(_to_u8(shaded, hit), "UV сетка")
    rig_u8 = _caption(_draw_bones(clay_u8, depth, bones, eye, forward, right, up, fov, size), "Риг")
    head_eye, head_forward, head_right, head_up, head_fov = _head_camera(verts, meta, device)
    region_uv, palette = _region_map(verts, meta)
    region, region_hit, region_depth = _render(
        verts, faces, region_uv, palette, head_eye, head_forward, head_right, head_up, head_fov, size, device,
    )
    region_shade = 0.42 + 0.8 * region.mean(axis=2, keepdims=True)
    colored = _to_u8(region * region_shade, region_hit, bg=(232, 224, 206))
    edges = _edges_of(faces, quads)
    label = f"Квады {len(quads)}" if quads is not None and len(quads) else "Треугольники"
    wire_u8 = _caption(
        _draw_edges(colored, region_depth, verts, edges, head_eye, head_forward, head_right, head_up, head_fov, size),
        label,
    )

    def pack(arr):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32) / 255.0).unsqueeze(0)

    return pack(uv_u8), pack(rig_u8), pack(wire_u8)


def _head_camera(verts, meta, device):
    from comfy_extras.nodes_mesh_postprocess import _mesh_view_camera

    v = verts.astype(np.float64)
    y = v[:, 1]
    y0 = float(y.min())
    height = max(float(y.max()) - y0, 1e-6)
    upper = v[y > y0 + 0.62 * height]
    center = upper.mean(0) if len(upper) else v.mean(0)
    facing = np.asarray(meta["facing"], dtype=np.float64)
    radius = float(np.linalg.norm(upper.max(0) - upper.min(0))) * 0.55 if len(upper) else 0.2
    dist = radius / math.tan(math.radians(16.0)) * 1.15
    eye = center + facing * dist
    cam = {
        "position": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "target": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "fov": 32.0,
        "zoom": 1.0,
        "cameraType": "perspective",
        "up": {"y": 1.0},
    }
    eye_t, forward, right_t, up_t, fov, _ortho = _mesh_view_camera(cam, device)
    return eye_t, forward, right_t, up_t, fov


def _region_map(verts, meta):
    """Цвета как на анимационной сетке: голова, торс, руки, ноги."""
    v = verts.astype(np.float64)
    y = v[:, 1]
    y0 = float(y.min())
    height = max(float(y.max()) - y0, 1e-6)
    side = (v - np.asarray(meta["origin"], dtype=np.float64)) @ np.asarray(meta["right"], dtype=np.float64)
    arm = (y > y0 + 0.48 * height) & (y < y0 + 0.82 * height) & (np.abs(side) > 0.13 * height)
    leg = y < y0 + 0.50 * height
    head = y > y0 + 0.80 * height
    ids = np.ones(len(v), dtype=np.int64)
    ids[leg] = 3
    ids[arm] = 2
    ids[head] = 0
    palette = np.array([
        [0.93, 0.84, 0.28],
        [0.42, 0.62, 0.90],
        [0.42, 0.78, 0.45],
        [0.90, 0.52, 0.66],
    ], dtype=np.float32)
    uv = np.stack([(ids.astype(np.float32) + 0.5) / 4.0, np.full(len(v), 0.5, dtype=np.float32)], axis=1)
    tex = np.zeros((1, 4, 3), dtype=np.float32)
    tex[0] = palette
    return uv, tex


def export_paths(prefix):
    prefix = prefix.replace("\\", "/").strip().lstrip("/")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("Имя файла не должно быть пустым и не должно содержать ..")
    root = folder_paths.get_output_directory()
    obj = os.path.join(root, prefix + "_quad.obj")
    glb = os.path.join(root, prefix + "_rig.glb")
    stl = os.path.join(root, prefix + "_print.stl")
    return obj, glb, stl


def write_binary_stl(path, verts, faces):
    """Монолитный бинарный STL печатной сетки, до квад-ремеша."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    tri = v[f]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(length, 1e-12)
    records = np.zeros(len(f), dtype=np.dtype([
        ("normal", np.float32, (3,)),
        ("v0", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("attr", np.uint16),
    ]))
    records["normal"] = normal
    records["v0"] = tri[:, 0]
    records["v1"] = tri[:, 1]
    records["v2"] = tri[:, 2]
    with open(path, "wb") as fh:
        fh.write(b"ComfyLLM print mesh binary STL".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(records.tobytes())


def saved_3d(path):
    root = os.path.normpath(folder_paths.get_output_directory())
    full = os.path.normpath(path)
    rel = os.path.relpath(full, root).replace("\\", "/")
    subfolder, _, filename = rel.rpartition("/")
    return {"filename": filename, "subfolder": subfolder, "type": "output"}


def prepare_topology(mesh, topology, target_quads, preview_size, file_prefix, uv_margin=0.02):
    verts, faces, uvs = mesh_arrays(mesh)
    topology = (topology or "quads").lower()
    size = int(np.clip(preview_size, 256, 1280))
    if topology.startswith("quad"):
        qv, quv, qf, quads = run_blender_quads(verts, faces, int(target_quads), float(uv_margin))
        # Ось лица считается по исходной сетке: после склейки треугольников бант может перевесить чёлку.
        _, frame = fit_humanoid(cluster_decimate(verts, faces, 40000)[0])
        bones, meta = fit_humanoid(qv, frame)
        joints, weights = heat_weights(qv, qf, bones)
        from comfy_extras.nodes_mesh_postprocess import _smooth_vertex_normals
        normals = _smooth_vertex_normals(qv, qf, weld=True)
        obj_path, glb_path, stl_path = export_paths(file_prefix)
        write_quad_obj(obj_path, qv, quv, qf, quads)
        write_skinned_glb(glb_path, qv, qf, quv, normals, bones, joints, weights)
        write_binary_stl(stl_path, verts, faces)
        print(f"[topology] OBJ {obj_path}")
        print(f"[topology] GLB {glb_path}")
        print(f"[topology] STL {stl_path}")
        quv_comfy = quv.copy()
        quv_comfy[:, 1] = 1.0 - quv_comfy[:, 1]
        previews = build_previews(qv, qf, quv_comfy, quads, bones, meta, size)
        import copy
        out_mesh = copy.copy(mesh)
        out_mesh.vertices = torch.from_numpy(qv).unsqueeze(0)
        out_mesh.faces = torch.from_numpy(qf.astype(np.int64)).unsqueeze(0)
        out_mesh.uvs = torch.from_numpy(quv_comfy.astype(np.float32)).unsqueeze(0)
        out_mesh.normals = torch.from_numpy(normals).unsqueeze(0)
        out_mesh.vertex_colors = None
        out_mesh.texture = None
        out_mesh.vertex_counts = None
        out_mesh.face_counts = None
        files = [saved_3d(glb_path), saved_3d(obj_path), saved_3d(stl_path)]
        return (out_mesh, *previews), files
    bones, meta = fit_humanoid(verts)
    _obj, _glb, stl_path = export_paths(file_prefix)
    write_binary_stl(stl_path, verts, faces)
    previews = build_previews(verts, faces, uvs, None, bones, meta, size)
    return (mesh, *previews), [saved_3d(stl_path)]


class GarageKitTopologyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Готовая сетка фигурки, та же, что идёт в SaveGLB"}),
                "topology": (["quads", "triangles"], {
                    "default": "quads",
                    "tooltip": "quads — квадраты для анимации, UV и риг. triangles — исходная треугольная сетка печати.",
                }),
                "target_quads": ("INT", {
                    "default": 18000,
                    "min": 4000,
                    "max": 80000,
                    "step": 1000,
                    "tooltip": "Ровная квад-сетка для Blender. 14–22 тысячи, чтобы петли лица были видны.",
                }),
                "preview_size": ("INT", {
                    "default": 768,
                    "min": 384,
                    "max": 1280,
                    "step": 64,
                    "tooltip": "Сторона экранов UV, рига и сетки.",
                }),
                "file_prefix": ("STRING", {
                    "default": "3d/2B_Anim",
                    "tooltip": "Папка output. Рядом появятся кнопки: _rig.glb и _quad.obj для Blender, _print.stl для печати.",
                }),
            }
        }

    RETURN_TYPES = ("MESH", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mesh", "uv_preview", "rig_preview", "wire_preview")
    FUNCTION = "run"
    CATEGORY = "3d/garage_kit"
    OUTPUT_NODE = True

    def run(self, mesh, topology, target_quads, preview_size, file_prefix):
        result, files = prepare_topology(mesh, topology, target_quads, preview_size, file_prefix)
        return {"ui": {"3d": files}, "result": result}


NODE_CLASS_MAPPINGS = {"GarageKitTopology": GarageKitTopologyNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GarageKitTopology": "Топология · квады, UV, риг, STL",
}
