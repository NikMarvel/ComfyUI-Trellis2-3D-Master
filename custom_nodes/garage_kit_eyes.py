"""
Глаза: снять рваную чёрную маску с лица и нарисовать глаза как на референсе.

Яблоки и отдельные веки не строятся. Тонкая сплошная повязка не трогается.
"""

from __future__ import annotations

import numpy as np
import torch


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros(3)


def _as_arrays(mesh):
    verts, faces, uvs = mesh.vertices, mesh.faces, mesh.uvs
    tex = mesh.texture
    if isinstance(verts, (list, tuple)):
        v, f = verts[0], faces[0]
        uv = None if uvs is None else uvs[0]
        image = None if tex is None else tex[0]
    elif torch.is_tensor(verts) and verts.ndim == 3:
        vc = getattr(mesh, "vertex_counts", None)
        fc = getattr(mesh, "face_counts", None)
        nv = int(vc[0]) if vc is not None else verts.shape[1]
        nf = int(fc[0]) if fc is not None else faces.shape[1]
        v, f = verts[0, :nv], faces[0, :nf]
        uv = None if uvs is None else uvs[0, :nv]
        image = None if tex is None else (tex[0] if tex.ndim == 4 else tex)
    else:
        v, f, uv, image = verts, faces, uvs, tex
    v = v.detach().float().cpu().numpy()
    f = f.detach().long().cpu().numpy()
    uv = None if uv is None else uv.detach().float().cpu().numpy()
    if image is not None:
        image = image.detach().float().cpu().numpy()
        if image.max() <= 1.0 + 1e-4:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[-1] == 4:
            image = image[..., :3]
    return v, f, uv, image


def _colors(uv, image):
    if uv is None or image is None:
        return np.full((len(uv) if uv is not None else 1, 3), 180, dtype=np.float64)
    h, w = image.shape[:2]
    tx = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    ty = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    return image[ty, tx].astype(np.float64)


def _body_axes(verts):
    y = verts[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    height = max(y1 - y0, 1e-6)
    origin = np.array([float(np.median(verts[:, 0])), 0.0, float(np.median(verts[:, 2]))])
    up = np.array([0.0, 1.0, 0.0])
    torso = verts[(y > y0 + 0.45 * height) & (y < y0 + 0.75 * height)]
    if len(torso) < 50:
        torso = verts
    xz = torso[:, [0, 2]] - origin[[0, 2]]
    _, evec = np.linalg.eigh(xz.T @ xz)
    axis = _unit(np.array([evec[0, 0], 0.0, evec[1, 0]]))
    head = verts[y > y0 + 0.82 * height]
    proj = (head - origin) @ axis if len(head) else np.array([1.0])
    pos, neg = proj[proj > 0], -proj[proj < 0]
    pos_s = float(np.percentile(pos, 90)) if pos.size > 30 else 0.0
    neg_s = float(np.percentile(neg, 90)) if neg.size > 30 else 0.0
    outward = _unit(axis if pos_s >= neg_s else -axis)
    right = _unit(np.cross(up, outward))
    return outward, right, up, y0, height


def find_eyes(verts, colors):
    """Два тёмных миндаля в верхней части лица. Повязка, волосы и чёрная одежда не подходят."""
    lum = colors.mean(1)
    outward, right, up, y0, height = _body_axes(verts)
    origin = np.median(verts, 0)
    depth = (verts - origin) @ outward
    side = (verts - origin) @ right
    y = verts[:, 1]
    # Только верх головы. Ниже лежат ворот и чёрная одежда.
    band = (y > y0 + 0.875 * height) & (y < y0 + 0.97 * height)
    band &= depth > np.percentile(depth, 62)
    dark = band & (lum < 55)
    if int(dark.sum()) < 200:
        return []
    eyes = []
    for sign in (-1.0, 1.0):
        m = dark & (side * sign > height * 0.01)
        if int(m.sum()) < 80:
            return []
        pts = verts[m]
        front = depth[m] > np.percentile(depth[m], 55)
        use = pts[front] if int(front.sum()) > 30 else pts
        raw = np.median(use, 0)
        rel = use - raw
        wide = max(float(np.percentile(np.abs(rel @ right), 70)), height * 0.008)
        tall = float(np.percentile(np.abs(rel @ up), 70))
        tall = min(max(tall, wide * 0.28), wide * 0.75)
        deep = max(float(np.percentile(np.abs(rel @ outward), 70)), tall)
        # Вокруг глаза кожа. Белая масса — волосы вокруг повязки, чёрная — одежда.
        ring = (np.abs((verts - raw) @ right) < wide * 2.2) & (np.abs((verts - raw) @ up) < tall * 2.4)
        ring &= np.abs((verts - raw) @ outward) < deep * 2.0
        ring &= ~m
        if int(ring.sum()) < 40:
            return []
        skin = (lum > 120) & (lum < 235) & (colors[:, 0] + 6 > colors[:, 2])
        if float(skin[ring].mean()) < 0.35:
            return []
        center = raw
        # Внутри миндаля должна просвечивать радужка, а не сплошная повязка.
        inner = (np.abs((verts - raw) @ right) < wide * 0.7) & (np.abs((verts - raw) @ up) < tall * 0.7)
        midtone = (lum > 60) & (lum < 170)
        if float(midtone[inner].mean()) < 0.04:
            return []
        eyes.append({
            "center": center,
            "wide": wide,
            "tall": tall,
            "deep": deep,
            "sign": sign,
            "count": int(m.sum()),
        })
    if abs(eyes[0]["center"][1] - eyes[1]["center"][1]) > height * 0.04:
        return []
    return eyes


def _boundary_loops(faces):
    nxt = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            rev = (v, u)
            if rev in nxt:
                nxt.pop(rev, None)
                nxt.pop((u, v), None)
            else:
                nxt[(u, v)] = v
    out_of = {}
    for (u, v) in nxt:
        out_of.setdefault(u, []).append(v)
    loops = []
    seen = set()
    for start in list(out_of):
        if start in seen:
            continue
        loop = []
        cur = start
        guard = 0
        while cur not in seen and guard < len(out_of) + 2:
            seen.add(cur)
            loop.append(cur)
            nxts = out_of.get(cur)
            if not nxts:
                break
            cur = nxts.pop()
            guard += 1
            if cur == start:
                break
        if len(loop) >= 16 and cur == start:
            loops.append(loop)
    return loops


def _sphere(center, radius, outward, right, up, n_u=40, n_v=24):
    verts, faces = [], []
    for i in range(n_v + 1):
        phi = np.pi * i / n_v
        for j in range(n_u):
            theta = 2 * np.pi * j / n_u
            normal = (
                outward * np.sin(phi) * np.cos(theta)
                + right * np.sin(phi) * np.sin(theta)
                + up * np.cos(phi)
            )
            verts.append(center + radius * normal)
    verts = np.asarray(verts, dtype=np.float32)

    def idx(i, j):
        return i * n_u + (j % n_u)

    for i in range(n_v):
        for j in range(n_u):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces.append((a, c, b))
            faces.append((b, c, d))
    return verts, np.asarray(faces, dtype=np.int64)


def _paint_eye_tile(image, u0, v0, tile, skin, iris):
    """u0, v0 are pixel origins. V in the image grows downward."""
    yy, xx = np.mgrid[0:tile, 0:tile]
    cx = cy = (tile - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy) / (tile * 0.48)
    img = np.empty((tile, tile, 3), dtype=np.uint8)
    img[:] = np.array([232, 230, 226], dtype=np.uint8)
    iris_c = np.clip(iris, 0, 255)
    ring = (r < 0.42) & (r >= 0.16)
    blend = np.clip((0.42 - r) / 0.26, 0, 1)[..., None]
    img = np.where(ring[..., None], img * (1 - blend) + iris_c * blend, img)
    img[r < 0.16] = (18, 18, 20)
    # Кожа по краю тайла — UV век попадает сюда.
    border = (xx < 6) | (yy < 6)
    img[border] = np.clip(skin, 0, 255).astype(np.uint8)
    image[v0:v0 + tile, u0:u0 + tile] = img.astype(np.uint8)
    return image


def _empty_tile(uv, size, tile):
    """Пиксельный угол, куда не попадает ни один существующий UV. Иначе нос белеет."""
    grid = max(1, size // tile)
    hist = np.zeros((grid, grid), dtype=np.int32)
    ui = np.clip((np.mod(uv[:, 0], 1.0) * grid).astype(np.int32), 0, grid - 1)
    vi = np.clip((np.mod(uv[:, 1], 1.0) * grid).astype(np.int32), 0, grid - 1)
    np.add.at(hist, (vi, ui), 1)
    empty = np.argwhere(hist == 0)
    if len(empty) == 0:
        gy, gx = divmod(int(np.argmin(hist)), grid)
    else:
        gy, gx = (int(empty[0, 0]), int(empty[0, 1]))
    return int(gx * tile), int(gy * tile)


def _lid_strip(outer_ids, inner_pts, verts):
    """Треугольники от края кожи к кромке века. Кромка — новые вершины."""
    base = len(verts)
    verts = np.vstack([verts, np.asarray(inner_pts, dtype=np.float32)])
    faces = []
    n = len(outer_ids)
    for i in range(n - 1):
        a, b = outer_ids[i], outer_ids[i + 1]
        c, d = base + i, base + i + 1
        faces.append((a, b, d))
        faces.append((a, d, c))
    return verts, faces


def _paint_reference_eye(image, u0, v0, tile, skin):
    """Чистый глаз как на референсе: кожа, светлая радужка, зрачок, без ресниц."""
    yy, xx = np.mgrid[0:tile, 0:tile]
    cx = cy = (tile - 1) / 2.0
    rx = np.abs(xx - cx) / (tile * 0.22)
    ry = np.abs(yy - cy) / (tile * 0.16)
    r = np.hypot(rx, ry)
    img = np.empty((tile, tile, 3), dtype=np.float32)
    img[:] = skin
    iris = np.array([158.0, 174.0, 186.0])
    img[r < 1.0] = iris
    img[(r < 1.0) & (r > 0.72)] = iris * 0.78
    img[r < 0.28] = (28.0, 30.0, 34.0)
    image[v0:v0 + tile, u0:u0 + tile] = np.clip(img, 0, 255).astype(np.uint8)


def _texel_index(uv, height, width):
    tx = np.clip((uv[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    ty = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)
    return ty, tx


def _paint_eye_disk(image, cx, cy, rx, ry):
    h, w = image.shape[:2]
    x0 = max(int(cx - rx - 2), 0)
    x1 = min(int(cx + rx + 3), w)
    y0 = max(int(cy - ry - 2), 0)
    y1 = min(int(cy + ry + 3), h)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot((xx - cx) / max(rx, 1.0), (yy - cy) / max(ry, 1.0))
    iris = np.array([158.0, 174.0, 186.0])
    patch = image[y0:y1, x0:x1].astype(np.float32)
    patch[r < 1.0] = iris
    patch[(r < 1.0) & (r > 0.72)] = iris * 0.78
    patch[r < 0.28] = (28.0, 30.0, 34.0)
    image[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def repaint_face(verts, faces, uvs, image, colors):
    """Чёрная маска на всё лицо заменяется одним куском кожи с двумя глазами."""
    if image is None or uvs is None or len(colors) != len(verts):
        return 0
    outward, right, up, y0, height = _body_axes(verts)
    origin = np.median(verts, 0)
    depth = (verts - origin) @ outward
    side = (verts - origin) @ right
    y = verts[:, 1]
    lum = colors.mean(1)
    front = depth > np.percentile(depth, 60)
    head = (y > y0 + 0.80 * height) & front
    dark = head & (lum < 58)
    if int(dark.sum()) < 1500:
        return 0
    y_lo = float(np.percentile(y[dark], 6))
    y_hi = float(np.percentile(y[dark], 99))
    y_span = (y_hi - y_lo) / height
    x_half = float(np.percentile(np.abs(side[dark]), 96))
    x_span = (2.0 * x_half) / height
    if y_span < 0.055 or x_span < 0.035:
        print(f"[eyes] тонкая повязка, не трогаю span y {y_span:.3f} x {x_span:.3f}")
        return 0
    light = head & (lum > 145) & (lum < 235) & (colors[:, 0] + 8 > colors[:, 2])
    skin = np.median(colors[light], 0) if int(light.sum()) > 40 else np.array([214.0, 186.0, 176.0])
    centroids = verts[faces].mean(1)
    c_side = (centroids - origin) @ right
    c_y = centroids[:, 1]
    c_depth = (centroids - origin) @ outward
    # Все три вершины грани тёмные. Иначе губа и нос уезжают на плоский тайл.
    chosen = (lum[faces] < 62).all(axis=1)
    chosen &= (c_y > y_lo + 0.18 * (y_hi - y_lo)) & (c_y < y_hi) & (np.abs(c_side) < x_half * 1.05)
    chosen &= c_depth > np.percentile(depth, 68)
    if int(chosen.sum()) < 80:
        return 0
    eye_verts = np.zeros(len(verts), dtype=bool)
    eye_verts[np.unique(faces[chosen])] = True
    h, w = image.shape[:2]
    tile = 512 if min(h, w) >= 512 else 256
    u0, v0 = _empty_tile(uvs, w, tile)
    image[v0:v0 + tile, u0:u0 + tile] = np.clip(skin, 0, 255).astype(np.uint8)
    mid_y = 0.5 * (y_lo + y_hi)
    dv = verts[eye_verts]
    uu = np.clip(0.5 + ((dv - origin) @ right) / max(x_half * 2.6, 1e-6), 0.04, 0.96)
    vv = np.clip(0.5 + (dv[:, 1] - mid_y) / max((y_hi - y_lo) * 1.35, 1e-6), 0.04, 0.96)
    row = v0 + (1.0 - vv) * (tile - 1)
    uvs[eye_verts, 0] = (u0 + uu * (tile - 1)) / max(w - 1, 1)
    uvs[eye_verts, 1] = 1.0 - row / max(h - 1, 1)
    half = max(x_half, height * 0.02)
    sep = half * 0.36
    eye_r = half * 0.13
    eye_y = y_lo + 0.78 * (y_hi - y_lo)
    side_v = (verts - origin) @ right
    for sign in (-1.0, 1.0):
        here = eye_verts & (np.abs(y - eye_y) < eye_r * 1.6) & (np.abs(side_v - sign * sep) < eye_r * 1.4)
        if int(here.sum()) < 12:
            continue
        ty, tx = _texel_index(uvs[here], h, w)
        cx = float(np.median(tx))
        cy = float(np.median(ty))
        _paint_eye_disk(image, cx, cy, 22, 13)
        print(f"[eyes] радужка {sign:+.0f} px {cx:.0f},{cy:.0f} verts {int(here.sum())}")
    print(f"[eyes] маска лица: span y {y_span:.3f}, вершин {int(eye_verts.sum())}, тайл {u0},{v0}")
    return int(eye_verts.sum())


def repair_eyes(verts, faces, uvs, image):
    colors = _colors(uvs, image)
    if image is None or uvs is None or len(colors) != len(verts):
        return verts, faces, uvs, image, "нет цвета вершин"
    uvs = np.array(uvs, dtype=np.float32, copy=True)
    image = np.array(image, copy=True)
    mask_n = repaint_face(verts, faces, uvs, image, colors)
    if mask_n > 200:
        report = f"лицо: чёрная маска заменена кожей и глазами, вершин {mask_n}"
        return verts.astype(np.float32), faces.astype(np.int64), uvs, image, report
    eyes = find_eyes(verts, colors)
    if len(eyes) != 2:
        return verts, faces, uvs, image, "глаза не найдены, сетка не менялась"
    # Открытые глаза: закрашиваем только чёрные пиксели ресниц, UV не переносим.
    outward, right, up, _y0, _height = _body_axes(verts)
    lum = colors.mean(1)
    h, w = image.shape[:2]
    painted = 0
    for eye in eyes:
        center = np.asarray(eye["center"], dtype=np.float64)
        radius = max(float(eye["tall"]) * 0.85, 0.004)
        d = verts - center
        q = ((d @ right) / (radius * 2.4)) ** 2 + ((d @ up) / (radius * 1.6)) ** 2
        lash = (q < 1.0) & (np.abs(d @ outward) < radius * 5.0) & (lum < 52)
        lash &= np.linalg.norm(d, axis=1) > radius * 0.45
        if int(lash.sum()) < 30:
            continue
        skin_m = (np.linalg.norm(d, axis=1) < radius * 3.0) & (lum > 160)
        skin = colors[skin_m].mean(0) if int(skin_m.sum()) > 20 else np.array([214.0, 186.0, 176.0])
        ty, tx = _texel_index(uvs, h, w)
        own = np.zeros((h, w), dtype=np.uint8)
        own[ty[lash], tx[lash]] = 1
        other = ~lash
        own[ty[other], tx[other]] = 0
        ys, xs = np.nonzero(own)
        image[ys, xs] = np.clip(skin, 0, 255).astype(np.uint8)
        painted += int(ys.size)
    if painted < 40:
        return verts, faces, uvs, image, "ресниц не найдено, сетка не менялась"
    report = f"глаза: закрашены ресницы, {painted} пикселей"
    return verts.astype(np.float32), faces.astype(np.int64), uvs, image, report


def _same_vertex_count(tensor, count):
    if tensor is None or not torch.is_tensor(tensor) or tensor.ndim < 2:
        return False
    return int(tensor.shape[-2]) == int(count)


def _store(mesh, verts, faces, uvs, image):
    import copy
    out = copy.copy(mesh)
    out.vertices = torch.from_numpy(np.ascontiguousarray(verts, dtype=np.float32)).unsqueeze(0)
    out.faces = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).unsqueeze(0)
    out.uvs = torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).unsqueeze(0)
    tex = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    out.texture = tex
    out.vertex_counts = None
    out.face_counts = None
    # Позиции те же, нормали со сглаживания остаются. Касательные завязаны на UV и уже неверны.
    normals = getattr(mesh, "normals", None)
    out.normals = normals if _same_vertex_count(normals, len(verts)) else None
    out.tangents = None
    colors = getattr(mesh, "vertex_colors", None)
    if colors is not None:
        n = colors.shape[-2] if torch.is_tensor(colors) and colors.ndim >= 2 else None
        if n != len(verts):
            out.vertex_colors = None
    return out


class GarageKitEyesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Снимает рваную чёрную маску с лица. Тонкая повязка и нормали сетки остаются."}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"
    CATEGORY = "3d/garage_kit"

    def run(self, mesh):
        verts, faces, uvs, image = _as_arrays(mesh)
        if uvs is None or image is None:
            print("[eyes] нет UV или текстуры, пропуск")
            return (mesh,)
        v, f, uv, img, report = repair_eyes(verts, faces, uvs, image)
        print("[eyes]", report)
        if v is verts:
            return (mesh,)
        return (_store(mesh, v, f, uv, img),)


NODE_CLASS_MAPPINGS = {"GarageKitEyes": GarageKitEyesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GarageKitEyes": "Глаза · убрать ресницы"}
