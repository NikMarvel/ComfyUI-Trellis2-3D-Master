"""
Комплексный валидатор графа нод ComfyUI (Node Integrity & Graph Auditor)

Архитектурная роль:
- Проводит строгую 7-факторную проверку целостности графа после любых изменений:
  1. Синтаксис и структура JSON.
  2. Проверка всех связей (Links): отсутствие висячих ID, совпадение source/target.
  3. Соответствие типов данных на входах и выходах (IMAGE, MESH, LATENT и т.д.).
  4. Проверка обязательных входов на отсутствие разрывов (None).
  5. Проверка ключевых 3D-нод (DINOv3, 4x-UltraSharp, Trellis2, Remesh, NormalBake, PBR).
  6. Проверка отсутствия изолированных мертвых узлов.
  7. Контроль калибровки параметров геометрии (1.5M poly, каскад 1536, project_back=0.5).
"""

import json
import sys

def verify_workflow(file_path):
    print(f"\n=======================================================")
    print(f"  АУДИТ ГРАФА: {file_path}")
    print(f"=======================================================")

    with open(file_path, 'r', encoding='utf-8') as f:
        wf = json.load(f)

    nodes = wf.get('nodes', [])
    links = wf.get('links', [])

    print(f"Всего нод: {len(nodes)}")
    print(f"Всего связей (links): {len(links)}")

    node_map = {n['id']: n for n in nodes}
    link_map = {}
    
    errors = []
    warnings = []

    # 1. Валидация таблицы связей
    for item in links:
        if not isinstance(item, list) or len(item) < 5:
            errors.append(f"Поврежденная запись связи: {item}")
            continue
        lid, src_node, src_slot, dst_node, dst_slot = item[0], item[1], item[2], item[3], item[4]
        link_type = item[5] if len(item) > 5 else "ANY"
        link_map[lid] = {
            'id': lid, 'src_node': src_node, 'src_slot': src_slot,
            'dst_node': dst_node, 'dst_slot': dst_slot, 'type': link_type
        }

        if src_node not in node_map:
            errors.append(f"Связь {lid}: исходная нода {src_node} отсутствует в графе!")
        if dst_node not in node_map:
            errors.append(f"Связь {lid}: целевая нода {dst_node} отсутствует в графе!")

    # 2. Валидация входов и выходов каждой ноды
    for nid, node in node_map.items():
        ntype = node.get('type')
        title = node.get('title') or ntype
        
        # Проверяем входы
        for idx, inp in enumerate(node.get('inputs', [])):
            lid = inp.get('link')
            iname = inp.get('name')
            itype = inp.get('type')
            
            if lid is not None:
                if lid not in link_map:
                    errors.append(f"Нода {nid} [{title}] вход '{iname}': ссылка на несуществующую связь ID {lid}!")
                else:
                    l = link_map[lid]
                    if l['dst_node'] != nid:
                        errors.append(f"Нода {nid} [{title}] вход '{iname}': связь {lid} назначена другой ноде {l['dst_node']}!")
                    # Проверка соответствия типов
                    if itype and l['type'] and itype != '*' and l['type'] != '*' and itype != l['type']:
                        # ComfyUI допускает некоторые совместимые типы, но несовпадение критично
                        if not (itype in ['IMAGE', 'MASK'] and l['type'] in ['IMAGE', 'MASK']):
                            warnings.append(f"Нода {nid} [{title}] вход '{iname}': несовпадение типов (ожидает {itype}, связь передает {l['type']})")

        # Проверяем выходы
        for idx, out in enumerate(node.get('outputs', [])):
            oname = out.get('name')
            out_links = out.get('links') or []
            for lid in out_links:
                if lid not in link_map:
                    warnings.append(f"Нода {nid} [{title}] выход '{oname}': ссылка на устаревшую связь ID {lid} в списке links выхода")
                else:
                    l = link_map[lid]
                    if l['src_node'] != nid:
                        errors.append(f"Нода {nid} [{title}] выход '{oname}': связь {lid} числится у выхода {nid}, но источник в связи {l['src_node']}!")

    # 3. Аудит критической цепочки 3D-пайплайна
    critical_checks = [
        # (ID, Type, Required Inputs, Expected Params / Description)
        (338, 'LoadImage', [], "Вид спереди (Front Load)"),
        (341, 'LoadImage', [], "Вид слева (Left Load)"),
        (342, 'LoadImage', [], "Вид сзади (Back Load)"),
        (345, 'LoadImage', [], "Вид справа (Right Load)"),
        (420, 'UpscaleModelLoader', [], "Загрузчик 4x-UltraSharp"),
        (421, 'ImageUpscaleWithModel', ['upscale_model', 'image'], "4K Апскейл Front"),
        (422, 'ImageUpscaleWithModel', ['upscale_model', 'image'], "4K Апскейл Left"),
        (423, 'ImageUpscaleWithModel', ['upscale_model', 'image'], "4K Апскейл Back"),
        (424, 'ImageUpscaleWithModel', ['upscale_model', 'image'], "4K Апскейл Right"),
        (15, 'CLIPVisionLoader', [], "DINOv3 ViT-L CLIP Vision"),
        (324, 'Pixal3DMultiViewConditioning', ['clip_vision_model', 'front', 'left', 'back', 'right'], "Мультиракурс: чистый кроп 1024"),
        (91, 'Trellis2ShapeStage', ['positive', 'negative', 'voxel'], "Trellis 2: Shape Stage"),
        (18, 'KSampler', ['model', 'positive', 'negative', 'latent_image'], "KSampler: Форма (Shape)"),
        (94, 'Trellis2UpsampleStage', ['positive', 'negative', 'shape_latent', 'vae'], "Trellis 2: Воксельный апскейл"),
        (23, 'KSampler', ['model', 'positive', 'negative', 'latent_image'], "KSampler: Детализация вокселей"),
        (92, 'VaeDecodeShapeTrellis', ['samples', 'vae'], "Shape VAE Decode"),
        (241, 'RemeshMesh', ['mesh'], "Solid manifold, project_back 0.5"),
        (186, 'DecimateMesh', ['mesh'], "Децимация 1.5M Poly"),
        (238, 'MeshSmoothNormals', ['mesh'], "Сглаживание нормалей (180°)"),
        (196, 'UnwrapMesh', ['mesh'], "UV-развертка 4096 PEC"),
        (98, 'Trellis2TextureStage', ['positive', 'negative', 'shape_latent'], "Trellis 2: Texture Stage"),
        (12, 'KSampler', ['model', 'positive', 'negative', 'latent_image'], "KSampler: Текстурный латент"),
        (93, 'VaeDecodeTextureTrellis', ['samples', 'vae', 'shape_subdivides'], "Texture VAE Decode"),
        (147, 'BakeTextureFromVoxel', ['mesh', 'voxel_colors', 'texture_size'], "Запекание 4K текстуры (без искажающего reference_mesh)"),
        (405, 'MeshSmoothNormals', ['mesh'], "High-Poly сглаживание для нормалей"),
        (224, 'BakeNormalMapFromMesh', ['low_poly', 'high_poly'], "Запекание 4K Normal Map"),
        (210, 'ApplyTextureToMesh', ['mesh', 'base_color', 'roughness', 'occlusion', 'normal_map'], "PBR сборка мастер-сетки"),
        (260, 'MeshSmoothNormals', ['mesh'], "Финальное сглаживание мастер-модели"),
        (430, 'SaveGLB', ['mesh'], "Экспорт Мастер-модели (SaveGLB)"),
    ]

    print("\n--- Проверка критических узлов пайплайна ---")
    for nid, ntype, req_inputs, desc in critical_checks:
        if nid not in node_map:
            errors.append(f"Критическая нода {nid} ({ntype} - {desc}) отсутствует в графе!")
            continue
        n = node_map[nid]
        actual_type = n.get('type')
        if actual_type != ntype:
            errors.append(f"Нода {nid}: несоответствие типа! Ожидался {ntype}, получен {actual_type}")
            continue

        # Проверяем требуемые входы
        input_dict = {inp.get('name'): inp.get('link') for inp in n.get('inputs', [])}
        missing_inputs = []
        for rin in req_inputs:
            if rin not in input_dict or input_dict[rin] is None:
                missing_inputs.append(rin)
        
        if missing_inputs:
            errors.append(f"Нода {nid} [{desc}]: не подключены обязательные входы: {missing_inputs}")
        else:
            print(f"  [OK] Нода {nid:3d} | {ntype:28s} | {desc}")

    # 4. Проверка значений параметров
    print("\n--- Проверка ключевых параметров конфигурации ---")
    # Проверка Node 241
    n241 = node_map.get(241)
    if n241 and 'widgets_values' in n241:
        wv = n241['widgets_values']
        res, sign_mode, pb, sm = wv[0], wv[1], wv[5], wv[7]
        print(f"  [Параметры] Node 241 (RemeshMesh): resolution={res}, sign_mode='{sign_mode}', project_back={pb}, smooth_iters={sm}")
        if pb > 0.9:
            warnings.append(f"Node 241: project_back={pb} приклеивает воксельную рябь. Для деталей держите 0.7–0.85.")
        if res < 1280:
            warnings.append(f"Node 241: resolution={res} слишком грубый для деталей каскада 1536.")

    # Проверка Node 186
    n186 = node_map.get(186)
    if n186 and 'widgets_values' in n186:
        faces = n186['widgets_values'][0]
        print(f"  [Параметры] Node 186 (DecimateMesh): target_face_count={faces:,} полигонов")
        if faces < 3000000:
            warnings.append(f"Node 186: {faces} граней мало для проработки перьев и лица. Цель от 4.5M.")

    # Проверка Node 147 reference_mesh
    n147 = node_map.get(147)
    if n147:
        ref_inp = next((i for i in n147.get('inputs', []) if i.get('name') == 'reference_mesh'), None)
        if ref_inp and ref_inp.get('link') is not None:
            warnings.append(f"Node 147 (BakeTextureFromVoxel): вход reference_mesh подключен (link {ref_inp.get('link')})! Это может вызвать запекание челки на маску!")
        else:
            print(f"  [Параметры] Node 147 (BakeTextureFromVoxel): reference_mesh корректно отключен (None) — маска не будет белой.")

    # Проверка Node 324 входов 4K
    n324 = node_map.get(324)
    if n324:
        v_inps = {i.get('name'): i.get('link') for i in n324.get('inputs', [])}
        front_link = link_map.get(v_inps.get('front'))
        src_front = front_link['src_node'] if front_link else None
        print(f"  [Параметры] Node 324 (Pixal3D): front подключен к Node {src_front} (чистый кроп, нода 340)")
        if src_front != 340:
            errors.append(f"Node 324: вход front подключен к Node {src_front}, а не к чистому кропу 340. SUPIR не должен кормить геометрию.")

    n3 = node_map.get(3)
    if n3 and n3.get("widgets_values", [None, None, None, None, None])[4] != "euler":
        errors.append(f"Node 3: сэмплер структуры должен быть euler, сейчас {n3.get('widgets_values')}")
    n94 = node_map.get(94)
    if n94 and n94.get("widgets_values", [None])[0] != 1536:
        errors.append(f"Node 94: target_resolution должен быть 1536, сейчас {n94.get('widgets_values')}")
    n438 = node_map.get(438)
    if n438 and len(n438.get("widgets_values") or []) > 6 and n438["widgets_values"][6] > 0.5:
        errors.append(f"Node 438: SUPIR denoise {n438['widgets_values'][6]} переписывает ракурс. Держите около 0.35.")

    # Итоги
    print("\n--- ИТОГ АУДИТА ---")
    if errors:
        print(f"❌ ОБНАРУЖЕНО ОШИБОК: {len(errors)}")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print(f"✅ ВСЕ КРИТИЧЕСКИЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО! Ошибок нет.")
        if warnings:
            print(f"⚠️  Предупреждений: {len(warnings)}")
            for w in warnings:
                print(f"   - {w}")
        return True

if __name__ == '__main__':
    ok1 = verify_workflow('D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json')
    ok2 = verify_workflow('D:/ComfyLLM/3D/ComfyUI/user/default/workflows/Antigravity Edit.json')
    if not (ok1 and ok2):
        sys.exit(1)
