import json
import copy

def update_workflow(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        wf = json.load(f)

    nodes = wf.get('nodes', [])
    links = wf.get('links', [])

    nodes_to_remove = {432, 433, 434, 435, 436}
    # Keep only nodes not in nodes_to_remove
    wf['nodes'] = [n for n in nodes if n.get('id') not in nodes_to_remove]

    # Map node id to node
    node_map = {n['id']: n for n in wf['nodes']}

    # 1. Update Node 324 (Pixal3DMultiViewConditioning) to take upscaled images from 421, 422, 423, 424
    # Find existing links from 421, 422, 423, 424
    # In outputs of 421, 422, 423, 424:
    max_link_id = max(l[0] for l in links if isinstance(l, list) and len(l) > 0)

    # Let's see current links for node 324 inputs
    n324 = node_map.get(324)
    if n324:
        for inp in n324.get('inputs', []):
            name = inp.get('name')
            old_link = inp.get('link')
            # remove old link from links
            links = [l for l in links if l[0] != old_link]
            
            # create new link from upscale node
            target_source = None
            if name == 'front':
                target_source = 421
            elif name == 'left':
                target_source = 422
            elif name == 'back':
                target_source = 423
            elif name == 'right':
                target_source = 424

            if target_source and target_source in node_map:
                max_link_id += 1
                new_link_id = max_link_id
                inp['link'] = new_link_id
                links.append([new_link_id, target_source, 0, 324, n324['inputs'].index(inp), 'IMAGE'])
                # Also add to target_source output links
                src_node = node_map[target_source]
                if src_node.get('outputs') and len(src_node['outputs']) > 0:
                    out_links = src_node['outputs'][0].get('links', [])
                    if out_links is None:
                        out_links = []
                    if new_link_id not in out_links:
                        out_links.append(new_link_id)
                    src_node['outputs'][0]['links'] = out_links

    # 2. In Node 147 (BakeTextureFromVoxel), disconnect reference_mesh (which was pointing to noisy Node 92)
    n147 = node_map.get(147)
    if n147:
        for inp in n147.get('inputs', []):
            if inp.get('name') == 'reference_mesh':
                old_link = inp.get('link')
                if old_link:
                    links = [l for l in links if l[0] != old_link]
                inp['link'] = None

    # 3. In Node 241 (RemeshMesh): set resolution=1536, project_back=0.95, smooth_iters=2
    n241 = node_map.get(241)
    if n241 and 'widgets_values' in n241:
        # widgets: [resolution, sign_mode, fill, qef, band, project_back, fix_poles, smooth_iters, drop_small_components, precluster_max_verts]
        # Current: [1280, 'solid', 20, False, 1.0, 0.05, False, 8, 0.01, 20000000]
        wv = n241['widgets_values']
        wv[0] = 1536  # higher narrow-band DC grid
        wv[1] = 'solid'
        wv[2] = 20.0
        wv[3] = False
        wv[4] = 1.0
        wv[5] = 0.95  # PROJECT BACK 95% to original curved surface! Eliminates polygonal facets!
        wv[6] = True  # fix_poles = True (collapses T-junction artifacts)
        wv[7] = 2     # Taubin 2 iterations (smooths voxel steps without blurring organic details)
        n241['title'] = "Высокодетализированный Solid Manifold (RemeshMesh 1536 ProjectBack=0.95)"

    # 4. In Node 186 (DecimateMesh): set target_face_count=2500000
    n186 = node_map.get(186)
    if n186 and 'widgets_values' in n186:
        n186['widgets_values'][0] = 2500000
        n186['title'] = "Децимация до 2.5M полигонов (DecimateMesh 2.5M Poly - Ультра-детализация)"

    # 5. In Node 430 (SaveGLB): update title
    n430 = node_map.get(430)
    if n430:
        n430['title'] = "Мастер-модель (Full Master 8K - 2.5M Poly Ultra Detail)"
        n430['widgets_values'] = ["3d/2B_Figure_Full_Master_8K_Ultra_Detail"]

    # Clean up removed links
    valid_node_ids = set(node_map.keys())
    wf['links'] = [l for l in links if isinstance(l, list) and len(l) >= 5 and l[1] in valid_node_ids and l[3] in valid_node_ids]

    # Save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)

    print(f"Successfully updated {file_path}")

if __name__ == '__main__':
    update_workflow('D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json')
    update_workflow('D:/ComfyLLM/3D/ComfyUI/user/default/workflows/Antigravity Edit.json')
