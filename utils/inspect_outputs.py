import json

def check_outputs(node_ids):
    with open('D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json', 'r', encoding='utf-8') as f:
        wf = json.load(f)
    nodes = {n['id']: n for n in wf.get('nodes', [])}
    links = {l[0]: l for l in wf.get('links', []) if isinstance(l, list) and len(l) >= 5}

    for nid in node_ids:
        n = nodes[nid]
        for out in n.get('outputs', []):
            print(f"Node {nid} ({n.get('title')}) out '{out.get('name')}' links: {out.get('links')}")
            for lid in out.get('links', []):
                linfo = links.get(lid)
                if linfo:
                    target_node = nodes.get(linfo[3])
                    print(f"   -> Link {lid} to Node {linfo[3]} ({target_node.get('type')}, {target_node.get('title')}) input '{target_node.get('inputs')[linfo[4]].get('name') if linfo[4] < len(target_node.get('inputs')) else 'idx ' + str(linfo[4])}'")

if __name__ == '__main__':
    check_outputs([421, 422, 423, 424])
