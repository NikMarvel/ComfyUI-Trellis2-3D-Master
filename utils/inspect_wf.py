import json
import sys

def inspect_nodes(node_ids):
    with open('D:/ComfyLLM/3D/Antigravity_Edit_8K_Ultimate.json', 'r', encoding='utf-8') as f:
        wf = json.load(f)
    nodes = {n['id']: n for n in wf.get('nodes', [])}
    links = {l[0]: l for l in wf.get('links', []) if isinstance(l, list) and len(l) >= 5}

    for nid in node_ids:
        if nid in nodes:
            n = nodes[nid]
            t = n.get('type')
            ttl = n.get('title')
            print(f"Node {nid} ({t}, title={ttl}):")
            for inp in n.get('inputs', []):
                lid = inp.get('link')
                linfo = links.get(lid)
                print(f"   input: {inp.get('name')} <- {linfo}")
            for out in n.get('outputs', []):
                print(f"   output: {out.get('name')} (type: {out.get('type')}, links: {out.get('links')})")
            print(f"   widgets: {n.get('widgets_values')}")

if __name__ == '__main__':
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [92, 93, 147, 164, 186, 196, 208, 210, 224, 226, 235, 241, 405, 416, 417]
    inspect_nodes(ids)
