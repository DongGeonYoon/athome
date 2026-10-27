#!/usr/bin/env python3
"""
require3 problem 1 check: are object boxes tall enough / not too tall, and do "object on furniture" pairs touch?
  GT scenes:  python bbox_eval.py GT DATASET_DIR [DATASET_DIR ...]      (scenes/<id>/objects.json + gt_objects.json)
  run4:       python bbox_eval.py RUN4 OBJECTS_JSON [OBJECTS_JSON ...]  (no GT: small objects lying on desks / tables)
GT: map objects matched to GT objects (compatible label, XY distance <= 0.6 m, nearest). For every GT pair
child -> parent (ProcTHOR "parent" = the object it rests on, floor excluded) with both matched:
  gap error = (child box bottom - parent box top) - (same from the GT boxes); |error| <= 0.10 m counts as "on".
Top error = predicted box top - GT box top for every matched object (positive = box too tall).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../scene_graph_dataset'))
from build_dataset import compatible

SMALL = {'cup', 'mug', 'remote control', 'remote', 'bottle', 'cell phone', 'phone', 'book', 'keyboard', 'mouse',
         'computer mouse', 'laptop', 'bowl', 'vase', 'lamp', 'desk lamp', 'speaker', 'box', 'container', 'monitor',
         'computer monitor', 'tissue', 'tissue box', 'pen', 'clock', 'alarm clock', 'plant', 'potted plant', 'can'}
SUPPORT = ('desk', 'table')


def box(o):
    c, s = np.asarray(o['bbox_3d']['center']), np.asarray(o['bbox_3d']['size'])
    return c, s, c[2] - s[2] / 2, c[2] + s[2] / 2


def gt_scene(sd):
    objs = json.load(open(os.path.join(sd, 'objects.json')))['objects']
    gtf = os.path.join(sd, 'ground_truth', 'gt_objects.json')
    if not os.path.exists(gtf):
        return None
    gt = [g for g in json.load(open(gtf)) if max(g['size']) >= 0.05]
    byid = {g['id']: g for g in gt}
    C = np.array([box(o)[0][:2] for o in objs]) if objs else np.zeros((0, 2))
    match = {}
    for g in gt:
        d = np.linalg.norm(C - np.asarray(g['position'][:2]), axis=1) if len(C) else np.array([])
        ok = [i for i in np.argsort(d) if d[i] <= 0.6 and compatible(objs[i]['semantic_tag'], g['gt_class'])]
        if ok:
            match[g['id']] = objs[ok[0]]
    top_err = [box(o)[3] - (g['position'][2] + g['size'][2] / 2) for gid, o in match.items() for g in [byid[gid]]]
    gaps = []
    for g in gt:
        p = byid.get(g.get('parent'))
        if p is None or g['id'] not in match or p['id'] not in match or match[g['id']] is match[p['id']]:
            continue
        pred = box(match[g['id']])[2] - box(match[p['id']])[3]
        ref = (g['position'][2] - g['size'][2] / 2) - (p['position'][2] + p['size'][2] / 2)
        gaps.append(pred - ref)
    return np.array(top_err), np.array(gaps)


def summary(name, top_err, gaps):
    t, g = np.asarray(top_err), np.asarray(gaps)
    print(f"{name:16s} matched {len(t):4d}  top error median {np.median(np.abs(t)) * 100:5.1f} cm, too tall >10 cm {np.mean(t > 0.10) * 100:5.1f}% "
          f"({int((t > 0.10).sum())})  | on-pairs {len(g):3d}: |gap error| median {np.median(np.abs(g)) * 100:5.1f} cm, "
          f"within 10 cm {np.mean(np.abs(g) <= 0.10) * 100:5.1f}%")


def on_footprint(c, u, margin=0.05):
    """XY point inside the (yawed) footprint of box u."""
    cu, su = box(u)[0], box(u)[1]; yaw = u['bbox_3d'].get('yaw', 0.0)
    d = c[:2] - cu[:2]; loc = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
    return bool(np.all(np.abs(loc) <= su[:2] / 2 + margin))


def run4(path):
    objs = json.load(open(path))['objects']
    sup = [o for o in objs if any(s in o['semantic_tag'] for s in SUPPORT)]
    rows = []
    for o in objs:
        if o['semantic_tag'] not in SMALL:
            continue
        c, s, bot, top = box(o)
        under = [u for u in sup if on_footprint(c, u) and box(u)[2] < bot + 0.05 and box(u)[3] < top + 0.20]
        if under:
            u = max(under, key=lambda u: box(u)[3])
            rows.append((o['object_id'], o['semantic_tag'], u['object_id'], u['semantic_tag'], box(u)[3], bot - box(u)[3]))
    return sup, rows


if __name__ == '__main__':
    mode, paths = sys.argv[1], sys.argv[2:]
    if mode == 'GT':
        for ds in paths:
            T, G = [], []
            for sd in sorted(glob.glob(os.path.join(ds, 'scenes', '*'))):
                r = gt_scene(sd)
                if r:
                    T += list(r[0]); G += list(r[1])
            summary(os.path.basename(ds.rstrip('/')), T, G)
    else:
        for p in paths:
            sup, rows = run4(p)
            print(f"== {p}")
            print('  desk/table tops: ' + ', '.join(f"{u['object_id']}:{u['semantic_tag']} {box(u)[3]:.2f}" for u in sup))
            g = np.array([r[5] for r in rows])
            for r in rows:
                print(f"  {r[0]:4d} {r[1]:16s} on {r[2]:4d} {r[3]:14s} top {r[4]:.2f}  gap {r[5] * 100:6.1f} cm")
            if len(g):
                print(f"  small objects on desks {len(g)}: gap median {np.median(g) * 100:.1f} cm, |gap| <= 10 cm {np.mean(np.abs(g) <= 0.10) * 100:.0f}%")
