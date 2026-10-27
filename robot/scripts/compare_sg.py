#!/usr/bin/env python3
"""Compare Scene Graph objects.json files of one scan with the monitor's checks (no ROS needed).
  python3 compare_sg.py NAME=objects.json [NAME=objects.json ...] [--floor-z -0.41 --map-yaml ...]"""
import argparse, json, math, sys, os
from collections import Counter
import numpy as np, cv2, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor_run import compatible, SMALL

ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--floor-z', type=float, default=-0.41)
ap.add_argument('--bev', default='', help='draw the last file on the map into this png')
ap.add_argument('--map-yaml', default=os.path.expanduser('~/fast_lio_ws/maps/lab_map_20261005_174210_structural.yaml'))
a = ap.parse_args()
my = yaml.safe_load(open(a.map_yaml)); grid = cv2.imread(os.path.join(os.path.dirname(a.map_yaml), my['image']), 0)[::-1]
res, ox, oy = my['resolution'], my['origin'][0], my['origin'][1]
print(f"{'run':16s} {'objs':>4s} {'n>=3':>5s} {'splits<0.5m':>11s} {'dups<0.15m':>10s} {'overmerge':>9s} {'>1.5m':>5s} {'near occ':>8s} tags")
for f in a.files:
    name, path = f.split('=', 1)
    O = json.load(open(path))['objects']
    C = np.array([o['bbox_3d']['center'] for o in O]); L = [o['semantic_tag'] for o in O]
    pairs = [(i, j, float(np.linalg.norm(C[i, :2] - C[j, :2]))) for i in range(len(O)) for j in range(i + 1, len(O))
             if compatible(L[i], L[j]) and np.linalg.norm(C[i, :2] - C[j, :2]) < 0.5]
    over = sum(1 for o in O if o['semantic_tag'] in SMALL and max(o['bbox_3d']['size'][:2]) > 1.5)
    high = sum(1 for o in O if o['bbox_3d']['center'][2] - o['bbox_3d']['size'][2] / 2 - a.floor_z > 1.5)
    occ = 0
    for o in O:
        i, j = int((o['centroid']['y'] - oy) / res), int((o['centroid']['x'] - ox) / res)
        w = grid[max(0, i - 4):i + 5, max(0, j - 4):j + 5]
        occ += bool((w < 100).any())          # dark pixel = occupied in the map image
    n3 = sum(o['num_observations'] >= 3 for o in O)
    last = (O, name)
    print(f"{name:16s} {len(O):4d} {n3:5d} {len(pairs):11d} {sum(d < 0.15 for *_, d in pairs):10d} {over:9d} {high:5d} {occ:8d} "
          f"{Counter(L).most_common(8)}")

if a.bev:
    O, name = last
    img = cv2.cvtColor(cv2.resize(grid[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    H = img.shape[0]; px = lambda x, y: (int((x - ox) / res * 2), int(H - (y - oy) / res * 2))
    for o in O:
        c, sz, yw = o['bbox_3d']['center'], o['bbox_3d']['size'], o['bbox_3d']['yaw']
        R = np.array([[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]])
        poly = np.array([px(*(R @ np.array([sz[0] / 2 * u, sz[1] / 2 * v]) + c[:2])) for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1))], np.int32)
        high = c[2] - sz[2] / 2 - a.floor_z > 1.5
        col = (160, 160, 160) if high else (0, 150, 0) if o['num_observations'] >= 3 else (0, 120, 230)
        cv2.polylines(img, [poly], True, col, 2); cv2.putText(img, o['semantic_tag'], px(*c[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
    cv2.putText(img, f"{name}: {len(O)} objects (green >= 3 views, orange 2, grey > 1.5 m above floor)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(a.bev, img); print('bev ->', a.bev)
