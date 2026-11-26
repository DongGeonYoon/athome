#!/usr/bin/env python3
"""Step 1 of the label correction: write the corrected detections next to the CG-D ones.

  python prepare.py SCENE_DIR LABELS [--exclude-keyframes 0] [--in-suffix v2] [--out-suffix v2labeled]

- gsa_detections_ram_<out-suffix>/: same detections, tag from LABELS; detections not listed and all detections of excluded keyframes
  get confidence 0, so the builder's confidence gate drops them (detection ids stay the same). The CG-D folder is not touched.
- Excluded keyframes (e.g. taken before the localization pose was set) also take their LiDAR map points out of map_points.npz:
  points whose scan origin lies within 0.1 m of the keyframe's LiDAR position. The original is kept as map_points_raw.npz.
"""
import argparse, glob, gzip, os, pickle, shutil
import numpy as np
from common import read_labels

ap = argparse.ArgumentParser()
ap.add_argument('scene_dir'); ap.add_argument('labels')
ap.add_argument('--exclude-keyframes', default='', help='comma separated keyframe numbers kept out of 3D')
ap.add_argument('--in-suffix', default='v2'); ap.add_argument('--out-suffix', default='v2labeled')
a = ap.parse_args()
labels = read_labels(a.labels); excl = {int(x) for x in a.exclude_keyframes.split(',') if x.strip()}
gi = os.path.join(a.scene_dir, f'gsa_detections_ram_{a.in_suffix}'); go = os.path.join(a.scene_dir, f'gsa_detections_ram_{a.out_suffix}')
os.makedirs(go, exist_ok=True)
tags = sorted({t for v in labels.values() for t in v.values()}); kept = total = 0
for fp in sorted(glob.glob(gi + '/*.pkl.gz')):
    k = int(os.path.basename(fp)[5:11]); g = pickle.load(gzip.open(fp, 'rb')); classes = list(g['classes'])
    classes += [t for t in tags if t not in classes]
    conf = np.array(g['confidence'], copy=True); cid = np.array(g['class_id'], copy=True); keep = labels.get(k, {})
    assert all(i < len(conf) for i in keep), f'keyframe {k}: detection id out of range'
    for i in range(len(conf)):
        total += 1
        if i in keep and k not in excl:
            cid[i] = classes.index(keep[i]); kept += 1
        else:
            conf[i] = 0.0
    g['classes'], g['confidence'], g['class_id'] = classes, conf, cid
    pickle.dump(g, gzip.open(os.path.join(go, os.path.basename(fp)), 'wb'))
print(f'[prepare] {len(tags)} tags, kept {kept} of {total} detections -> {go}')
if excl:
    raw, cur = os.path.join(a.scene_dir, 'map_points_raw.npz'), os.path.join(a.scene_dir, 'map_points.npz')
    if not os.path.exists(raw):
        shutil.copy(cur, raw)
    m = np.load(raw); o = m['origins'].astype(float); keep = np.ones(len(o), bool)
    for k in excl:
        p = np.load(os.path.join(a.scene_dir, 'lidar', f'frame{k:06d}.npz'))['T_map_sensor'][:2, 3]
        keep &= np.linalg.norm(o[:, :2] - p, axis=1) > 0.1
    np.savez_compressed(cur, points=m['points'][keep], origins=m['origins'][keep])
    print(f'[prepare] map points {len(o)} -> {int(keep.sum())} (keyframes {sorted(excl)} excluded)')
