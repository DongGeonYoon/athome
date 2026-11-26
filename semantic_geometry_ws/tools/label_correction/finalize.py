#!/usr/bin/env python3
"""Step 3: one object per single item, placed from several views, and fixed furniture fitted on the LiDAR map. Writes the final dataset.

  python finalize.py CONFIG.yaml SCENE_DIR LABELS BUILT_DATASET OUT_DATASET [--trace trace.json] [--keyframe-json DIR] [--figures DIR]

SCENE_DIR: keyframe folder (robot/sources/<name>), BUILT_DATASET: output of build.py (folder with index.json, schema/, scenes/),
OUT_DATASET: copy of it with the changes below. Settings per room in CONFIG (see configs/room_baram.yaml).

items (tags that exist once in the room; all their objects are replaced by one):
  position  the ray through the bottom of every mask of the tag meets the surface the item stands on (surfaces: desk top or floor);
            median over the views, views farther than 0.45 m from it dropped. Rays alone do not fix the depth: the views of one item are
            nearly parallel.
  points    LiDAR points near the position that fall inside the masks, kept if inside in at least half of the views that see them,
            largest cluster. Used only if >= 20 points whose centre is within 0.3 m of the position; otherwise a placeholder box of
            the configured size at the position (points_source views_only: size not measured).
  feature   mean CLIP of the views used, mean confidence, num_observations = number of views
  position: [x, y] in the config overrides the estimate; inside: <tag> puts the item at that item's position (e.g. a spoon in a bowl).
desks (furniture with a known count): one rectangle per seed polygon fitted on the LiDAR slice at desk-top height; every detection of
  the tag is given to the desk holding most of its LiDAR points.
"""
import argparse, collections, json, os, shutil
import numpy as np, cv2
from common import Keyframes, read_labels, read_config, mean_unit, largest_cluster, box_points
from semantic_geometry_fusion.geometry import compute_geometry

ap = argparse.ArgumentParser()
ap.add_argument('config'); ap.add_argument('scene_dir'); ap.add_argument('labels'); ap.add_argument('built'); ap.add_argument('out')
ap.add_argument('--suffix', default='v2', help='CG-D detection suffix holding the masks and CLIP features')
ap.add_argument('--trace', default='', help='trace.json of build.py: per-object keyframe lists for --keyframe-json')
ap.add_argument('--keyframe-json', default='', help='write objects / objects_all / keyframe labels with their keyframes here')
ap.add_argument('--figures', default='', help='write check figures here')
a = ap.parse_args()
C = read_config(a.config); KF = Keyframes(a.scene_dir, a.suffix); labels = read_labels(a.labels)
excl = set(C.get('exclude_keyframes', [])); surf = C['surfaces']; items = C.get('items', {})


def views_of(tag):
    return [(k, i) for k in sorted(labels) if k not in excl for i, t in labels[k].items() if t == tag]


# ---------------------------------------------------------------- single items
placed = {}
for tag, it in items.items():
    if 'inside' in it:
        continue
    z0 = surf[it.get('surface', 'desk')]; extent = it.get('extent', .5); views = []
    for k, i in views_of(tag):
        m = np.asarray(KF.det(k)['mask'][i], bool); ys, xs = np.nonzero(m); low = ys >= np.percentile(ys, 92)
        d = KF.T[k][:3, :3] @ (KF.Ki @ [xs[low].mean(), ys.max(), 1.0]); o = KF.T[k][:3, 3]
        if d[2] >= -1e-3:
            continue
        p = o + d * ((z0 - o[2]) / d[2])
        if np.linalg.norm(p - o) <= 7:
            views.append(dict(k=k, i=i, m=m, p=p))
    A = np.array([v['p'][:2] for v in views]); c = np.median(A, axis=0)
    for _ in range(3):
        keep = np.linalg.norm(A - c, axis=1) < .45; c = np.median(A[keep], axis=0)
    used = [v for v, f in zip(views, keep) if f]
    cand = []
    for v in used:
        P = KF.cloud(v['k']); P = P[np.linalg.norm(P[:, :2] - c, axis=1) < extent]; P = P[(P[:, 2] > z0 - .03) & (P[:, 2] < z0 + 1.6)]
        if len(P):
            cand.append(P[KF.in_mask(v['k'], v['m'], P)[0]])
    P = np.concatenate(cand) if cand else np.zeros((0, 3))
    if len(P):
        _, idx = np.unique(np.floor(P / .03).astype(int), axis=0, return_index=True); P = P[idx]
        votes = np.zeros(len(P)); seen = np.zeros(len(P))
        for v in used:
            h, inn = KF.in_mask(v['k'], v['m'], P); votes += h; seen += inn
        P = largest_cluster(P[votes >= np.maximum(.5 * seen, 2)], .10)
    pos = np.round(np.array(it['position'] + [z0] if 'position' in it else [c[0], c[1], z0], float), 3)
    lidar = len(P) >= 20 and np.linalg.norm(P.mean(0)[:2] - pos[:2]) <= .3
    P = P.astype(np.float32).astype(float) if lidar else box_points(pos[:2], z0, *it['size'])
    placed[tag] = dict(points=P, clip=mean_unit([KF.clip(v['k'], v['i']) for v in used]), used=[KF.record(v['k'], v['i']) for v in used],
                       conf=float(np.mean([KF.det(v['k'])['confidence'][v['i']] for v in used])),
                       source='lidar' if lidar else ('views_only, position set by hand' if 'position' in it else 'views_only'),
                       report=dict(detections=len(views), view_spread_m=round(float(np.median(np.linalg.norm(A[keep] - c, axis=1))), 3),
                                   rejected_keyframes=sorted({v['k'] for v in views} - {v['k'] for v in used}), surface_position=[round(float(x), 3) for x in pos]),
                       pos=pos)
for tag, it in items.items():
    if 'inside' not in it:
        continue
    base = placed[it['inside']]['pos']; vs = views_of(tag)
    placed[tag] = dict(points=box_points(base[:2], base[2], *it['size']), clip=mean_unit([KF.clip(k, i) for k, i in vs]), used=[KF.record(k, i) for k, i in vs],
                       conf=float(np.mean([KF.det(k)['confidence'][i] for k, i in vs])),
                       source=f"views_only (position of the {it['inside']})", report=dict(detections=len(vs)), pos=base)
for t, p in placed.items():
    p['conf'] = round(p['conf'], 4)

# ---------------------------------------------------------------- desks
desks = {}
dc = C.get('desks')
if dc:
    mp = np.load(os.path.join(a.scene_dir, 'map_points.npz'))['points'].astype(float); z_lo, z_hi = dc['slice_z']
    S = mp[(mp[:, 2] > z_lo) & (mp[:, 2] < z_hi)]
    for n, poly in dc['seeds'].items():
        pl = np.array(poly, np.float32); P = S[[cv2.pointPolygonTest(pl, (float(x), float(y)), False) >= 0 for x, y in S[:, :2]]]
        (cx, cy), _, ang = cv2.minAreaRect(P[:, :2].astype(np.float32)); t = np.radians(ang)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); q = (P[:, :2] - [cx, cy]) @ R
        lo, hi = np.percentile(q, 1, axis=0), np.percentile(q, 99, axis=0); cen = np.array([cx, cy]) + R @ ((lo + hi) / 2); sz = hi - lo
        if sz[0] < sz[1]:
            sz = sz[::-1]; t += np.pi / 2
        t = (t + np.pi / 2) % np.pi - np.pi / 2; R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        desks[n] = dict(center=cen, size=sz, yaw=t, R=R, used=[], feats=[])

    def inrect(P, d, m=.03):
        q = (P[:, :2] - d['center']) @ d['R']
        return (np.abs(q[:, 0]) <= d['size'][0] / 2 + m) & (np.abs(q[:, 1]) <= d['size'][1] / 2 + m)
    unassigned = 0
    for k, i in views_of(dc['tag']):
        m = np.asarray(KF.det(k)['mask'][i], bool); P = KF.cloud(k); Q = P[KF.in_mask(k, m, P)[0]]; Q = Q[(Q[:, 2] > surf['floor'] + .01) & (Q[:, 2] < z_hi + .04)]
        cnt = {n: int(inrect(Q, d).sum()) for n, d in desks.items()}; n = max(cnt, key=cnt.get)
        if cnt[n] < 5:
            unassigned += 1; continue
        desks[n]['used'].append(KF.record(k, i)); desks[n]['feats'].append(KF.clip(k, i))
    for n, d in desks.items():
        q = mp[(mp[:, 2] > surf['floor'] + .01) & (mp[:, 2] < z_hi)]; q = q[inrect(q, d, 0)]
        _, idx = np.unique(np.floor(q / .05).astype(int), axis=0, return_index=True); d['points'] = q[idx]
        d['z'] = (float(np.percentile(d['points'][:, 2], 1)), float(np.percentile(d['points'][:, 2], 99)))
        d['clip'] = mean_unit(d['feats']); d['conf'] = round(float(np.mean([KF.det(u['keyframe'])['confidence'][u['detection_id']] for u in d['used']])), 4)
    print(f'[finalize] desks: {len(desks)}, detections without a desk: {unassigned}')

# ---------------------------------------------------------------- write the dataset
shutil.rmtree(a.out, ignore_errors=True); shutil.copytree(a.built, a.out)
scene = os.path.join(a.out, 'scenes', os.listdir(os.path.join(a.out, 'scenes'))[0])
Oa = json.load(open(f'{scene}/objects_all.json')); Oe = json.load(open(f'{scene}/objects.json'))
replace = set(placed) | ({dc['tag']} if dc else set())
ids = {t: min(o['object_id'] for o in Oa['objects'] if o['semantic_tag'] == t) for t in placed if any(o['semantic_tag'] == t for o in Oa['objects'])}
nxt = max(o['object_id'] for o in Oa['objects']) + 1
for t in placed:
    if t not in ids:
        ids[t] = nxt; nxt += 1
for o in Oa['objects']:
    if o['semantic_tag'] in replace and os.path.exists(f"{scene}/points/obj_{o['object_id']:04d}.npy"):
        os.remove(f"{scene}/points/obj_{o['object_id']:04d}.npy")


def obj(i, tag, P, conf, clip, n, box=None):
    g = compute_geometry(P); c = P.mean(0) if box else g.centroid
    np.save(f'{scene}/points/obj_{i:04d}.npy', P.astype(np.float32))
    return dict(object_id=i, semantic_tag=tag, confidence=conf, centroid=dict(x=round(float(c[0]), 4), y=round(float(c[1]), 4), z=round(float(c[2]), 4)),
                bbox_3d=box or dict(center=[round(float(x), 4) for x in g.bbox_center], size=[round(float(x), 4) for x in g.size], yaw=round(float(g.yaw), 3)),
                clip_feature=[float(x) for x in np.asarray(clip, np.float32)], num_observations=n)


new, used_of, report = [], {}, []
for t, p in placed.items():
    new.append(obj(ids[t], t, p['points'], p['conf'], p['clip'], len(p['used']))); used_of[ids[t]] = p['used']
    report.append(dict(tag=t, object_id=ids[t], position_source=p['source'], views=len(p['used']), **p['report']))
keep_ids = (dc or {}).get('keep_ids', {})
for n, d in desks.items():
    i = keep_ids.get(n)
    if i is None:
        i = nxt; nxt += 1
    box = dict(center=[round(float(d['center'][0]), 3), round(float(d['center'][1]), 3), round((round(d['z'][0], 3) + round(d['z'][1], 3)) / 2, 4)],
               size=[round(float(d['size'][0]), 3), round(float(d['size'][1]), 3), round(round(d['z'][1], 3) - round(d['z'][0], 3), 4)], yaw=round(float(d['yaw']), 3))
    new.append(obj(i, dc['tag'], d['points'], d['conf'], d['clip'], len(d['used']), box)); used_of[i] = d['used']
    report.append(dict(tag=f"{dc['tag']} {n}", object_id=i, position_source='lidar slice at desk height', views=len(d['used'])))
for d in (Oa, Oe):
    d['objects'] = sorted([o for o in d['objects'] if o['semantic_tag'] not in replace] + new, key=lambda o: o['object_id'])
json.dump(Oa, open(f'{scene}/objects_all.json', 'w'), indent=2); json.dump(Oe, open(f'{scene}/objects.json', 'w'), indent=2)
meta = json.load(open(f'{scene}/metadata.json')); meta['counts'].update(objects=len(Oe['objects']), objects_all=len(Oa['objects']))
meta['label_correction'] = dict(note=C.get('note', ''), items=report); json.dump(meta, open(f'{scene}/metadata.json', 'w'), indent=1)
idx = json.load(open(f'{a.out}/index.json')); idx['totals'].update(objects=len(Oe['objects']), objects_all=len(Oa['objects']))
for e in idx['scenes']:
    e.update(objects=len(Oe['objects']), objects_all=len(Oa['objects']), note='Labels corrected by hand; see metadata.json label_correction')
json.dump(idx, open(f'{a.out}/index.json', 'w'), indent=2)
print(f"[finalize] objects {len(Oe['objects'])} (all {len(Oa['objects'])}):", sorted(collections.Counter(o['semantic_tag'] for o in Oe['objects']).items()))

# ---------------------------------------------------------------- per-object keyframes (optional)
if a.keyframe_json:
    os.makedirs(a.keyframe_json, exist_ok=True); tr = json.load(open(a.trace))['objects'] if a.trace else {}
    K = {o['object_id']: ([dict(keyframe=u['keyframe'], image=f"frame{u['keyframe']:06d}.jpg", detection_id=u['detection_id'], label=o['semantic_tag'],
                                confidence=u['confidence'], bbox_xyxy=u['bbox_xyxy']) for u in used_of[o['object_id']]]
                          if o['object_id'] in used_of else tr.get(str(o['object_id']), [])) for o in Oa['objects']}
    for src, dst in (('objects.json', 'objects_with_keyframes.json'), ('objects_all.json', 'objects_all_with_keyframes.json')):
        d = json.load(open(f'{scene}/{src}'))
        for o in d['objects']:
            o['keyframes'] = K[o['object_id']]
        json.dump(d, open(os.path.join(a.keyframe_json, dst), 'w'), indent=2, ensure_ascii=False)
    to_obj = {(x['keyframe'], x['detection_id']): i for i, v in K.items() for x in v}; kept = {o['object_id'] for o in Oe['objects']}; rows = []
    for k in sorted(labels):
        g = KF.det(k); R = []
        for i, t in sorted(labels[k].items()):
            oid = to_obj.get((k, i))
            R.append(dict(detection_id=i, tag=t, original_label=g['classes'][g['class_id'][i]], confidence=round(float(g['confidence'][i]), 4),
                          bbox_xyxy=[round(float(v), 1) for v in g['xyxy'][i]], object_id=oid, in_objects_json=oid in kept))
        rows.append(dict(keyframe=k, image=f'frame{k:06d}.jpg', pose_valid=k not in excl, detections=R))
    json.dump(dict(scene=os.path.basename(scene), keyframes=rows), open(os.path.join(a.keyframe_json, 'keyframe_labels.json'), 'w'), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- check figures (optional)
if a.figures:
    os.makedirs(a.figures, exist_ok=True); tiles = []
    for t, p in placed.items():
        us = p['used']; pick = [us[int(j)] for j in np.linspace(0, len(us) - 1, min(4, len(us)))]; row = []
        for u in pick:
            k = u['keyframe']; im = cv2.imread(os.path.join(a.scene_dir, 'results', f'frame{k:06d}.jpg'))
            uv, inn, _ = KF.project(k, p['points'])
            for q in uv[inn]:
                cv2.circle(im, (int(q[0]), int(q[1])), 3, (0, 255, 0), -1)
            q, ii, _ = KF.project(k, p['pos'][None])
            if ii[0]:
                cv2.drawMarker(im, (int(q[0, 0]), int(q[0, 1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
            x1, y1, x2, y2 = [int(v) for v in u['bbox_xyxy']]; cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 0), 2)
            im = cv2.resize(im, (320, 180)); cv2.putText(im, f'{t} kf{k}', (4, 16), 0, .5, (0, 255, 255), 1, cv2.LINE_AA); row.append(im)
        row += [np.zeros((180, 320, 3), np.uint8)] * (4 - len(row)); tiles.append(np.hstack(row))
    cv2.imwrite(os.path.join(a.figures, 'items_check.jpg'), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if desks:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10), dpi=90); ax.plot(S[:, 0], S[:, 1], '.', ms=.8, color='#1f6fb5')
        for n, d in desks.items():
            q = (np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]]) * d['size']) @ d['R'].T + d['center']; ax.plot(q[:, 0], q[:, 1], 'r-'); ax.text(*d['center'], n, color='r')
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title('desk-height LiDAR slice and fitted desks'); fig.savefig(os.path.join(a.figures, 'desks_check.png'))
