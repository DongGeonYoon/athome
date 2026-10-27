#!/usr/bin/env python3
"""
Compare CG-D variants against a reference (fp32 repo setting) on the same keyframes.
  python precision_compare.py [EXP=~/real/results/run4/precision_cmp] [REF=v2fp32] [OTHERS...]
Frame level: detections matched by box IoU >= 0.5 (Hungarian on 1-IoU); mask IoU, CLIP image-feature cosine,
class agreement of matched pairs. Object level (sg_<variant>/scenes/robot_run4/objects.json): objects matched by
centroid distance <= 0.3 m (Hungarian); label agreement, CLIP cosine, centre and size difference.
"""
import glob, gzip, json, os, pickle, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

E = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/real/results/run4/precision_cmp')
REF = sys.argv[2] if len(sys.argv) > 2 else 'v2fp32'
OTH = sys.argv[3:] or ['v2split', 'mobilesam_v2mobile', 'v2old']
SG = {'v2fp32': 'sg_fp32', 'v2split': 'sg_split', 'mobilesam_v2mobile': 'sg_mobile'}
S = os.path.join(E, 'robot/sources/run4')


def load(det):
    out = {}
    for p in sorted(glob.glob(f'{S}/gsa_detections_ram_{det}/*.pkl.gz')):
        out[os.path.basename(p)] = pickle.load(gzip.open(p))
    return out


def iou_box(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def unit(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def frames(ref, oth):
    r = dict(n_ref=0, n_oth=0, matched=0, cls=0, tags_same=0, frames=0, miou=[], cos=[])
    for k, a in ref.items():
        b = oth.get(k)
        if b is None:
            continue
        r['frames'] += 1; r['tags_same'] += set(a['classes']) == set(b['classes'])
        na, nb = len(a['xyxy']), len(b['xyxy']); r['n_ref'] += na; r['n_oth'] += nb
        if not na or not nb:
            continue
        iou = iou_box(a['xyxy'], b['xyxy']); i, j = linear_sum_assignment(1 - iou)
        for ii, jj in zip(i, j):
            if iou[ii, jj] < 0.5:
                continue
            r['matched'] += 1
            r['cls'] += a['classes'][a['class_id'][ii]] == b['classes'][b['class_id'][jj]]
            ma, mb = a['mask'][ii], b['mask'][jj]
            r['miou'].append((ma & mb).sum() / max((ma | mb).sum(), 1))
            r['cos'].append(float(unit(a['image_feats'][ii]) @ unit(b['image_feats'][jj])))
    return r


def objs(name):
    p = os.path.join(E, name, 'scenes/robot_run4/objects.json')
    return json.load(open(p))['objects'] if os.path.exists(p) else None


def objects(A, B):
    ca = np.array([[o['centroid'][c] for c in 'xyz'] for o in A]); cb = np.array([[o['centroid'][c] for c in 'xyz'] for o in B])
    d = np.linalg.norm(ca[:, None] - cb[None], axis=-1); i, j = linear_sum_assignment(d)
    keep = [(ii, jj) for ii, jj in zip(i, j) if d[ii, jj] <= 0.3]
    lab = np.mean([A[ii]['semantic_tag'] == B[jj]['semantic_tag'] for ii, jj in keep])
    cos = [float(unit(A[ii]['clip_feature']) @ unit(B[jj]['clip_feature'])) for ii, jj in keep]
    sz = [np.abs(np.array(A[ii]['bbox_3d']['size']) - np.array(B[jj]['bbox_3d']['size'])).max() for ii, jj in keep]
    return dict(n_ref=len(A), n_oth=len(B), matched=len(keep), label=lab, cos=np.median(cos), cos_min=min(cos),
                dist=np.median([d[ii, jj] for ii, jj in keep]), size=np.median(sz))


ref = load(REF)
print(f'reference {REF}: {len(ref)} frames, {sum(len(v["xyxy"]) for v in ref.values())} detections\n')
print('== frame level (vs reference) ==')
for o in OTH:
    r = frames(ref, load(o))
    if not r['frames']:
        print(f'{o:20s} (none)'); continue
    print(f"{o:20s} frames {r['frames']}  tags same {r['tags_same']}/{r['frames']}  det {r['n_oth']} vs {r['n_ref']}  "
          f"matched {r['matched']} (recall {r['matched'] / max(r['n_ref'], 1):.3f}, precision {r['matched'] / max(r['n_oth'], 1):.3f})  "
          f"class same {r['cls'] / max(r['matched'], 1):.3f}  mask IoU median {np.median(r['miou']):.3f} p10 {np.percentile(r['miou'], 10):.3f}  "
          f"CLIP cos median {np.median(r['cos']):.4f} p10 {np.percentile(r['cos'], 10):.4f}")
print('\n== objects.json (vs reference) ==')
A = objs(SG[REF])
for o in OTH:
    B = objs(SG.get(o, ''))
    if A is None or B is None:
        continue
    r = objects(A, B)
    print(f"{o:20s} objects {r['n_oth']} vs {r['n_ref']}  matched {r['matched']}  label same {r['label']:.3f}  "
          f"CLIP cos median {r['cos']:.4f} min {r['cos_min']:.4f}  centre diff median {r['dist'] * 100:.1f} cm  size diff median {r['size'] * 100:.1f} cm")
