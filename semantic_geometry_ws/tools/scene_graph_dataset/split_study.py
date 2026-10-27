"""
Study: splitting fused objects that absorbed touching neighbours (e.g. adjacent chairs, cushion + sofa).

Fusion keeps observation provenance (which detections went into each map object), then each object is
re-split with one of the methods below; every resulting part is rebuilt from its observations
(points, label vote, CLIP mean, confidence) and cleaned with the v0.3 largest-component step.

  current     v0.3: largest 0.20 m-connected component only (no split)
  dbscan08/10 DBSCAN on object points (eps 0.08 / 0.10 m), observations assigned to the cluster holding most of their voxels
  hdbscan     HDBSCAN on object points (density-adaptive), same observation assignment
  covis       multi-view co-visibility: k = max detections of this object in one frame; if k >= 2, seed k parts
              from that frame's detections and assign every observation to the seed it overlaps most
  obsgraph    observation graph: edge if voxel containment >= 0.3 AND CLIP cosine >= 0.80; parts = components
  covis+graph covis, then obsgraph inside each part

Metrics per scene: objects, implausible small-class objects (> 1.5 m), same-tag pairs within 0.5 m
(over-split proxy), thin bbox ratio, label purity. BEV images of the flagged objects go to <out>/cases/.

Usage: python split_study.py --scenes office0 office4 room0 [--out ~/exp1/split_study]
"""
import argparse, collections, itertools, json, os, sys
import numpy as np, cv2
from sklearn.cluster import DBSCAN, HDBSCAN

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dataset as B
from build_dataset import R
from semantic_geometry_fusion.geometry import compute_geometry, voxel_downsample
from semantic_geometry_fusion.feature_fusion import ObjectFeatureMap, MapObject, denoise_object

SMALL_KEYS = ('chair', 'stool', 'pillow', 'cushion', 'lamp', 'vase', 'monitor', 'plant', 'flower', 'sculpture',
              'box', 'basket', 'bottle', 'cup', 'book', 'clock', 'frame', 'footrest', 'trash', 'bin', 'hassock', 'figurine')
VOX = 0.05


def is_small(tag):
    return any(k in tag for k in SMALL_KEYS) and 'armchair' not in tag or tag == 'armchair'


def implausible(o):
    lim = 1.3 if o.label == 'armchair' else 1.5
    return is_small(o.label) and max(o.size[:2]) > lim


def vox(p):
    return set(map(tuple, np.floor(np.asarray(p) / VOX).astype(np.int64)))


def fuse_with_provenance(frames):
    fmap = ObjectFeatureMap(R.NN_DELTA, R.ASSOC_THR, voxel_size=R.VOXEL)
    prov = collections.defaultdict(list)
    for idx, obs in frames:
        for o in obs:
            obj, *_ = fmap.update(o)
            prov[obj.object_id].append((idx, o))
    return fmap.objects, prov


def rebuild(obs_list, oid):
    pts = voxel_downsample(np.concatenate([o.points_map for _, o in obs_list]), R.VOXEL)
    if len(pts) < 3:
        return None
    g = compute_geometry(pts)
    counts = collections.Counter(o.label for _, o in obs_list)
    f = np.mean([o.clip_feature for _, o in obs_list], axis=0); f /= np.linalg.norm(f)
    return MapObject(object_id=oid, label=counts.most_common(1)[0][0], points_map=pts, clip_feature=f,
                     num_observations=len(obs_list), centroid=g.centroid, size=g.size, yaw=g.yaw,
                     bbox_center=g.bbox_center, confidence=float(np.mean([o.confidence for _, o in obs_list])),
                     label_counts=dict(counts))


def assign_by_point_labels(obs_list, pts, labels):
    """Each observation goes to the cluster that holds most of its voxels; noise (-1) ignored."""
    key = {k: l for k, l in zip(map(tuple, np.floor(pts / VOX).astype(np.int64)), labels)}
    parts = collections.defaultdict(list)
    for item in obs_list:
        votes = collections.Counter(key.get(v, -1) for v in vox(item[1].points_map))
        votes.pop(-1, None)
        parts[votes.most_common(1)[0][0] if votes else -1].append(item)
    return [p for k, p in parts.items() if k != -1 or len(parts) == 1]


def split_points(obs_list, method):
    pts = voxel_downsample(np.concatenate([o.points_map for _, o in obs_list]), VOX)
    if len(pts) < 20:
        return [obs_list]
    if method.startswith('dbscan'):
        lab = DBSCAN(eps=float(method[6:]) / 100, min_samples=5).fit_predict(pts)
    else:
        lab = HDBSCAN(min_cluster_size=max(15, len(pts) // 20)).fit_predict(pts)
    return assign_by_point_labels(obs_list, pts, lab)


def overlap(a, b):
    return len(a & b) / max(1, min(len(a), len(b)))


def split_covis(obs_list):
    per_frame = collections.defaultdict(list)
    for item in obs_list:
        per_frame[item[0]].append(item)
    seed_frame, seeds = max(per_frame.items(), key=lambda kv: len(kv[1]))
    if len(seeds) < 2:
        return [obs_list]
    seed_vox = [vox(s[1].points_map) for s in seeds]
    seed_c = [s[1].points_map.mean(0) for s in seeds]
    parts = [[s] for s in seeds]
    for item in obs_list:
        if item[0] == seed_frame and any(item is s for s in seeds):
            continue
        v = vox(item[1].points_map); c = item[1].points_map.mean(0)
        ov = [overlap(v, sv) for sv in seed_vox]
        k = int(np.argmax(ov)) if max(ov) > 0 else int(np.argmin([np.linalg.norm(c - sc) for sc in seed_c]))
        parts[k].append(item); seed_vox[k] |= v
    return parts


def split_graph(obs_list, contain=0.3, cos=0.80):
    n = len(obs_list)
    if n < 2:
        return [obs_list]
    V = [vox(o.points_map) for _, o in obs_list]
    F = np.array([o.clip_feature for _, o in obs_list])
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in itertools.combinations(range(n), 2):
        if overlap(V[i], V[j]) >= contain and float(F[i] @ F[j]) >= cos:
            parent[find(i)] = find(j)
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[find(i)].append(obs_list[i])
    parts = sorted(comp.values(), key=len, reverse=True)
    # single-observation fragments rejoin the part they overlap most (avoid over-splitting)
    big = [p for p in parts if len(p) >= 2] or parts[:1]
    for p in parts:
        if len(p) < 2 and p is not big[0] and p not in big:
            v = vox(p[0][1].points_map)
            k = int(np.argmax([max(overlap(v, vox(o.points_map)) for _, o in q) for q in big]))
            big[k] = big[k] + p
    return big


def apply(objs, prov, method):
    out, oid, n_split = [], 0, 0
    for o in objs:
        obs_list = prov[o.object_id]
        if method == 'current' or len(obs_list) < 2:
            parts = [obs_list]
        elif method in ('dbscan8', 'dbscan10', 'hdbscan'):
            parts = split_points(obs_list, {'dbscan8': 'dbscan8', 'dbscan10': 'dbscan10', 'hdbscan': 'hdbscan'}[method])
        elif method == 'covis':
            parts = split_covis(obs_list)
        elif method == 'obsgraph':
            parts = split_graph(obs_list)
        elif method == 'covis+graph':
            parts = [q for p in split_covis(obs_list) for q in split_graph(p)]
        n_split += len(parts) > 1
        for p in parts:
            m = rebuild(p, oid)
            if m is not None:
                out.append((denoise_object(m, B.FUSION['denoise_eps']), o.object_id)); oid += 1
    return out, n_split


def metrics(objs):
    exp = [o for o in objs if B.keep_for_scene_graph(o)]
    q = B.quality(exp)
    pairs = sum(a.label == b.label and np.linalg.norm(a.centroid - b.centroid) < 0.5 for a, b in itertools.combinations(exp, 2))
    return dict(objects=len(exp), implausible_small=int(sum(map(implausible, exp))),
                same_tag_pairs_0_5m=int(pairs), thin_bbox_ratio=q.get('thin_bbox_ratio'), label_purity=q.get('label_purity'))


def draw_case(path, title, groups):
    """groups: list of (method name, list of point arrays); BEV scatter per method side by side."""
    allp = np.concatenate([p for _, ps in groups for p in ps])
    lo, hi = allp[:, :2].min(0) - 0.1, allp[:, :2].max(0) + 0.1
    s = 240 / max(hi - lo); W, H = int((hi - lo)[0] * s) + 1, int((hi - lo)[1] * s) + 1
    panels = []
    cols = [(230, 80, 60), (60, 160, 60), (60, 90, 230), (200, 160, 30), (160, 60, 200), (40, 190, 190), (120, 120, 120)]
    for name, ps in groups:
        img = np.full((H + 24, W, 3), 255, np.uint8)
        for k, p in enumerate(ps):
            q = ((p[:, :2] - lo) * s).astype(int)
            img[24 + H - 1 - np.clip(q[:, 1], 0, H - 1), np.clip(q[:, 0], 0, W - 1)] = cols[k % len(cols)]
        cv2.putText(img, f'{name}: {len(ps)}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        panels.append(img)
    canvas = np.hstack([np.pad(p, ((0, 0), (0, 6), (0, 0)), constant_values=255) for p in panels])
    cv2.putText(canvas, title, (4, canvas.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(path, canvas)


METHODS = ['current', 'dbscan8', 'dbscan10', 'hdbscan', 'covis', 'obsgraph', 'covis+graph']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=os.path.expanduser('~/exp1'))
    ap.add_argument('--suffix', default='rigs2')
    ap.add_argument('--scenes', nargs='+', default=['room0', 'office0', 'office1', 'office3', 'office4'])
    ap.add_argument('--out', default=os.path.expanduser('~/exp1/split_study'))
    ap.add_argument('--obs-eps', nargs='+', type=float, default=[0.0], help='per-observation DBSCAN eps (0 = off)')
    ap.add_argument('--methods', nargs='+', default=METHODS)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'cases'), exist_ok=True)
    table = {}
    for scene in args.scenes:
        R.set_scene(scene)
        sd = os.path.join(args.exp, 'replica_rig', f'{scene}_rig')
        results = {}
        for eps in args.obs_eps:
            frames, _, _ = B.frame_observations(sd, os.path.join(sd, f'gsa_detections_ram_{args.suffix}'), dedup=True,
                                                obs_cluster_eps=eps or None)
            objs, prov = fuse_with_provenance(frames)
            for m in args.methods:
                results[f'obs{eps:.2f}/{m}'] = apply(objs, prov, m)
        table[scene] = {m: dict(metrics([o for o, _ in r[0]]), objects_split=int(r[1])) for m, r in results.items()}
        for m in results:
            print(f'{scene:8s} {m:22s}', json.dumps(table[scene][m]), flush=True)
        # case images: original objects flagged implausible under 'current'
        base = f'obs{args.obs_eps[0]:.2f}/current'
        cur = {src: o for o, src in results[base][0]}
        for src, o in cur.items():
            if not (B.keep_for_scene_graph(o) and implausible(o)):
                continue
            groups = [(m, [p.points_map for p, s in results[m][0] if s == src]) for m in results if m.startswith(f'obs{args.obs_eps[0]:.2f}/')]
            draw_case(os.path.join(args.out, 'cases', f'{scene}_obj{src:03d}_{o.label.replace(" ", "_")}.png'),
                      f'{scene} obj {src} "{o.label}" size {np.round(o.size, 2).tolist()} obs {o.num_observations}', groups)
    json.dump(table, open(os.path.join(args.out, 'split_study.json'), 'w'), indent=2)
    keys = list(next(iter(table.values())).keys())
    tot = {m: {k: sum(table[s][m][k] for s in table) for k in ('objects', 'implausible_small', 'same_tag_pairs_0_5m', 'objects_split')}
           for m in keys}
    print('TOTAL'); [print(f'  {m:12s}', json.dumps(v)) for m, v in tot.items()]
    json.dump(dict(per_scene=table, total=tot), open(os.path.join(args.out, 'split_study.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
