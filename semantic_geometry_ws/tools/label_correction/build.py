#!/usr/bin/env python3
"""Step 2: build the dataset from the corrected detections with build_dataset.py (code unchanged) plus two rules, and record which
keyframe detection became which object.

  python build.py --exp EXP --occupancy-map MAP.yaml [--det-suffix v2labeled] [--out-name sg_labeled]
                  [--single-view-tags cup,headphone,plate] [--trace trace.json]

- same-tag association: a detection joins only an object of the same corrected tag; fragments are re-merged only within one tag
- --single-view-tags: these tags are exported with one observation too (small items the LiDAR hits only once or twice)
- --trace: per object, the keyframe detections it was built from. The builder is run unchanged and only watched; the fused result is
  checked against the written objects_all.json.
"""
import argparse, glob, gzip, json, os, pickle, sys
from copy import deepcopy
import common  # noqa: F401  (paths)
import build_dataset as B

ap = argparse.ArgumentParser()
ap.add_argument('--exp', required=True); ap.add_argument('--occupancy-map', required=True)
ap.add_argument('--det-suffix', default='v2labeled'); ap.add_argument('--out-name', default='sg_labeled')
ap.add_argument('--single-view-tags', default=''); ap.add_argument('--trace', default='')
a = ap.parse_args()
SINGLE = {t.strip() for t in a.single_view_tags.split(',') if t.strip()}
where, members, groups = {}, {}, {}


class SameTag(B.ObjectFeatureMap):
    def associate(self, obs, exclude=()):
        return super().associate(obs, set(exclude) | {i for i, o in enumerate(self.objects) if o.label != obs.label})

    def update(self, obs, exclude=()):
        r = super().update(obs, exclude)
        members.setdefault(int(r[0].object_id), []).append((where.get(id(obs)), int(obs.detection_id), obs.label, float(obs.confidence)))
        return r


_merge = B.merge_fragments


def merge_same_tag(objs, *args, **kw):
    """Re-merge within each tag. A shadow run whose vote tables hold object ids (the merge decision uses CLIP and geometry only)
    tells which objects went into each result."""
    tag = lambda o: getattr(o, '_tag', o.label)
    out, shadow_out = [], []
    for t in sorted({tag(o) for o in objs}):
        part = [o for o in objs if tag(o) == t]; out += _merge(part, *args, **kw)
        shadow = deepcopy(part)
        for o in shadow:
            o._tag = t; o.label_counts = {f'@{int(o.object_id)}': 1}; o.label = f'@{int(o.object_id)}'
        shadow_out += _merge(shadow, *args, **kw)
    by = {int(o.object_id): o for o in objs}
    for r in shadow_out:
        groups[int(r.object_id)] = [by[int(x[1:])] for x in r.label_counts]
    return sorted(out, key=lambda o: o.object_id)


_frame_obs = B.frame_observations


def frame_observations(*args, **kw):
    frames, n_det, n_dup = _frame_obs(*args, **kw)
    for idx, obs in frames:
        for o in obs:
            where[id(o)] = idx
    return frames, n_det, n_dup


B.ObjectFeatureMap = SameTag; B.merge_fragments = merge_same_tag; B.frame_observations = frame_observations
_keep = B.keep_for_scene_graph
B.keep_for_scene_graph = lambda o: _keep(o) or (o.label in SINGLE and o.num_observations >= 1)
sys.argv = ['build_dataset.py', '--exp', a.exp, '--sources', 'robot', '--det-suffix', a.det_suffix, '--out-name', a.out_name,
            '--occupancy-map', a.occupancy_map]
B.main()

if a.trace:
    scene = glob.glob(os.path.join(a.exp, a.out_name, 'scenes', 'robot_*'))[0]; src = os.path.join(a.exp, 'robot', 'sources', os.path.basename(scene)[6:])
    gd = os.path.join(src, f'gsa_detections_ram_{a.det_suffix}'); objs = json.load(open(os.path.join(scene, 'objects_all.json')))['objects']
    trace, cache = {}, {}
    for o in objs:
        i = o['object_id']; rows = sorted(r for g in groups.get(i, []) for r in members[int(g.object_id)])
        assert len(rows) == o['num_observations'], (i, len(rows), o['num_observations'])
        L = []
        for kf, did, lab, conf in rows:
            g = cache.setdefault(kf, pickle.load(gzip.open(os.path.join(gd, f'frame{kf:06d}.pkl.gz'), 'rb')))
            L.append(dict(keyframe=kf, image=f'frame{kf:06d}.jpg', detection_id=did, label=lab, confidence=round(conf, 4),
                          bbox_xyxy=[round(float(v), 1) for v in g['xyxy'][did]]))
        trace[str(i)] = L
    json.dump(dict(scene=os.path.basename(scene), objects=trace), open(a.trace, 'w'), indent=1)
    print(f'[build] trace of {len(trace)} objects -> {a.trace}')
