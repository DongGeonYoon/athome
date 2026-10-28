#!/usr/bin/env python3
"""
CLIP features of the search-evaluation detections, with right / wrong verdicts, for the search server's CLIP check
(target name must rank first among the household vocabulary). Same 3D observations as target_eval.py.
One JSON line per detection that is either named like a command target or lies on a target's GT object:
  scene, condition ("no_query" = v2, "query" = v2tgt), frame, label, confidence, target, query_names,
  kind: "tp" (target name, conf >= 0.4, on a GT object of the target) | "fp" (target name, conf >= 0.4, elsewhere)
        | "low_conf" (target name, conf < 0.4) | "other_name" (on the target's GT object, another label),
  gt_class (GT class of the object it really is, "" if none), clip_feature (1024 floats: L2-normalised
  ViT-H-14 / laion2b_s32b_b79k image feature of the padded bbox crop, as in /athome/perception/objects)
  python export_target_clip.py [--root ~/exp1/search_eval] -> ROOT/target_clip_features.jsonl, ROOT/label_counts.json
"""
import argparse, collections, json, os
import numpy as np
from target_eval import B, inside, CONF


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.path.expanduser('~/exp1/search_eval'))
    a = ap.parse_args()
    cmds = json.load(open(os.path.join(a.root, 'commands.json')))['scenes']
    out = open(os.path.join(a.root, 'target_clip_features.jsonl'), 'w')
    counts, labels = collections.Counter(), collections.Counter()
    for scene, cmd in cmds.items():
        sd = os.path.join(a.root, 'sources', scene)
        gt = json.load(open(os.path.join(sd, 'gt_objects.json')))
        for cond, det in (('no_query', 'v2'), ('query', 'v2tgt')):
            frames, _, _ = B.frame_observations(sd, os.path.join(sd, f'gsa_detections_ram_{det}'), dedup=True,
                                                obs_cluster_eps=B.FUSION['obs_cluster_eps'])
            for idx, obs in frames:
                for o in obs:
                    labels[o.label] += 1
                    c = o.bbox_center
                    near = [g for g in gt if inside(c, g, 0.05)]
                    actual = min(near, key=lambda g: max(g['size']))['gt_class'] if near else ''
                    for t in cmd['targets']:
                        inst = [g for g in gt if g['gt_class'] in t['gt_classes']]
                        hit = any(inside(c, g) for g in inst)
                        if o.label in t['query_names']:
                            kind = ('tp' if hit else 'fp') if o.confidence >= CONF else 'low_conf'
                        elif hit and max(o.size) < 1.5:
                            kind = 'other_name'
                        else:
                            continue
                        counts[(cond, kind)] += 1
                        f = np.asarray(o.clip_feature, np.float64); f /= np.linalg.norm(f)
                        out.write(json.dumps(dict(scene=scene, condition=cond, frame=idx, label=o.label,
                                                  confidence=round(float(o.confidence), 4), target=t['name'],
                                                  query_names=t['query_names'], target_present=t['present'], kind=kind,
                                                  gt_class=actual, clip_feature=[round(float(v), 6) for v in f])) + '\n')
    out.close()
    json.dump(dict(note='labels of all 3D observations in the evaluation scenes (both conditions)',
                   counts=dict(labels.most_common())), open(os.path.join(a.root, 'label_counts.json'), 'w'), indent=1)
    for k, v in sorted(counts.items()):
        print(k, v)


if __name__ == '__main__':
    main()
