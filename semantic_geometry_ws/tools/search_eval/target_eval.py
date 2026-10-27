#!/usr/bin/env python3
"""
Search-target detection with / without the target names in the DINO query (require3 items 4-5), on the
search_eval scenes (commands.json from make_search_set.py, ground truth gt_objects.json).
Per keyframe the 3D observations are built exactly like the dataset / online node (mask -> LiDAR points, depth
fallback, DBSCAN, stray-point removal). A target detection = observation whose label is one of the target's
query_names with confidence >= 0.4 (what the search server accepts). It is correct (TP) when its centre lies in a
GT box of the target's classes (+0.15 m), otherwise wrong (FP; the GT class of the object it really was is noted).
Missed-name check: observations inside a target's GT box whose label is NOT a query name (the "remote" / "device"
case of require3).
  python target_eval.py [--root ~/exp1/search_eval] [--base v2] [--tgt v2tgt]
"""
import argparse, collections, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../scene_graph_dataset'))
import build_dataset as B

MARGIN, CONF = 0.15, 0.4


def inside(p, g, m=MARGIN):
    return bool(np.all(np.abs(np.asarray(p) - np.asarray(g['position'])) <= np.asarray(g['size']) / 2 + m))


def evaluate(sd, det, cmd):
    gt = json.load(open(os.path.join(sd, 'gt_objects.json')))
    frames, n_det, _ = B.frame_observations(sd, os.path.join(sd, f'gsa_detections_ram_{det}'), dedup=True,
                                            obs_cluster_eps=B.FUSION['obs_cluster_eps'])
    res = {}
    for t in cmd['targets']:
        inst = [g for g in gt if g['gt_class'] in t['gt_classes']]
        tp_frames, tp, fp, fp_what, other_names = set(), 0, 0, collections.Counter(), collections.Counter()
        for idx, obs in frames:
            for o in obs:
                c = o.bbox_center
                hit = any(inside(c, g) for g in inst)
                if o.label in t['query_names'] and o.confidence >= CONF:
                    if hit:
                        tp += 1; tp_frames.add(idx)
                    else:
                        fp += 1
                        near = [g for g in gt if inside(c, g, 0.05)]
                        fp_what[min(near, key=lambda g: max(g['size']))['gt_class'] if near else '(none)'] += 1
                elif hit and max(o.size) < 1.5:
                    other_names[o.label if o.label not in t['query_names'] else f'{o.label} (conf<0.4)'] += 1
        vf = set(t['visible_frames'])
        res[t['name']] = dict(present=t['present'], found=tp > 0, tp=tp, fp=fp,
                              frame_recall=round(len(tp_frames & vf) / len(vf), 3) if vf else None,
                              fp_was=dict(fp_what.most_common(4)), other_labels=dict(other_names.most_common(4)))
    return res, n_det, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/exp1/search_eval'))
    ap.add_argument('--base', default='v2'); ap.add_argument('--tgt', default='v2tgt')
    a = ap.parse_args()
    cmds = json.load(open(os.path.join(a.root, 'commands.json')))['scenes']
    out, tot = {}, {c: collections.Counter() for c in (a.base, a.tgt)}
    for scene, cmd in cmds.items():
        sd = os.path.join(a.root, 'sources', scene)
        if len(glob.glob(os.path.join(sd, f'gsa_detections_ram_{a.tgt}', '*.pkl.gz'))) == 0:
            print(f'{scene}: no {a.tgt} detections yet'); continue
        out[scene] = {}
        for cond in (a.base, a.tgt):
            r, n_det, n_fr = evaluate(sd, cond, cmd)
            out[scene][cond] = dict(targets=r, detections=n_det, frames=n_fr)
            for name, v in r.items():
                k = 'present' if v['present'] else 'absent'
                tot[cond][f'{k}_targets'] += 1; tot[cond][f'{k}_found'] += v['found'] if v['present'] else v['fp'] > 0
                tot[cond][f'{k}_tp'] += v['tp']; tot[cond][f'{k}_fp'] += v['fp']
                tot[cond]['other_label_obs'] += sum(v['other_labels'].values())
            tot[cond]['detections'] += n_det; tot[cond]['frames'] += n_fr
        print(f"\n== {scene} (command {cmd['command_id']})")
        for name in out[scene][a.tgt]['targets']:
            b, t = out[scene][a.base]['targets'][name], out[scene][a.tgt]['targets'][name]
            print(f"  {name:14s} {'present' if t['present'] else 'ABSENT ':7s} | without query: found {b['found']!s:5s} TP {b['tp']:3d} FP {b['fp']:3d} "
                  f"recall {b['frame_recall']} other names {b['other_labels']}\n  {'':22s} | with query:    found {t['found']!s:5s} TP {t['tp']:3d} FP {t['fp']:3d} "
                  f"recall {t['frame_recall']} FP were {t['fp_was']} other names {t['other_labels']}")
    print('\n== total')
    for cond in (a.base, a.tgt):
        c = tot[cond]
        print(f"  {cond:6s}: present targets found {c['present_found']}/{c['present_targets']}, TP {c['present_tp']}, FP {c['present_fp']} | "
              f"absent targets falsely found {c['absent_found']}/{c['absent_targets']} (FP {c['absent_fp']}) | "
              f"target-object observations with another name {c['other_label_obs']} | detections/frame {c['detections'] / max(c['frames'], 1):.1f}")
    json.dump(dict(per_scene=out, total={k: dict(v) for k, v in tot.items()}), open(os.path.join(a.root, 'target_eval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
