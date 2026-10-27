#!/usr/bin/env python3
"""
Search-target evaluation set from the ProcTHOR scenes of the dataset (require3 item 5: no run4-like recording with
targets is possible, so search commands are rebuilt on scenes with ground truth).

For every scene: which GT objects of the target catalogue are visible in which keyframe (GT box centre projected
inside the image, camera depth at that pixel agrees with the distance, apparent size >= MIN_PX), then one search
command = up to N_PRESENT visible target categories + one category absent from the house (false-positive probe).
Target names / query_names follow the search server's rule (require3: "lamp" also accepts "desk lamp", "floor lamp").
  python make_search_set.py [--src ~/exp1/procthor/sources] [--out ~/exp1/search_eval] [--scenes 6]
-> OUT/commands.json  {scene: {command_id, targets: [{name, query_names, gt_classes, present, visible_frames}]}}
"""
import argparse, glob, json, os
import numpy as np, cv2

# name -> (query_names, AI2-THOR GT classes)
CATALOGUE = {
    'lamp': (['lamp', 'desk lamp', 'floor lamp'], ['DeskLamp', 'FloorLamp']),
    'remote control': (['remote control'], ['RemoteControl']),
    'cup': (['cup', 'mug'], ['Cup', 'Mug']),
    'laptop': (['laptop'], ['Laptop']),
    'book': (['book'], ['Book']),
    'cell phone': (['cell phone'], ['CellPhone']),
    'alarm clock': (['alarm clock'], ['AlarmClock']),
    'vase': (['vase'], ['Vase']),
    'bowl': (['bowl'], ['Bowl']),
    'houseplant': (['houseplant', 'potted plant'], ['HousePlant']),
    'pillow': (['pillow'], ['Pillow']),
    'television': (['television', 'tv'], ['Television']),
    'teddy bear': (['teddy bear'], ['TeddyBear']),
    'watch': (['watch'], ['Watch']),
    'apple': (['apple'], ['Apple']),
    'spray bottle': (['spray bottle'], ['SprayBottle']),
    'statue': (['statue'], ['Statue']),
}
MIN_PX, N_PRESENT = 15, 3


def visible_frames(sd, gt):
    c = json.load(open(os.path.join(sd, 'camera.json'))); K = np.array(c['K']); W, H = c['width'], c['height']
    P = np.loadtxt(os.path.join(sd, 'traj.txt')).reshape(-1, 4, 4)
    vis = {g['id']: [] for g in gt}
    for idx, T_wc in enumerate(P):
        d = cv2.imread(os.path.join(sd, 'results', f'depth{idx:06d}.png'), -1)
        if d is None:
            continue
        d = d / 6553.5; T_cw = np.linalg.inv(T_wc)
        for g in gt:
            p = T_cw[:3, :3] @ np.asarray(g['position']) + T_cw[:3, 3]
            if p[2] < 0.2:
                continue
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            if not (0 <= u < W and 0 <= v < H):
                continue
            px = K[0, 0] * max(g['size']) / p[2]
            z = d[int(v), int(u)]
            if px >= MIN_PX and z > 0 and abs(z - p[2]) <= max(0.25, 0.6 * max(g['size'])):
                vis[g['id']].append(idx)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/exp1/procthor/sources'))
    ap.add_argument('--out', default=os.path.expanduser('~/exp1/search_eval'))
    ap.add_argument('--scenes', type=int, default=6)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    cands = []
    for sd in sorted(glob.glob(os.path.join(a.src, 'procthor_*'))):
        gt = [g for g in json.load(open(os.path.join(sd, 'gt_objects.json'))) if any(g['gt_class'] in v[1] for v in CATALOGUE.values())]
        vis = visible_frames(sd, gt)
        targets, absent = [], []
        for name, (q, cls) in CATALOGUE.items():
            inst = [g for g in gt if g['gt_class'] in cls]
            frames = sorted({f for g in inst for f in vis[g['id']]})
            if not inst:
                absent.append(name)
            elif frames:
                targets.append(dict(name=name, query_names=q, gt_classes=cls, present=True, visible_frames=frames,
                                    instances=len(inst), visible_instances=sum(bool(vis[g['id']]) for g in inst)))
        cands.append((len(targets), os.path.basename(sd), targets, absent))
    cands.sort(key=lambda c: -c[0])
    out = {}
    for k, (n, scene, targets, absent) in enumerate(cands[:a.scenes]):
        pick = [targets[i] for i in sorted(rng.choice(len(targets), min(N_PRESENT, len(targets)), replace=False))]
        miss = absent[rng.integers(len(absent))] if absent else None
        if miss:
            q, cls = CATALOGUE[miss]
            pick.append(dict(name=miss, query_names=q, gt_classes=cls, present=False, visible_frames=[], instances=0,
                             visible_instances=0))
        out[scene] = dict(command_id=k + 1, targets=pick, visible_target_categories=n)
        print(scene, [(t['name'], t['present'], len(t['visible_frames'])) for t in pick])
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(catalogue={k: dict(query_names=v[0], gt_classes=v[1]) for k, v in CATALOGUE.items()}, min_px=MIN_PX,
                   scenes=out), open(os.path.join(a.out, 'commands.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
