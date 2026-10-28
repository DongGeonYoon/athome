"""
Build the Perception -> Scene Graph feature dataset (v0.8) for every experiment source (Replica, ProcTHOR, Gazebo, ...).

Experiment input: <exp>/replica_rig/<scene>_rig/  (rendered Tidybot-rig trajectory, traj.txt,
CG-D detections gsa_detections_ram_<suffix>), see tools/replica_rig/render_rig.py and run_cgd_all.sh.

Per keyframe: CG-D semantics + simulated Mid-360 geometry (FAST-LIO /cloud_registered_body
emulation, T_cam_body, map <- body), then fusion with the over-merge fixes:
  1. per-frame duplicate-mask suppression (mask IoU >= 0.8, keep higher confidence)
  2. per-observation DBSCAN (eps 0.15 m, largest cluster; proposal 2-3) before fusion
  3. object-level cleanup before export (largest 0.20 m-connected component)
  (one-to-one frame matching and the bbox growth gate are implemented but off: see eval_variants)
Export per scene:
  objects.json        filtered final features (num_observations >= 2, generic tags removed)
  objects_all.json    every fused object (same schema)
  points/obj_<id>.npy fused map-frame points (float32, voxel 5 cm)
  occupancy/          2D occupancy (proposal 1 height-band BEV of an emulated LiDAR map) + pcd_map.pcd;
                      with --occupancy-map the Nav2 map_server map instead (same origin / extent as /map, copied as is)
  metadata.json       frame, units, sensors, CLIP, detector/fusion/filter params, quality, provenance

Usage:
  python build_dataset.py [--exp ~/exp1] [--suffix rigs2] [--scenes room0 room1 ...]
  python build_dataset.py --eval-variants room0      # compare fusion variants on one scene
  python build_dataset.py --exp ~/real/results/run4/offline5 --sources robot --out-name sg_nav2map \
      --occupancy-map ~/fast_lio_ws/maps/lab_map_20261005_174210_structural.yaml   # robot: Nav2 /map as occupancy
"""
import argparse, glob, gzip, hashlib, itertools, json, os, pickle, shutil, sys
import yaml
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../replica_rig'))
import rig_exp as R
from render_rig import load_scene, CAM_HEIGHT
from render_active_obs import free_cells
from semantic_geometry_fusion.geometry import (points_inside_mask, transform_points, compute_geometry, depth_gate,
                                               voxel_downsample, largest_cluster, depth_points_in_mask,
                                               remove_outliers, erode_mask)
from semantic_geometry_fusion.feature_fusion import (Observation3D, ObjectFeatureMap, dedup_detections,
                                                     denoise_object, apply_size_prior, merge_fragments,
                                                     trim_below_support)
from semantic_geometry_fusion.scene_graph_export import scene_graph_features

COND = 'fastlio_body'
# Chosen on room0 (eval_variants_room0*.txt): frame one-to-one matching fragmented objects (82 -> 146) and the
# growth gate had no effect, so both are off; dedup + object cleanup cut oversize objects 19 -> 2.
# split_study (5 scenes): per-observation DBSCAN eps 0.15 cut implausible small-class objects 17 -> 10 without
# extra fragmentation; post-fusion DBSCAN/HDBSCAN, co-visibility, observation graph, CLIP gate, label split did not help.
FUSION = dict(obs_cluster_eps=0.15, dedup_iou=0.8, frame_one_to_one=False, growth_ratio=None, growth_margin=0.3,
              denoise_eps=0.20, depth_fallback=True, size_prior_slack=1.5,
              merge_fragments_cos=0.9,   # export re-merge, study: exp1/merge_sweep + real run2
              # stray points before every bbox (require3 problem 1, run4 desk top 0.44 -> 0.33 m); None disables
              outliers=dict(k=8, std_ratio=2.0, min_gap=0.08, min_points=20),
              # small objects on furniture: drop points below the support surface (search-server check, 2026-10-28).
              # Off: no effect on run4 (LiDAR below the desk never sees the surface under small objects); mask erosion
              # did the job there. --support-trim turns it on.
              support_trim=None,
              # 2D mask edge erosion [px] before LiDAR points, per source. Real robot only (require4 2): run4 small
              # objects on desks 1/6 -> 5/6 "on" the desk; simulation (perfect camera-LiDAR alignment) got worse
              # (ProcTHOR objects -8 %). To be re-checked once camera depth is recorded.
              mask_erosion_px=dict(robot=3))
FILTER = dict(min_observations=2, excluded_tags=['other item', 'furniture', 'pad', 'object', 'item', 'thing',
                                                # building parts RAM tags on walls / floors (ProcTHOR study)
                                                'tile wall', 'tile', 'wall', 'floor', 'ceiling', 'pillar', 'pole',
                                                'lamp post', 'balustrade', 'tile floor', 'wood floor'])
STRUCTURAL = {'window', 'curtain', 'blind', 'door', 'wall', 'floor', 'ceiling', 'shade', 'drape'}
OCC_RES, OCC_BAND, N_OCC = 0.05, (0.05, 0.95), 2
OBJ_KEYS = {"object_id", "semantic_tag", "confidence", "centroid", "bbox_3d", "clip_feature", "num_observations"}
SCENE_NOTES = {
    'room2': 'dense furniture: only 25 robot-free cells, many close-up views of chair backs',
    'office1': 'mesh vertex colours look purple / washed out with missing regions; low image quality',
    'office4': 'many close-up views of chair backs',
}


def camera_of(scene_dir):
    """(K, W, H) of a scene: camera.json (common intermediate format) or the Replica rig default."""
    cj = os.path.join(scene_dir, 'camera.json')
    if os.path.exists(cj):
        c = json.load(open(cj))
        return np.array(c['K'], dtype=np.float64), int(c['width']), int(c['height'])
    return R.K, R.W, R.H


def sensor_data(scene_dir, idx, T_wc):
    """(points in sensor frame, T_map_sensor, T_cam_sensor) for one keyframe: stored LiDAR or mesh simulation."""
    f = os.path.join(scene_dir, 'lidar', f'frame{idx:06d}.npz')
    if os.path.exists(f):
        d = np.load(f)
        return d['points'].astype(np.float64), d['T_map_sensor'], d['T_cam_sensor']
    T_wl = T_wc @ R.T_CAM_LIDAR
    return R.sensor_cloud(R.mid360_scans(T_wl, R.N_SCANS), T_wl, COND)


LAST_STATS = {}


def load_depth(scene_dir, idx):
    """Aligned camera depth in metres, or None when the source has no depth (Gazebo bags record colour only)."""
    d = cv2.imread(os.path.join(scene_dir, 'results', f'depth{idx:06d}.png'), -1)
    return d / 6553.5 if d is not None and d.max() > 0 else None


def frame_observations(scene_dir, gsa_dir, dedup, obs_cluster_eps=None, erosion=0):
    """Per keyframe list of Observation3D. Geometry = LiDAR points (FAST-LIO body cloud or its emulation) in the
    mask; when fewer than 3 survive and FUSION['depth_fallback'], camera depth in the mask is used instead."""
    P = np.loadtxt(os.path.join(scene_dir, 'traj.txt')).reshape(-1, 4, 4)
    K, W, H = camera_of(scene_dir)
    frames, n_det, n_dup, n_depth = [], 0, 0, 0
    for fp in sorted(glob.glob(os.path.join(gsa_dir, '*.pkl.gz'))):
        idx = int(os.path.basename(fp)[5:11])
        g = pickle.load(gzip.open(fp, 'rb'))
        pts_s, T_ws, T_cs = sensor_data(scene_dir, idx, P[idx])
        cand = []
        for i in range(len(g['xyxy'])):
            lab = g['classes'][g['class_id'][i]]; conf = float(g['confidence'][i])
            x1, y1, x2, y2 = g['xyxy'][i]
            if conf < R.CONF_TH or lab in R.BG or (x2 - x1) * (y2 - y1) > R.MAX_BBOX * W * H:
                continue
            cand.append(i)
        n_det += len(cand)
        keep = cand
        if dedup and cand:
            k = dedup_detections([g['mask'][i] for i in cand], [g['confidence'][i] for i in cand], FUSION['dedup_iou'])
            keep = [cand[j] for j in k]
            n_dup += len(cand) - len(keep)
        obs, depth = [], False      # depth image loaded only when a fallback is needed
        for i in keep:
            sel = depth_gate(points_inside_mask(pts_s, erode_mask(g['mask'][i], erosion), T_cs, K),
                             T_cs, **R.GATE)
            if obs_cluster_eps:
                sel = largest_cluster(sel, obs_cluster_eps)
            if len(sel) < 3 and FUSION['depth_fallback']:
                if depth is False:
                    depth = load_depth(scene_dir, idx)
                if depth is not None:
                    sel = depth_gate(depth_points_in_mask(depth, g['mask'][i], K, T_cs), T_cs, **R.GATE)
                    if obs_cluster_eps:
                        sel = largest_cluster(voxel_downsample(sel, R.VOXEL), obs_cluster_eps)
                    n_depth += len(sel) >= 3
            if len(sel) < 3:
                continue
            if FUSION['outliers']:
                sel = remove_outliers(sel, **FUSION['outliers'])
            pw = transform_points(sel, T_ws); geo = compute_geometry(pw)
            feat = g['image_feats'][i] / np.linalg.norm(g['image_feats'][i])
            obs.append(Observation3D(i, g['classes'][g['class_id'][i]], float(g['confidence'][i]), pw, feat,
                                     geo.centroid, geo.size, geo.yaw, geo.bbox_center))
        frames.append((idx, obs))
    LAST_STATS['observations_depth_fallback'] = n_depth
    return frames, n_det, n_dup


def fuse(frames, one_to_one, growth_ratio, denoise_eps):
    fmap = ObjectFeatureMap(R.NN_DELTA, R.ASSOC_THR, voxel_size=R.VOXEL, growth_ratio=growth_ratio,
                            growth_margin=FUSION['growth_margin'])
    for _, obs in frames:
        if one_to_one:
            fmap.update_frame(obs)
        else:
            for o in obs:
                fmap.update(o)
    objs = fmap.objects
    if denoise_eps:
        objs = [denoise_object(o, denoise_eps, outliers=FUSION['outliers']) for o in objs]
    if FUSION['merge_fragments_cos']:          # after denoise: a merged front + back of one object may be disconnected
        objs = merge_fragments(objs, R.VOXEL, min_cos=FUSION['merge_fragments_cos'])
    if FUSION['support_trim']:
        objs = trim_below_support(objs, **FUSION['support_trim'])
    if FUSION['size_prior_slack']:
        objs = [apply_size_prior(o, slack=FUSION['size_prior_slack']) for o in objs]
    return objs


def quality(objs):
    if not objs:
        return {}
    sz = np.array([o.size for o in objs]); longs = sz[:, :2].max(1); shorts = sz[:, :2].min(1)
    giants = [o for o, l in zip(objs, longs) if l > 2.0 and o.label not in STRUCTURAL]
    dup = sum(a.label == b.label and np.linalg.norm(a.centroid - b.centroid) < 0.3 for a, b in itertools.combinations(objs, 2))
    small = ('chair', 'stool', 'pillow', 'cushion', 'lamp', 'vase', 'monitor', 'plant', 'flower', 'sculpture', 'box',
             'basket', 'bottle', 'cup', 'book', 'clock', 'frame', 'footrest', 'trash', 'bin', 'hassock', 'figurine')
    size_mismatch = [o.object_id for o, l in zip(objs, longs)
                     if any(k in o.label for k in small) and l > (1.3 if o.label == 'armchair' else 1.5)]
    return dict(objects=len(objs), multi_view=int(sum(o.num_observations >= 2 for o in objs)),
                tag_size_mismatch_ids=size_mismatch,
                single_view=int(sum(o.num_observations == 1 for o in objs)),
                oversize_non_structural=len(giants), oversize_examples=[(o.label, [round(v, 2) for v in o.size]) for o in giants[:5]],
                same_tag_pairs_within_0_3m=int(dup), bbox_short_side_median_m=round(float(np.median(shorts)), 3),
                thin_bbox_ratio=round(float(np.mean(shorts < 0.15)), 3),
                label_purity=round(float(np.mean([max(o.label_counts.values()) / sum(o.label_counts.values()) for o in objs])), 3))


def keep_for_scene_graph(o):
    return o.num_observations >= FILTER['min_observations'] and o.label not in FILTER['excluded_tags']


def build_occupancy(scene, floor):
    """Emulated offline mapping run (Mid-360 scans from robot-free cells), then proposal-1 BEV occupancy."""
    pts, org, n_pos = emulated_map(scene, floor)
    occ, lo = occupancy_from_points(pts, org, floor)
    return occ, lo, pts, n_pos


def emulated_map(scene, floor):
    sc, _, _ = load_scene(scene)
    P_ref = np.loadtxt(os.path.join(R.REPLICA, scene, 'traj.txt')).reshape(-1, 4, 4)
    free = free_cells(sc, floor, P_ref)
    chosen = {}
    for xy in free:
        chosen.setdefault(tuple(np.floor(xy / 0.6).astype(int)), xy)
    pts, org = [], []
    for xy in chosen.values():
        T_wc = np.eye(4); fwd = np.array([1.0, 0, 0]); down = np.array([0, 0, -1.0])
        T_wc[:3, :3] = np.c_[np.cross(down, fwd), down, fwd]; T_wc[:3, 3] = [xy[0], xy[1], floor + CAM_HEIGHT]
        T_wl = T_wc @ R.T_CAM_LIDAR
        p = transform_points(np.concatenate(R.mid360_scans(T_wl, 2)), T_wl)
        pts.append(p); org.append(np.repeat(T_wl[None, :3, 3], len(p), axis=0))
    return np.concatenate(pts), np.concatenate(org), len(chosen)


def occupancy_from_points(pts, org, floor):
    """Proposal 1: free = sensor-to-hit rays, occupied = >= N_OCC points in the obstacle height band."""
    lo = pts[:, :2].min(0) - 0.5; hi = pts[:, :2].max(0) + 0.5
    nx, ny = np.ceil((hi - lo) / OCC_RES).astype(int)
    occ = np.full((ny, nx), -1, np.int8)
    sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 150000), replace=False)
    o, p = org[sel, :2], pts[sel, :2]
    steps = np.maximum(1, (np.linalg.norm(p - o, axis=1) / (OCC_RES * 0.5)).astype(int))
    for s in np.unique(steps):
        m = steps == s
        q = o[m][:, None, :] + (np.arange(s) / s)[None, :, None] * (p[m] - o[m])[:, None, :]
        ij = np.floor((q.reshape(-1, 2) - lo) / OCC_RES).astype(int)
        occ[ij[:, 1], ij[:, 0]] = 0
    h = pts[:, 2] - floor
    ij = np.floor((pts[(h > OCC_BAND[0]) & (h < OCC_BAND[1]), :2] - lo) / OCC_RES).astype(int)
    cnt = np.zeros((ny, nx), np.int32); np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1)
    occ[cnt >= N_OCC] = 100
    return occ, lo


def load_nav2_map(yaml_path):
    """Nav2 map_server map (trinary, negate 0) -> (occ int8 100/0/-1 with row 0 = y_min, origin xy, resolution, info)."""
    with open(yaml_path) as f:
        m = yaml.safe_load(f)
    if m.get('mode', 'trinary') != 'trinary' or int(m.get('negate', 0)) != 0:
        raise ValueError(f'{yaml_path}: only trinary, negate 0 maps are supported')
    img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), m['image'])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'cannot read {img_path}')
    p = (255.0 - img) / 255.0                  # map_server trinary rule
    occ = np.full(img.shape, -1, np.int8)
    occ[p > float(m['occupied_thresh'])] = 100
    occ[p < float(m['free_thresh'])] = 0
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    info = dict(source='nav2_map_server', yaml=os.path.basename(yaml_path), image=os.path.basename(img_path),
                sha256={os.path.basename(yaml_path): sha(yaml_path), os.path.basename(img_path): sha(img_path)},
                # search server's map check: "sha256:" + first 16 hex of sha256(yaml bytes + image bytes)
                map_version='sha256:' + hashlib.sha256(open(yaml_path, 'rb').read() + open(img_path, 'rb').read()).hexdigest()[:16],
                origin=[float(v) for v in m['origin']], size=[int(img.shape[1]), int(img.shape[0])],
                note='same file as the robot map_server /map (copied unchanged next to map.yaml); '
                     'unobserved cells are unknown')
    return occ[::-1].copy(), np.asarray(m['origin'][:2], np.float64), float(m['resolution']), info, (yaml_path, img_path), m


def save_occupancy(d, occ, lo, pts, res=OCC_RES, thresh=(0.65, 0.25), origin_fmt='.4f'):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'occupancy.npy'), occ)
    img = np.full(occ.shape, 205, np.uint8); img[occ == 0] = 254; img[occ == 100] = 0
    cv2.imwrite(os.path.join(d, 'map.pgm'), img[::-1])
    with open(os.path.join(d, 'map.yaml'), 'w') as f:
        f.write(f'image: map.pgm\nmode: trinary\nresolution: {res}\norigin: [{lo[0]:{origin_fmt}}, {lo[1]:{origin_fmt}}, 0.0]\n'
                f'negate: 0\noccupied_thresh: {thresh[0]}\nfree_thresh: {thresh[1]}\n')
    v = voxel_downsample(pts, 0.05).astype(np.float32)
    with open(os.path.join(d, 'pcd_map.pcd'), 'w') as f:
        f.write(f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(v)}\n'
                f'HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(v)}\nDATA ascii\n')
        np.savetxt(f, v, fmt='%.4f')


def check_format(sg):
    assert sg['frame_id'] == 'map' and set(sg) == {'frame_id', 'objects'}
    ids = [o['object_id'] for o in sg['objects']]
    assert len(ids) == len(set(ids))
    for o in sg['objects']:
        assert set(o) == OBJ_KEYS and len(o['clip_feature']) == 1024
        assert abs(np.linalg.norm(o['clip_feature']) - 1) < 1e-3 and min(o['bbox_3d']['size']) >= 0


def eval_variants(args, scene):
    R.set_scene(scene)
    sd = os.path.join(args.exp, 'replica_rig', f'{scene}_rig'); gd = os.path.join(sd, f'gsa_detections_ram_{args.suffix}')
    raw, _, _ = frame_observations(sd, gd, dedup=False)
    dd, _, n_dup = frame_observations(sd, gd, dedup=True)
    variants = [
        ('baseline (v0.2)', raw, False, None, 0),
        ('+dedup', dd, False, None, 0),
        ('+dedup +denoise0.12', dd, False, None, 0.12),
        ('+dedup +denoise0.20', dd, False, None, 0.20),
        ('+dedup +denoise0.30', dd, False, None, 0.30),
        ('+dedup +growth1.5 +denoise0.20', dd, False, 1.5, 0.20),
    ]
    print(f'{scene}: keyframes={len(raw)} duplicate masks removed={n_dup}')
    res = {}
    for name, frames, oto, gr, eps in variants:
        q = quality(fuse(frames, oto, gr, eps)); res[name] = q
        print(f'{name:36s}', json.dumps({k: v for k, v in q.items() if k != 'oversize_examples'}), q['oversize_examples'][:3])
    return res


SOURCES = {
    # name: (root relative to --exp, scene dir glob, CG-D detection suffix, description)
    'replica': ('replica_rig', '*_rig', 'v2', 'Replica NICE-SLAM release (single room), mesh-rendered rig, Mid-360 ray casting'),
    'replica_full': ('replica_full/sources', '*', 'v2', 'Replica full release (multi-room apartments), mesh-rendered rig'),
    'procthor': ('procthor/sources', 'procthor_*', 'v2', 'ProcTHOR-10k houses (AI2-THOR), rig cameras, Mid-360 from depth'),
    'gazebo': ('gazebo/sources', '*', 'v2', 'Gazebo AWS small house, ROS2 chain: Mid-360 plugin -> FAST-LIO -> TF'),
    'robot': ('robot/sources', '*', 'v2', 'Real Tidybot pre-mapping run (tools/sim_sources/robot_bag_extract.py)'),
}
FAMILIES = [
    {'sofa', 'couch', 'loveseat', 'settee'}, {'chair', 'armchair', 'stool', 'seat', 'bench'},
    {'table', 'desk', 'countertop', 'counter'}, {'bed', 'mattress'}, {'pillow', 'cushion'},
    {'painting', 'picture', 'frame', 'portrait', 'poster', 'photo', 'art', 'artwork'}, {'television', 'tv', 'monitor', 'screen'},
    {'garbagecan', 'garbage', 'trash', 'bin', 'wastebasket'}, {'refrigerator', 'fridge'},
    {'shelf', 'shelving', 'bookshelf', 'bookcase'}, {'drawer', 'dresser', 'cabinet', 'nightstand', 'sidetable', 'wardrobe', 'closet'},
    {'lamp', 'light', 'chandelier'}, {'plant', 'houseplant', 'flower', 'vase'}, {'laptop', 'computer', 'notebook'},
    {'cellphone', 'phone', 'smartphone'}, {'remotecontrol', 'remote'}, {'cup', 'mug', 'glass'}, {'bowl', 'plate', 'dish'},
    {'statue', 'sculpture', 'figurine'}, {'cabinet', 'countertop', 'counter', 'shelvingunit', 'shelf'},
    {'dumbbell', 'barbell', 'weight'}, {'rug', 'carpet', 'mat'},
    {'door', 'doorway'}, {'window', 'curtain', 'blind'}, {'sink', 'basin'}, {'toilet'}, {'bathtub', 'tub'},
]


def _tokens(name):
    import re
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower().replace('_', ' ').split()
    return set(words) | {''.join(words)}


def compatible(tag, gt_class):
    a, b = _tokens(tag), _tokens(gt_class)
    if a & b:
        return True
    return any(a & f and b & f for f in FAMILIES)


def gt_evaluation(objs, gt, radius=0.6, small=0.3):
    """Match exported objects to ground-truth objects (XY distance) - perception-side quality only."""
    if not gt:
        return None
    G = np.array([g['position'][:2] for g in gt]); sizes = [max(g.get('size') or [0.0]) for g in gt]
    det = [g for g, sz in zip(range(len(gt)), sizes) if sz == 0.0 or sz >= 0.05]    # ignore crumbs < 5 cm
    res = dict(radius_m=radius, gt_objects=len(det), exported=len(objs))
    if not objs:
        return res
    C = np.array([o.bbox_center[:2] for o in objs])
    D = np.linalg.norm(C[:, None, :] - G[None, :, :], axis=2)
    near_any = (D[:, det] <= radius).any(1)
    label_ok = np.array([any(D[i, j] <= radius and compatible(o.label, gt[j]['gt_class']) for j in det) for i, o in enumerate(objs)])
    found = np.array([any(D[i, j] <= radius and compatible(objs[i].label, gt[j]['gt_class']) for i in range(len(objs))) for j in det])
    found_geo = (D[:, det] <= radius).any(0)
    small_m = np.array([0.0 < sizes[j] < small for j in det])
    # parts of furniture (drawer of a dresser) and objects resting on / inside another object are not merge evidence
    PARTS = {'drawer', 'shelf', 'cabinet', 'countertop', 'sinkbasin', 'stoveburner', 'stoveknob', 'lightswitch', 'showerhead'}
    standalone = np.array([(not gt[j].get('parent') or str(gt[j]['parent']).startswith('Floor'))
                           and _tokens(gt[j]['gt_class']).isdisjoint(PARTS) for j in det])
    merged = 0
    for i, o in enumerate(objs):                       # one object covering >= 2 large standalone GT objects
        Rz = np.array([[np.cos(o.yaw), np.sin(o.yaw)], [-np.sin(o.yaw), np.cos(o.yaw)]])
        loc = (G[det] - o.bbox_center[:2]) @ Rz.T
        inside = (np.abs(loc) <= np.array(o.size[:2]) / 2 + 0.05).all(1) & ~small_m & standalone
        merged += int(inside.sum() >= 2)
    # ProcTHOR / Gazebo GT has no doors, windows or curtains: precision over non-structural tags as well
    nonstruct = np.array([_tokens(o.label).isdisjoint(STRUCTURAL) for o in objs])
    res.update(precision_geometric=round(float(near_any.mean()), 3), precision_semantic=round(float(label_ok.mean()), 3),
               precision_semantic_nonstructural=round(float(label_ok[nonstruct].mean()), 3) if nonstruct.any() else None,
               recall_semantic=round(float(found.mean()), 3), recall_geometric=round(float(found_geo.mean()), 3),
               recall_semantic_small=round(float(found[small_m].mean()), 3) if small_m.any() else None,
               recall_semantic_large=round(float(found[~small_m].mean()), 3) if (~small_m).any() else None,
               objects_covering_2plus_gt=merged,
               missed_classes=sorted({gt[j]['gt_class'] for j, f in zip(det, found) if not f})[:30])
    return res


def build_scene(args, source, scene_dir, sid):
    sd_in = scene_dir
    root, _, suffix, desc = SOURCES[source]
    suffix = args.det_suffix or suffix
    gd = os.path.join(sd_in, f'gsa_detections_ram_{suffix}')
    out = os.path.join(args.exp, args.out_name)
    sd = os.path.join(out, 'scenes', sid)
    shutil.rmtree(sd, ignore_errors=True); os.makedirs(os.path.join(sd, 'points'))
    src = json.load(open(os.path.join(sd_in, 'source.json'))) if os.path.exists(os.path.join(sd_in, 'source.json')) else {}
    if source == 'replica':
        R.set_scene(os.path.basename(sd_in)[:-4])

    frames, n_det, n_dup = frame_observations(sd_in, gd, dedup=True, obs_cluster_eps=FUSION['obs_cluster_eps'],
                                              erosion=FUSION['mask_erosion_px'].get(source, 0))
    objs = fuse(frames, FUSION['frame_one_to_one'], FUSION['growth_ratio'], FUSION['denoise_eps'])
    exported = [o for o in objs if keep_for_scene_graph(o)]
    sg_all = scene_graph_features(objs, 'map'); check_format(sg_all)
    sg = scene_graph_features(exported, 'map'); check_format(sg)
    json.dump(sg_all, open(os.path.join(sd, 'objects_all.json'), 'w'), indent=2)
    json.dump(sg, open(os.path.join(sd, 'objects.json'), 'w'), indent=2)
    for o in objs:
        np.save(os.path.join(sd, 'points', f'obj_{o.object_id:04d}.npy'), o.points_map.astype(np.float32))

    if 'floor_z' in src:
        floor = float(src['floor_z'])
    else:
        floor = float(np.loadtxt(os.path.join(sd_in, 'traj.txt')).reshape(-1, 4, 4)[0, 2, 3]) - CAM_HEIGHT
    mpf = os.path.join(sd_in, 'map_points.npz')
    occ_info, occ_res = None, OCC_RES
    if os.path.exists(mpf):
        m = np.load(mpf); pcd, org, n_pos = m['points'].astype(np.float64), m['origins'].astype(np.float64), None
        occ, lo = occupancy_from_points(pcd, org, floor)
    else:
        occ, lo, pcd, n_pos = build_occupancy(os.path.basename(sd_in)[:-4], floor)
    if args.occupancy_map and source == 'robot':
        occ, lo, occ_res, occ_info, files, m = load_nav2_map(args.occupancy_map)
        save_occupancy(os.path.join(sd, 'occupancy'), occ, lo, pcd, occ_res, (m['occupied_thresh'], m['free_thresh']),
                       origin_fmt='.17g')   # exactly the Nav2 origin
        for f in files:
            shutil.copy(f, os.path.join(sd, 'occupancy', os.path.basename(f)))
    else:
        save_occupancy(os.path.join(sd, 'occupancy'), occ, lo, pcd)

    gtf = os.path.join(sd_in, 'gt_objects.json')
    gt_eval = gt_evaluation(exported, json.load(open(gtf))) if os.path.exists(gtf) else None
    K, W, H = camera_of(sd_in)
    n_obs = sum(len(o) for _, o in frames)
    counts = dict(keyframes=len(frames), detections_after_filter=n_det, duplicate_masks_removed=n_dup,
                  observations_with_geometry=n_obs, observations_depth_fallback=LAST_STATS.get('observations_depth_fallback', 0),
                  objects_all=len(sg_all['objects']), objects=len(sg['objects']))
    md = dict(
        scene_id=sid, source=source, source_description=desc, source_details=src, frame_id='map',
        units=dict(length='m', angle='rad'), floor_z=round(floor, 4),
        map_frame_note='Scene map frame (z up). On the robot: FAST_LIO_LOCALIZATION2 map frame (floor z ~ -0.40).',
        scene_note=SCENE_NOTES.get(os.path.basename(sd_in)[:-4], '') if source == 'replica' else '',
        camera=dict(model='RealSense D435i color (rig geometry)', width=W, height=H, K=np.round(K, 6).tolist(),
                    height_above_floor=CAM_HEIGHT),
        lidar=dict(model='Livox Mid-360', input=src.get('lidar_input', 'FAST-LIO /cloud_registered_body emulation'),
                   accumulated_scans=R.N_SCANS, T_cam_lidar=R.T_CAM_LIDAR.round(9).tolist(),
                   T_body_lidar=R.T_BODY_LIDAR.tolist(), T_cam_body=R.T_CAM_BODY.round(9).tolist(), height_above_floor=0.445),
        pose_source=src.get('pose_source', 'ground truth (rendered)'),
        clip=dict(model='open_clip ViT-H-14', pretrained='laion2b_s32b_b79k', dim=1024, normalized='L2',
                  input='CG-D padded bbox image crop', inference='fp16 on GPU (cosine to fp32 CPU run: 1.0000)'),
        detector=dict(pipeline='CG-D generate_gsa_results.py via tools/replica_rig/gsa_fast.py (RAM CPU, GDINO/SAM/CLIP GPU fp16, '
                               'GroundingDINO phrase->class fallback fixed)',
                      box_threshold=0.2, text_threshold=0.2, add_bg_classes=True, accumu_classes=False, PYTHONHASHSEED=0,
                      vocabulary=dict(per_frame='RAM tags', minus='tools/replica_rig/vocab/blocklist.txt (scenery / picture '
                                      'content / abstract / people)', plus='tools/replica_rig/vocab/household.txt (46 generic '
                                      'household objects)', chosen_by='ProcTHOR vocabulary study (README section 4)'),
                      excluded_labels=sorted(R.BG), min_confidence=R.CONF_TH, max_bbox_area_ratio=R.MAX_BBOX),
        fusion=dict(nn_distance_threshold=R.NN_DELTA, association_threshold=R.ASSOC_THR, voxel_size=R.VOXEL,
                    depth_gate=R.GATE, min_lidar_points=3, **FUSION,
                    depth_fallback_note='camera depth inside the mask only when < 3 LiDAR points survive (count in quality)',
                    size_prior_note='small-object labels on oversized bboxes replaced (feature_fusion.SIZE_PRIOR_MAX)',
                    semantic_tag='majority vote of fused labels', confidence='mean detection confidence',
                    clip_feature='observation-count weighted mean, L2 re-normalised'),
        export_filter=dict(file='objects.json', **FILTER, note='objects_all.json has every fused object'),
        occupancy=dict(resolution=occ_res, obstacle_band_above_floor=OCC_BAND, n_occ=N_OCC, mapping_positions=n_pos,
                       values={'occupied': 100, 'free': 0, 'unknown': -1}, image_origin='map.pgm row 0 = y_max',
                       **({'map_version': occ_info} if occ_info else {})),
        quality_all=quality(objs), quality_exported=quality(exported), gt_evaluation=gt_eval,
        counts=counts, keyframes=[f for f, _ in frames],
        provenance=dict(experiment=sd_in, build_script='semantic_geometry_ws/tools/scene_graph_dataset/build_dataset.py'))
    json.dump(md, open(os.path.join(sd, 'metadata.json'), 'w'), indent=2)
    for extra in ('gt_objects.json', 'gt_rooms.json'):
        if os.path.exists(os.path.join(sd_in, extra)):
            os.makedirs(os.path.join(sd, 'ground_truth'), exist_ok=True)
            shutil.copy(os.path.join(sd_in, extra), os.path.join(sd, 'ground_truth', extra))
    print(sid, json.dumps(counts), json.dumps({k: v for k, v in (gt_eval or {}).items() if k != 'missed_classes'}), flush=True)
    return dict(scene_id=sid, source=source, path=f'scenes/{sid}', note=md['scene_note'], gt_evaluation=gt_eval, **counts)


def discover(args):
    found = []
    for source in args.sources:
        root, pat, suffix, _ = SOURCES[source]
        suffix = args.det_suffix or suffix
        for d in sorted(glob.glob(os.path.join(args.exp, root, pat))):
            if not glob.glob(os.path.join(d, f'gsa_detections_ram_{suffix}', '*.pkl.gz')):
                continue
            name = os.path.basename(d)
            sid = f'replica_{name}' if source == 'replica' else (name if name.startswith(source) else f'{source}_{name}')
            if args.scenes and sid not in args.scenes:
                continue
            found.append((source, d, sid))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=os.path.expanduser('~/exp1'))
    ap.add_argument('--suffix', default='rigs2', help='CG-D suffix for --eval-variants (replica)')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--scenes', nargs='*', default=None, help='scene ids to (re)build; others are kept')
    ap.add_argument('--eval-variants', default=None, metavar='SCENE')
    ap.add_argument('--no-depth-fallback', action='store_true', help='LiDAR-only geometry')
    ap.add_argument('--assoc-thr', type=float, default=None, help='override the association threshold (phi_sem + phi_geo)')
    ap.add_argument('--merge-cos', type=float, default=None, help='export-time fragment merge CLIP cosine (0 = off)')
    ap.add_argument('--growth-ratio', type=float, default=None, help='association growth gate (0 = off)')
    ap.add_argument('--det-suffix', default=None, help='override the CG-D detection suffix of every source')
    ap.add_argument('--no-outliers', action='store_true', help='skip stray-point removal (before require3, for comparison)')
    ap.add_argument('--support-trim', action='store_true', help='trim small objects below furniture surfaces (off by default)')
    ap.add_argument('--mask-erosion', type=int, default=None, help='2D mask edge erosion [px] before LiDAR points')
    ap.add_argument('--out-name', default='scene_graph_dataset', help='output folder under --exp')
    ap.add_argument('--occupancy-map', default=None, metavar='YAML',
                    help='robot scenes: use this Nav2 map_server map (the /map the robot navigates on) as occupancy/')
    args = ap.parse_args()
    FUSION["depth_fallback"] = not args.no_depth_fallback
    if args.assoc_thr is not None:
        R.ASSOC_THR = args.assoc_thr
    if args.merge_cos is not None:
        FUSION['merge_fragments_cos'] = args.merge_cos
    if args.no_outliers:
        FUSION['outliers'] = None
    if args.support_trim:
        FUSION['support_trim'] = dict(max_dist=0.25, margin=0.025, max_size=0.6)
    if args.mask_erosion is not None:
        FUSION['mask_erosion_px'] = {s: args.mask_erosion for s in SOURCES}
    if args.growth_ratio is not None:
        FUSION['growth_ratio'] = args.growth_ratio or None
    if args.eval_variants:
        eval_variants(args, args.eval_variants); return
    out = os.path.join(args.exp, args.out_name)
    os.makedirs(os.path.join(out, 'schema'), exist_ok=True)
    shutil.copy(os.path.join(HERE, 'objects.schema.json'), os.path.join(out, 'schema', 'objects.schema.json'))
    built = {e['scene_id']: e for e in (build_scene(args, src, d, sid) for src, d, sid in discover(args))}
    idx_path = os.path.join(out, 'index.json')
    index = {e['scene_id']: e for e in (json.load(open(idx_path))['scenes'] if os.path.exists(idx_path) else [])
             if 'source' in e and os.path.isdir(os.path.join(out, e['path']))}
    index.update(built)
    scenes = sorted(index.values(), key=lambda e: (e['source'], e['scene_id']))
    tot = {k: sum(e[k] for e in scenes) for k in ('keyframes', 'observations_with_geometry', 'objects_all', 'objects')}
    by_src = {s: sum(1 for e in scenes if e['source'] == s) for s in SOURCES}
    json.dump(dict(dataset='perception_to_scene_graph_features', version='0.8', totals=tot, scenes_per_source=by_src,
                   scenes=scenes), open(idx_path, 'w'), indent=2)
    print('TOTAL', json.dumps(tot), json.dumps(by_src))


if __name__ == '__main__':
    main()
