"""
Feasibility experiment: real CG-D detections (Replica room0) + simulated Mid-360 LiDAR
(raycast on room0 mesh, real T_cam_lidar extrinsic) -> semantic_geometry_fusion library.

Part A: per-detection geometry quality for each point-selection variant.
Part B: multi-view association with ObjectFeatureMap, old defaults vs new defaults.

Run with an env that has open3d + scipy + cv2 (e.g. conda `conceptgraph`):
  python tools/replica_feasibility/feas_exp.py [--gsa-dir DIR] [--per-label]
"""
import argparse, glob, gzip, itertools, json, os, pickle, sys
import numpy as np, cv2, open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '../../src/semantic_geometry_fusion'))
from plyload import load_ply_tris
from semantic_geometry_fusion.geometry import (
    points_inside_mask, transform_points, compute_geometry, project_lidar_to_image,
    depth_gate, erode_mask, voxel_downsample,
)
from semantic_geometry_fusion.feature_fusion import Observation3D, ObjectFeatureMap

REPLICA = os.path.expanduser('~/conceptgraph_ws/Replica/')
K = np.array([[600., 0, 599.5], [0, 600., 339.5], [0, 0, 1]])
W, H = 1200, 680
T_CAM_LIDAR = np.array([  # p_cam = T_CAM_LIDAR p_lidar (same as config/fusion_params.yaml)
    [-0.0019986, -0.9999089, -0.0133510, 0.0327582],
    [0.0034152, 0.0133441, -0.9999051, 0.3241458],
    [0.9999922, -0.0020440, 0.0033882, 0.0183492],
    [0, 0, 0, 1.]])
BG = {'wall', 'floor', 'ceiling'}
GATE = dict(k_mad=2.5, min_band=0.05)  # same as config defaults

rng = np.random.default_rng(0)
V, F = load_ply_tris(REPLICA + 'room0_mesh.ply')
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
POSES = np.loadtxt(REPLICA + 'room0/traj.txt').reshape(-1, 4, 4)  # camera-to-world, OpenCV


def mid360_scan(T_world_lidar, n_scans=1, noise=0.02):
    """Mid-360-like: 20k pts / 0.1 s, az 360 deg, el [-7, 52] deg, 2 cm range noise. Lidar-frame pts."""
    n = 20000 * n_scans
    az = rng.uniform(-np.pi, np.pi, n)
    el = np.radians(rng.uniform(-7, 52, n))
    d = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    dw = d @ T_world_lidar[:3, :3].T
    o = np.broadcast_to(T_world_lidar[:3, 3], dw.shape)
    t = scene.cast_rays(o3d.core.Tensor(np.c_[o, dw].astype(np.float32)))['t_hit'].numpy()
    ok = np.isfinite(t) & (t > 0.1) & (t < 40)
    r = t[ok] + rng.normal(0, noise, ok.sum())
    return d[ok] * r[:, None]


def dbscan_largest(pts, eps=0.10, minp=5):
    if len(pts) < minp:
        return pts[:0]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lab = np.asarray(pc.cluster_dbscan(eps, minp))
    if (lab >= 0).sum() == 0:
        return pts[:0]
    ids, c = np.unique(lab[lab >= 0], return_counts=True)
    return pts[lab == ids[np.argmax(c)]]


def bleed_ratio(pts_l, gt_depth, tol=0.05):
    """Fraction of selected points NOT on the camera-visible surface at their pixel (parallax occlusion)."""
    if len(pts_l) == 0:
        return np.nan
    uv, z, _ = project_lidar_to_image(pts_l, T_CAM_LIDAR, K)
    u = np.clip(np.rint(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(int), 0, H - 1)
    return float(np.mean(np.abs(z - gt_depth[v, u]) > tol))


def inside(p, m):
    return points_inside_mask(p, m, T_CAM_LIDAR, K)


FILTERS = {
    'raw': lambda p, m: inside(p, m),
    'erode5px': lambda p, m: inside(p, erode_mask(m, 5)),
    'depth_gate': lambda p, m: depth_gate(inside(p, m), T_CAM_LIDAR, **GATE),
    'erode+gate': lambda p, m: depth_gate(inside(p, erode_mask(m, 5)), T_CAM_LIDAR, **GATE),
    'dbscan': lambda p, m: dbscan_largest(inside(p, m)),
    'rgbd_dense': None,  # CG-style baseline: dense GT depth in mask
}


def build_observations(gsa_dir, n_scans, filt, conf_th=0.25, max_bbox=0.5):
    out, stats = [], []
    for fp in sorted(glob.glob(os.path.join(gsa_dir, '*.pkl.gz'))):
        idx = int(os.path.basename(fp)[5:11])
        g = pickle.load(gzip.open(fp, 'rb'))
        T_wc = POSES[idx]
        T_wl = T_wc @ T_CAM_LIDAR  # p_w = T_wc T_cl p_l
        pts_l = mid360_scan(T_wl, n_scans)
        gt = cv2.imread(REPLICA + f'room0/results/depth{idx:06d}.png', -1) / 6553.5
        fobs = []
        for i in range(len(g['xyxy'])):
            lab = g['classes'][g['class_id'][i]]
            conf = float(g['confidence'][i])
            x1, y1, x2, y2 = g['xyxy'][i]
            if conf < conf_th or lab in BG or (x2 - x1) * (y2 - y1) > max_bbox * W * H:
                continue
            m = g['mask'][i]
            vv, uu = np.nonzero(m & (gt > 0))
            zz = gt[vv, uu]
            pc = np.c_[(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz]
            ref_c = np.median(transform_points(pc, T_wc), axis=0) if len(pc) else None
            if filt == 'rgbd_dense':
                sel = transform_points(voxel_downsample(transform_points(pc, T_wc), 0.025),
                                       np.linalg.inv(T_wl)) if len(pc) else pc
            else:
                sel = FILTERS[filt](pts_l, m)
            st = dict(label=lab, n=len(sel), bleed=bleed_ratio(sel, gt), cerr=np.nan)
            stats.append(st)
            if len(sel) < 3:
                continue
            pw = transform_points(sel, T_wl)
            geo = compute_geometry(pw)
            if ref_c is not None:
                st['cerr'] = float(np.linalg.norm(geo.centroid - ref_c))
            f = g['image_feats'][i] / np.linalg.norm(g['image_feats'][i])
            fobs.append(Observation3D(i, lab, conf, pw, f, geo.centroid, geo.size, geo.yaw))
        out.append((idx, fobs))
    return out, stats


def run_assoc(frames_obs, delta, thr, voxel):
    fmap = ObjectFeatureMap(delta, thr, voxel_size=voxel)
    hist = {}
    for _, obs_list in frames_obs:
        for o in obs_list:
            obj, *_ = fmap.update(o)
            hist.setdefault(obj.object_id, []).append(o.label)
    objs = fmap.objects
    n_obs = np.array([o.num_observations for o in objs])
    purity = np.mean([max(map(hist[o.object_id].count, set(hist[o.object_id]))) / len(hist[o.object_id])
                      for o in objs])
    dup = sum(a.label == b.label and np.linalg.norm(a.centroid - b.centroid) < 0.3
              for a, b in itertools.combinations(objs, 2))
    return dict(objects=len(objs), multi_view=int((n_obs >= 2).sum()), single=int((n_obs == 1).sum()),
                max_obs=int(n_obs.max()), purity=round(float(purity), 3), dup_pairs=int(dup),
                total_obs=sum(len(x[1]) for x in frames_obs), max_pts=int(max(len(o.points_map) for o in objs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gsa-dir', default=REPLICA + 'room0/gsa_detections_ram_fusion_feas')
    ap.add_argument('--per-label', action='store_true')
    args = ap.parse_args()
    print(f"frames with CG-D detections: {len(glob.glob(os.path.join(args.gsa_dir, '*.pkl.gz')))}")

    print('\n=== Part A: per-detection geometry (non-BG, conf>=0.25) ===')
    cache = {}
    for n_scans in (1, 5):
        for filt in FILTERS:
            if filt == 'rgbd_dense' and n_scans > 1:
                continue
            obs, st = build_observations(args.gsa_dir, n_scans, filt)
            cache[(n_scans, filt)] = (obs, st)
            n = np.array([s['n'] for s in st]); b = np.array([s['bleed'] for s in st])
            ce = np.array([s['cerr'] for s in st])
            print(f'scans={n_scans} {filt:11s} dets={len(st):3d} usable={np.mean(n >= 3):.2f} '
                  f'median_pts={np.median(n):6.1f} mean_bleed={np.nanmean(b):.2f} | '
                  f'centroid_err median={np.nanmedian(ce):.3f}m p90={np.nanpercentile(ce, 90):.3f}m')

    if args.per_label:
        for key in [(1, 'raw'), (5, 'depth_gate')]:
            by = {}
            for s in cache[key][1]:
                by.setdefault(s['label'], []).append(s['n'])
            print(f'per-label median pts {key}:', {k: int(np.median(v)) for k, v in sorted(by.items())})

    print('\n=== Part B: multi-view association ===')
    runs = [
        ('RGB-D baseline (CG-style)', (1, 'rgbd_dense'), 0.025, 1.2, 0.0),
        ('OLD defaults: 1 scan, raw, d=2.5cm, no voxel', (1, 'raw'), 0.025, 1.2, 0.0),
        ('NEW defaults: 5 scans, depth_gate, d=5cm, voxel 5cm', (5, 'depth_gate'), 0.05, 1.2, 0.05),
        ('NEW but 1 scan (no accumulation)', (1, 'depth_gate'), 0.05, 1.2, 0.05),
        ('NEW + erode5px', (5, 'erode+gate'), 0.05, 1.2, 0.05),
        ('NEW, thr=1.4', (5, 'depth_gate'), 0.05, 1.4, 0.05),
    ]
    for name, key, delta, thr, vox in runs:
        print(f'{name:52s} -> {json.dumps(run_assoc(cache[key][0], delta, thr, vox))}')


if __name__ == '__main__':
    main()
