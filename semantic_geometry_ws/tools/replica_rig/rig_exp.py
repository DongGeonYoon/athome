"""
Replica room0 rendered from the real Tidybot rig + simulated Mid-360 -> semantic/geometric
feature fusion -> Scene Graph Object-Centric Feature JSON.

Conditions (same fusion code / parameters as geometry_fusion_node defaults):
  raw_livox     : raw Mid-360 scans in livox_frame, projected with T_cam_lidar, map <- lidar pose
  fastlio_body  : FAST-LIO /cloud_registered_body emulation (blind 0.2 m, point_filter_num 3,
                  IMU body frame), projected with T_cam_body, map <- body pose (costmap TF chain)
  rgbd_baseline : dense GT depth inside each mask (ConceptGraphs-style geometry), reference only

Outputs under --out (default ~/exp1):
  results/scene_graph_features_<cond>.json   (Scene Graph schema)
  results/metrics.json
  images/bev_objects_<cond>.png, images/keyframe_overlay_*.png
  videos/room0_rig_lidar_overlay.mp4, videos/room0_rig_detections.mp4
"""
import argparse, glob, gzip, itertools, json, os, pickle, sys
import numpy as np, cv2, open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../replica_feasibility'))
sys.path.insert(0, os.path.join(HERE, '../../src/semantic_geometry_fusion'))
from plyload import load_ply_tris
from semantic_geometry_fusion.geometry import (
    points_inside_mask, transform_points, compute_geometry, project_lidar_to_image, depth_gate, voxel_downsample)
from semantic_geometry_fusion.feature_fusion import Observation3D, ObjectFeatureMap
from semantic_geometry_fusion.scene_graph_export import save_scene_graph_json

REPLICA = os.path.expanduser('~/conceptgraph_ws/Replica/')
# Real robot parameters (Camera_pose.md, calib.json, fast_lio_ws mid360.yaml)
K = np.array([[602.312561, 0, 329.529297], [0, 602.256470, 240.361496], [0, 0, 1]])
W, H = 640, 480
T_CAM_LIDAR = np.array([
    [-0.001998602, -0.999908874, -0.013350974, 0.032758183],
    [0.003415168, 0.013344098, -0.999905131, 0.324145816],
    [0.999992171, -0.002044008, 0.003388188, 0.018349150],
    [0, 0, 0, 1.]])
T_BODY_LIDAR = np.eye(4); T_BODY_LIDAR[:3, 3] = [-0.011, -0.02329, 0.04412]   # FAST-LIO extrinsic_T, R = I
T_CAM_BODY = T_CAM_LIDAR @ np.linalg.inv(T_BODY_LIDAR)
# Fusion parameters = config/fusion_params.yaml defaults
NN_DELTA, ASSOC_THR, VOXEL = 0.05, 1.1, 0.05   # 1.1: association threshold study (exp1/assoc_sweep)
GATE = dict(k_mad=2.5, min_band=0.05)
N_SCANS = 5                  # accumulation_window_sec 0.5 at 10 Hz
FASTLIO_BLIND, FASTLIO_FILTER_NUM = 0.2, 3
CONF_TH, MAX_BBOX = 0.25, 0.5
BG = {'wall', 'floor', 'ceiling'}
HEIGHT_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9.0)]

rng = np.random.default_rng(0)
V = F = scene = None


def set_scene(name):
    """Load the Replica mesh used by the Mid-360 simulator (module default: room0)."""
    global V, F, scene
    V, F = load_ply_tris(REPLICA + f'{name}_mesh.ply')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))


if os.path.exists(REPLICA + 'room0_mesh.ply'):     # absent on machines without Replica (e.g. robot-only builds)
    set_scene('room0')


def mid360_scans(T_world_lidar, n_scans, noise=0.02):
    """List of Mid-360-like scans (lidar frame): 20k pts / 0.1 s, az 360, el [-7, 52] deg, 2 cm noise."""
    scans = []
    for _ in range(n_scans):
        az = rng.uniform(-np.pi, np.pi, 20000); el = np.radians(rng.uniform(-7, 52, 20000))
        d = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        dw = d @ T_world_lidar[:3, :3].T
        t = scene.cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(T_world_lidar[:3, 3], dw.shape), dw]
                                            .astype(np.float32)))['t_hit'].numpy()
        ok = np.isfinite(t) & (t > 0.1) & (t < 40)
        scans.append(d[ok] * (t[ok] + rng.normal(0, noise, ok.sum()))[:, None])
    return scans


def sensor_cloud(scans, T_wl, cond):
    """Returns (points in sensor frame, T_world_sensor, T_cam_sensor) as geometry_fusion_node would see them."""
    if cond == 'raw_livox':
        return np.concatenate(scans), T_wl, T_CAM_LIDAR
    # FAST-LIO preprocess: blind range, keep every point_filter_num-th point, IMU body frame.
    kept = []
    for s in scans:
        s = s[np.linalg.norm(s, axis=1) > FASTLIO_BLIND]
        kept.append(s[::FASTLIO_FILTER_NUM])
    pts_b = transform_points(np.concatenate(kept), T_BODY_LIDAR)
    return pts_b, T_wl @ np.linalg.inv(T_BODY_LIDAR), T_CAM_BODY


def load_frames(scene_dir, gsa_dir):
    P = np.loadtxt(os.path.join(scene_dir, 'traj.txt')).reshape(-1, 4, 4)
    for fp in sorted(glob.glob(os.path.join(gsa_dir, '*.pkl.gz'))):
        idx = int(os.path.basename(fp)[5:11])
        depth = cv2.imread(os.path.join(scene_dir, 'results', f'depth{idx:06d}.png'), -1) / 6553.5
        yield idx, P[idx], pickle.load(gzip.open(fp, 'rb')), depth


def run(args):
    scene_dir = os.path.join(args.out, 'replica_rig', 'room0_rig')
    gsa_dir = os.path.join(scene_dir, 'gsa_detections_ram_rig')
    floor = float(np.loadtxt(os.path.join(scene_dir, 'traj.txt')).reshape(-1, 4, 4)[0, 2, 3]) - 0.77
    for d in ('results', 'images', 'videos'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    # TF chain consistency: both conditions must map a world point to the same camera point.
    T_wl = np.eye(4); T_wb = T_wl @ np.linalg.inv(T_BODY_LIDAR)
    chain_err = float(np.abs(T_CAM_LIDAR @ np.linalg.inv(T_wl) - T_CAM_BODY @ np.linalg.inv(T_wb)).max())

    conds = ['raw_livox', 'fastlio_body', 'rgbd_baseline']
    fmaps = {c: ObjectFeatureMap(NN_DELTA, ASSOC_THR, voxel_size=VOXEL) for c in conds}
    stats = {c: [] for c in conds}
    det_writer = cv2.VideoWriter(os.path.join(args.out, 'videos', 'room0_rig_detections.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), 2, (W * 2, H))
    n_frames = 0
    for idx, T_wc, g, gt in load_frames(scene_dir, gsa_dir):
        n_frames += 1
        T_wl = T_wc @ T_CAM_LIDAR
        scans = mid360_scans(T_wl, N_SCANS)
        clouds = {c: sensor_cloud(scans, T_wl, c) for c in ('raw_livox', 'fastlio_body')}
        rgb = cv2.imread(os.path.join(scene_dir, 'results', f'frame{idx:06d}.jpg'))
        vis_mask, vis_pts = rgb.copy(), rgb.copy()
        for i in range(len(g['xyxy'])):
            lab = g['classes'][g['class_id'][i]]; conf = float(g['confidence'][i])
            x1, y1, x2, y2 = g['xyxy'][i]
            if conf < CONF_TH or lab in BG or (x2 - x1) * (y2 - y1) > MAX_BBOX * W * H:
                continue
            m = g['mask'][i]
            vv, uu = np.nonzero(m & (gt > 0)); zz = gt[vv, uu]
            pc = np.c_[(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz]
            if len(pc) == 0:
                continue
            ref_w = transform_points(pc, T_wc); ref_c = np.median(ref_w, axis=0)
            height = float(ref_c[2] - floor)
            feat = g['image_feats'][i] / np.linalg.norm(g['image_feats'][i])
            color = tuple(int(c) for c in rng.integers(60, 255, 3))
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_mask, cnts, -1, color, 2)
            cv2.putText(vis_mask, f'{lab} {conf:.2f}', (int(x1), max(12, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            for c in conds:
                if c == 'rgbd_baseline':
                    pw = voxel_downsample(ref_w, 0.025)
                else:
                    pts_s, T_ws, T_cs = clouds[c]
                    sel = depth_gate(points_inside_mask(pts_s, m, T_cs, K), T_cs, **GATE)
                    if c == 'fastlio_body' and len(sel):
                        uv, _, _ = project_lidar_to_image(sel, T_cs, K)
                        for u, v in uv.astype(int):
                            cv2.circle(vis_pts, (u, v), 2, color, -1)
                    pw = transform_points(sel, T_ws)
                st = dict(frame=idx, label=lab, n=len(pw), height=height, cerr=None)
                stats[c].append(st)
                if len(pw) < 3:
                    continue
                geo = compute_geometry(pw)
                st['cerr'] = float(np.linalg.norm(geo.centroid - ref_c))
                fmaps[c].update(Observation3D(i, lab, conf, pw, feat, geo.centroid, geo.size, geo.yaw, geo.bbox_center))
        cv2.putText(vis_mask, f'frame {idx}: CG-D masks', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis_pts, 'LiDAR pts kept (fastlio_body)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        panel = np.hstack([vis_mask, vis_pts])
        det_writer.write(panel)
        if idx in (0, 100, 200, 300):
            cv2.imwrite(os.path.join(args.out, 'images', f'keyframe_overlay_{idx:06d}.png'), panel)
    det_writer.release()

    metrics = dict(frames=n_frames, floor_z_map=floor, tf_chain_consistency_err=chain_err,
                   params=dict(nn_distance_threshold=NN_DELTA, association_threshold=ASSOC_THR, voxel_size=VOXEL,
                               depth_gate=GATE, accumulated_scans=N_SCANS, fastlio_blind=FASTLIO_BLIND,
                               fastlio_point_filter_num=FASTLIO_FILTER_NUM, conf_threshold=CONF_TH),
                   conditions={})
    for c in conds:
        st = stats[c]; n = np.array([s['n'] for s in st]); ce = np.array([s['cerr'] for s in st if s['cerr'] is not None])
        by_h = {}
        for lo, hi in HEIGHT_BINS:
            sel = [s for s in st if lo <= s['height'] < hi]
            by_h[f'{lo:.1f}-{hi:.1f}m'] = dict(dets=len(sel), usable=round(float(np.mean([s['n'] >= 3 for s in sel])), 3) if sel else None)
        by_cls = {}
        for s in st:
            by_cls.setdefault(s['label'], []).append(s['n'])
        objs = fmaps[c].objects
        n_obs = np.array([o.num_observations for o in objs])
        purity = float(np.mean([max(o.label_counts.values()) / sum(o.label_counts.values()) for o in objs]))
        dup = sum(a.label == b.label and np.linalg.norm(a.centroid - b.centroid) < 0.3 for a, b in itertools.combinations(objs, 2))
        metrics['conditions'][c] = dict(
            detections=len(st), usable=round(float(np.mean(n >= 3)), 3), median_points=float(np.median(n)),
            centroid_err_median_m=round(float(np.median(ce)), 4), centroid_err_p90_m=round(float(np.percentile(ce, 90)), 4),
            usable_by_height_above_floor=by_h,
            median_points_by_class={k: int(np.median(v)) for k, v in sorted(by_cls.items())},
            map=dict(objects=len(objs), multi_view=int((n_obs >= 2).sum()), single=int((n_obs == 1).sum()),
                     max_obs=int(n_obs.max()), label_purity=round(purity, 3), duplicate_pairs=int(dup),
                     max_points_per_object=int(max(len(o.points_map) for o in objs))))
        save_scene_graph_json(os.path.join(args.out, 'results', f'scene_graph_features_{c}.json'), objs, 'map')
        draw_bev(objs, floor, os.path.join(args.out, 'images', f'bev_objects_{c}.png'), c)
    json.dump(metrics, open(os.path.join(args.out, 'results', 'metrics.json'), 'w'), indent=2)
    print(json.dumps({c: {k: v for k, v in m.items() if k != 'median_points_by_class'}
                      for c, m in metrics['conditions'].items()}, indent=1))
    print('tf chain consistency err:', chain_err)

    lidar_overlay_video(scene_dir, os.path.join(args.out, 'videos', 'room0_rig_lidar_overlay.mp4'))


def draw_bev(objs, floor, path, title, res=0.02):
    """Top view: mesh slice (0.05-1.5 m above floor) in grey, object boxes + tags."""
    sl = V[(V[:, 2] > floor + 0.05) & (V[:, 2] < floor + 1.5)]
    lo, hi = V[:, :2].min(0) - 0.3, V[:, :2].max(0) + 0.3
    sz = np.ceil((hi - lo) / res).astype(int)
    img = np.full((sz[1], sz[0], 3), 255, np.uint8)
    px = ((sl[:, :2] - lo) / res).astype(int)
    img[sz[1] - 1 - px[:, 1], px[:, 0]] = (190, 190, 190)
    for o in objs:
        c, (l, w, _), yaw = o.bbox_center, o.size, o.yaw
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = (np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) / 2) @ R.T + c[:2]
        pc = ((corners - lo) / res).astype(int); pc[:, 1] = sz[1] - 1 - pc[:, 1]
        col = (40, 40, 220) if o.num_observations >= 2 else (200, 120, 40)
        cv2.polylines(img, [pc.reshape(-1, 1, 2)], True, col, 2)
        cv2.putText(img, f'{o.object_id}:{o.label}', tuple(pc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, f'{title}: {len(objs)} objects (red: >=2 obs, blue: 1 obs)', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def lidar_overlay_video(scene_dir, path):
    """Every rendered frame: one Mid-360 scan (fastlio_body emulation) projected, colored by depth."""
    P = np.loadtxt(os.path.join(scene_dir, 'traj.txt')).reshape(-1, 4, 4)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
    for idx, T_wc in enumerate(P):
        T_wl = T_wc @ T_CAM_LIDAR
        pts_s, _, T_cs = sensor_cloud(mid360_scans(T_wl, 1), T_wl, 'fastlio_body')
        img = cv2.imread(os.path.join(scene_dir, 'results', f'frame{idx:06d}.jpg'))
        uv, z, _ = project_lidar_to_image(pts_s, T_cs, K)
        ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        cols = cv2.applyColorMap(np.clip((z[ok] - 0.3) / 5 * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
        for (u, v), c in zip(uv[ok].astype(int), cols):
            cv2.circle(img, (u, v), 2, tuple(int(x) for x in c), -1)
        wr.write(img)
    wr.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/exp1'))
    run(ap.parse_args())
