"""
Active Observation experiment on Replica room0 from the Tidybot rig (require.md items 1 and 3).

For each search location and config (A6: 1280x720, 6 views; B8: 640x480, 8 views):
  CG-D semantics on rendered views + simulated Mid-360 geometry (FAST-LIO body cloud emulation,
  T_cam_body, map <- body TF chain) -> semantic_geometry_fusion -> outputs:

  results/perception_objects/<cfg>/loc<j>_view<k>.json  per-view message mirror (map frame, empty list allowed)
  results/scene_graph_features_<cfg>.json               Scene Graph Object-Centric Feature (static map)
  results/floor_plane.json                              RANSAC floor plane from LiDAR (map frame)
  results/metrics.json                                  coverage / detection / geometry / timing / targets
  images/, videos/                                       panoramas, BEV, rotation videos
"""
import argparse, glob, gzip, itertools, json, os, pickle, re, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_exp as R  # mesh, Mid-360 simulator, FAST-LIO body emulation, rig extrinsics, fusion params
from rig_exp import (mid360_scans, sensor_cloud, T_CAM_LIDAR, NN_DELTA, ASSOC_THR, VOXEL, GATE, N_SCANS,
                     CONF_TH, MAX_BBOX, BG, draw_bev)
from semantic_geometry_fusion.geometry import (points_inside_mask, transform_points, compute_geometry,
                                               project_lidar_to_image, depth_gate)
from semantic_geometry_fusion.feature_fusion import Observation3D, ObjectFeatureMap
from semantic_geometry_fusion.scene_graph_export import save_scene_graph_json, map_object_to_dict

COND = 'fastlio_body'   # runtime-equivalent input (costmap TF chain)
SMALL_OBJ_M = 0.30      # "small object": largest GT extent below this


def r4(x):
    return [round(float(v), 4) for v in x]


def fit_floor(points_map, iters=300, tol=0.03, seed=0):
    """RANSAC plane on the lowest 30 % of points; returns (normal, d) with n.z > 0 and inlier count."""
    rs = np.random.default_rng(seed)
    low = points_map[points_map[:, 2] < np.percentile(points_map[:, 2], 30)]
    best = (None, None, 0)
    for _ in range(iters):
        p = low[rs.choice(len(low), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) < 0.95:
            continue
        n = n if n[2] > 0 else -n
        d = -n @ p[0]
        inl = int((np.abs(low @ n + d) < tol).sum())
        if inl > best[2]:
            best = (n, d, inl)
    n, d, inl = best
    pts = low[np.abs(low @ n + d) < tol]          # least-squares refit on inliers
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    a, b, c = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
    n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
    return n, -c * n[2], inl


def load_scene_dir(sd, suffix):
    P = np.loadtxt(os.path.join(sd, 'traj.txt')).reshape(-1, 4, 4)
    out = []
    for fp in sorted(glob.glob(os.path.join(sd, f'gsa_detections_ram_{suffix}', '*.pkl.gz'))):
        k = int(os.path.basename(fp)[5:11])
        depth = cv2.imread(os.path.join(sd, 'results', f'depth{k:06d}.png'), -1) / 6553.5
        rgb = cv2.imread(os.path.join(sd, 'results', f'frame{k:06d}.jpg'))
        out.append((k, P[k], pickle.load(gzip.open(fp, 'rb')), depth, rgb))
    return out


def process_view(g, T_wc, gt, K, W, H, scans_cache_key, scans_cache):
    """Detections of one view -> list of (Observation3D, detection record)."""
    T_wl = T_wc @ T_CAM_LIDAR
    if scans_cache_key not in scans_cache:
        scans_cache[scans_cache_key] = mid360_scans(T_wl, N_SCANS)
    pts_s, T_ws, T_cs = sensor_cloud(scans_cache[scans_cache_key], T_wl, COND)
    res = []
    for i in range(len(g['xyxy'])):
        lab = g['classes'][g['class_id'][i]]; conf = float(g['confidence'][i])
        x1, y1, x2, y2 = [float(v) for v in g['xyxy'][i]]
        if conf < CONF_TH or lab in BG or (x2 - x1) * (y2 - y1) > MAX_BBOX * W * H:
            continue
        m = g['mask'][i]
        vv, uu = np.nonzero(m & (gt > 0))
        if len(vv) == 0:
            continue
        zz = gt[vv, uu]
        ref = transform_points(np.c_[(uu - K[0, 2]) * zz / K[0, 0], (vv - K[1, 2]) * zz / K[1, 1], zz], T_wc)
        lo, hi = np.percentile(ref, 5, axis=0), np.percentile(ref, 95, axis=0)
        rec = dict(det=i, label=lab, conf=round(conf, 4), bbox_px=r4([x1, y1, x2, y2]),
                   box_w_px=round(x2 - x1, 1), box_h_px=round(y2 - y1, 1),
                   gt_extent_m=round(float(np.max(hi - lo)), 3), gt_centroid=r4(np.median(ref, axis=0)),
                   lidar_pts=0, obs=None)
        sel = depth_gate(points_inside_mask(pts_s, m, T_cs, K), T_cs, **GATE)
        rec['lidar_pts'] = int(len(sel))
        if len(sel) >= 3:
            pw = transform_points(sel, T_ws); geo = compute_geometry(pw)
            feat = g['image_feats'][i] / np.linalg.norm(g['image_feats'][i])
            rec['obs'] = Observation3D(i, lab, conf, pw, feat, geo.centroid, geo.size, geo.yaw, geo.bbox_center)
        res.append(rec)
    return res, pts_s, T_ws, T_cs


def topic_message(cfg, j, k, yaw_deg, recs, stamp):
    """Mirror of one per-view perception message (empty objects list is valid)."""
    objs = []
    for r in recs:
        o = r['obs']
        if o is None:
            continue
        objs.append(dict(detection_id=int(r['det']), label=r['label'], confidence=r['conf'],
                         centroid=dict(x=round(float(o.centroid[0]), 4), y=round(float(o.centroid[1]), 4), z=round(float(o.centroid[2]), 4)),
                         bbox_3d=dict(center=r4(o.bbox_center), size=r4(o.size), yaw=round(float(o.yaw), 4)),
                         num_points=int(len(o.points_map)),
                         clip_feature=[round(float(v), 6) for v in o.clip_feature]))
    return dict(header=dict(frame_id='map', stamp=stamp), config=cfg, location=j, view=k,
                yaw_deg=round(yaw_deg, 1), objects=objs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/exp1'))
    args = ap.parse_args()
    root = os.path.join(args.out, 'active_obs')
    meta = json.load(open(os.path.join(root, 'meta.json')))
    floor_gt = meta['floor_z']
    for d in ('results/perception_objects/A6', 'results/perception_objects/B8', 'images', 'videos'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    timing = {}
    if os.path.exists(os.path.join(root, 'cgd_timing.txt')):
        for line in open(os.path.join(root, 'cgd_timing.txt')):
            m = re.search(r'\] (\S+) \((\S+)\) exit=\d+ sec=(\d+) frames=(\d+)', line)
            if m:
                timing[f'{m.group(1)}:{m.group(2)}'] = int(m.group(3)) / int(m.group(4))

    metrics = dict(floor_gt_z=floor_gt, params=dict(condition=COND, nn_distance_threshold=NN_DELTA,
                   association_threshold=ASSOC_THR, voxel_size=VOXEL, depth_gate=GATE, accumulated_scans=N_SCANS,
                   conf_threshold=CONF_TH), configs={})
    # Only locations whose CG-D run finished for every config (the run can be interrupted).
    def complete(j, cfg):
        sd = os.path.join(root, f'loc{j}_{cfg}')
        return len(glob.glob(os.path.join(sd, 'gsa_detections_ram_base', '*.pkl.gz'))) == meta['configs'][cfg]['n_views']
    locs_done = [j for j in range(len(meta['locations'])) if all(complete(j, c) for c in ('A6', 'B8'))]
    metrics['locations_used'] = locs_done
    all_lidar_map = []
    stamp = 0.0
    fmaps = {}
    for cfg in ('A6', 'B8'):
        c = meta['configs'][cfg]; K = np.array(c['K']); W, H, nv = c['W'], c['H'], c['n_views']
        hfov = c['hfov_deg']; step = 360.0 / nv
        fmap = ObjectFeatureMap(NN_DELTA, ASSOC_THR, voxel_size=VOXEL); fmaps[cfg] = fmap
        recs_all, per_loc = [], {}
        vid = cv2.VideoWriter(os.path.join(args.out, 'videos', f'active_obs_{cfg}.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), 2, (W, H))
        for j in locs_done:
            sd = os.path.join(root, f'loc{j}_{cfg}')
            views = load_scene_dir(sd, 'base')
            loc_map = ObjectFeatureMap(NN_DELTA, ASSOC_THR, voxel_size=VOXEL)
            cache, pano, n_empty = {}, [], 0
            for k, T_wc, g, gt, rgb in views:
                recs, pts_s, T_ws, T_cs = process_view(g, T_wc, gt, K, W, H, (j, k), cache)
                all_lidar_map.append(transform_points(pts_s, T_ws))
                for r in recs:
                    if r['obs'] is not None:
                        fmap.update(r['obs']); loc_map.update(r['obs'])
                msg = topic_message(cfg, j, k, k * step, recs, stamp); stamp += 0.1
                n_empty += int(len(msg['objects']) == 0)
                json.dump(msg, open(os.path.join(args.out, 'results', 'perception_objects', cfg, f'loc{j}_view{k}.json'), 'w'))
                # visual: masks + kept LiDAR points
                vis = rgb.copy()
                uv, z, _ = project_lidar_to_image(pts_s, T_cs, K)
                ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                for u, v in uv[ok].astype(int)[::2]:
                    cv2.circle(vis, (u, v), 1, (200, 200, 200), -1)
                for r in recs:
                    i = r['det']; col = (0, 0, 255) if r['obs'] is not None else (0, 165, 255)
                    cnts, _ = cv2.findContours(g['mask'][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnts, -1, col, 2)
                    x1, y1 = int(r['bbox_px'][0]), int(r['bbox_px'][1])
                    cv2.putText(vis, f"{r['label']} {r['conf']:.2f} L{r['lidar_pts']}", (x1, max(14, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * W / 640, col, 1)
                cv2.putText(vis, f'{cfg} loc{j} view{k} yaw {k * step:.0f}deg', (8, int(22 * W / 640)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * W / 640, (0, 0, 255), 2)
                vid.write(vis); pano.append(cv2.resize(vis, (320, int(320 * H / W))))
                for r in recs:
                    r.update(location=j, view=k)
                recs_all += recs
            cv2.imwrite(os.path.join(args.out, 'images', f'panorama_{cfg}_loc{j}.jpg'), np.hstack(pano))
            per_loc[f'loc{j}'] = dict(views=len(views), empty_views=n_empty,
                                      detections=sum(1 for r in recs_all if r['location'] == j),
                                      fused_objects=len(loc_map.objects),
                                      labels=sorted({o.label for o in loc_map.objects}),
                                      cgd_sec_per_frame=round(timing.get(f'loc{j}_{cfg}:base', float('nan')), 1))
        vid.release()
        n = np.array([r['lidar_pts'] for r in recs_all])
        small = [r for r in recs_all if r['gt_extent_m'] < SMALL_OBJ_M]
        small_objs = [o for o in fmap.objects if max(o.size[:2]) < SMALL_OBJ_M]
        metrics['configs'][cfg] = dict(
            resolution=f'{W}x{H}', fx=round(float(K[0, 0]), 2), hfov_deg=round(hfov, 2), views=nv, step_deg=step,
            overlap_deg=round(hfov - step, 2), rotation_tolerance_deg=round((hfov - step) / 2, 2),
            angular_coverage_pct=round(min(100.0, 100 * nv * hfov / 360), 1),
            frames=len(set((r['location'], r['view']) for r in recs_all)) if recs_all else 0,
            detections=len(recs_all), lidar_usable=round(float(np.mean(n >= 3)), 3),
            small_detections=len(small),
            small_box_median_px=round(float(np.median([max(r['box_w_px'], r['box_h_px']) for r in small])), 1) if small else None,
            small_lidar_usable=round(float(np.mean([r['lidar_pts'] >= 3 for r in small])), 3) if small else None,
            fused_objects=len(fmap.objects), fused_small_objects=len(small_objs),
            fused_labels=sorted({o.label for o in fmap.objects}),
            cgd_sec_per_frame_mean=round(float(np.nanmean([v['cgd_sec_per_frame'] for v in per_loc.values()])), 1),
            per_location=per_loc)
        save_scene_graph_json(os.path.join(args.out, 'results', f'scene_graph_features_{cfg}.json'), fmap.objects, 'map')
        draw_bev(fmap.objects, floor_gt, os.path.join(args.out, 'images', f'bev_objects_{cfg}.png'), f'active obs {cfg}')

    # Floor plane from accumulated LiDAR (map frame)
    n_vec, d, inl = fit_floor(np.concatenate(all_lidar_map))
    floor_est = -d / n_vec[2]
    floor = dict(frame_id='map', normal=r4(n_vec), d=round(float(d), 4), floor_z_at_origin=round(float(floor_est), 4),
                 tilt_deg=round(float(np.degrees(np.arccos(n_vec[2]))), 3), inliers=inl,
                 gt_floor_z=round(floor_gt, 4), error_m=round(float(floor_est - floor_gt), 4))
    json.dump(floor, open(os.path.join(args.out, 'results', 'floor_plane.json'), 'w'), indent=2)
    metrics['floor_plane'] = floor
    for cfg, fmap in fmaps.items():
        metrics['configs'][cfg]['object_height_above_floor_m'] = {
            f'{o.object_id}:{o.label}': round(float(o.bbox_center[2] + o.size[2] / 2 - floor_est), 3) for o in fmap.objects}

    # Search-target query (item 3): loc2_A6 with vs without targets
    tgt_dir = os.path.join(root, 'loc2_A6')
    if glob.glob(os.path.join(tgt_dir, 'gsa_detections_ram_targets', '*.pkl.gz')):
        targets = ['vase', 'book', 'remote control', 'throw blanket', 'picture frame']
        c = meta['configs']['A6']; K = np.array(c['K'])
        out = {}
        for suffix in ('base', 'targets'):
            recs, cache = [], {}
            for k, T_wc, g, gt, rgb in load_scene_dir(tgt_dir, suffix):
                rr, *_ = process_view(g, T_wc, gt, K, c['W'], c['H'], (2, k), cache)
                recs += rr
            hits = [r for r in recs if r['label'] in targets]
            out[suffix] = dict(detections=len(recs), labels=sorted({r['label'] for r in recs}),
                               target_hits={t: [dict(view=None, conf=r['conf'], lidar_pts=r['lidar_pts'], gt_extent_m=r['gt_extent_m'])
                                                for r in hits if r['label'] == t] for t in targets},
                               cgd_sec_per_frame=round(timing.get(f'loc2_A6:{suffix}', float('nan')), 1))
        metrics['search_target_query'] = dict(scene='loc2_A6', targets=targets, **out)

    json.dump(metrics, open(os.path.join(args.out, 'results', 'metrics.json'), 'w'), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'configs'}, indent=1)[:3000])
    for cfg, m in metrics['configs'].items():
        print(cfg, json.dumps({k: v for k, v in m.items() if k not in ('per_location', 'object_height_above_floor_m', 'fused_labels')}))
        for l, v in m['per_location'].items():
            print('   ', l, json.dumps(v))


if __name__ == '__main__':
    main()
