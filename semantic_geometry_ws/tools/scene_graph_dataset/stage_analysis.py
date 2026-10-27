"""
Where are ground-truth objects lost?  Stage-wise funnel for a scene with gt_objects.json:
  visible   : GT centre projects into a keyframe within 4 m and the rendered depth agrees (not occluded)
  detected  : a CG-D mask (after the dataset's detection filter) covers the projected centre
  geometry  : that detection got >= 3 LiDAR points after depth gate + per-observation DBSCAN (camera-depth fallback
              when the LiDAR has fewer, as in the builder)
  exported  : an exported object (objects.json) lies within 0.6 m (XY) with a compatible tag
Reported separately for small (< 0.3 m) and large GT objects.

Usage: python stage_analysis.py SCENE_SOURCE_DIR DATASET_SCENE_DIR [--suffix sg]
"""
import argparse, glob, gzip, json, os, pickle, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dataset as B
from build_dataset import R
from semantic_geometry_fusion.geometry import points_inside_mask, depth_gate, largest_cluster, depth_points_in_mask, voxel_downsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('ds'); ap.add_argument('--suffix', default='sg')
    args = ap.parse_args()
    gt = json.load(open(os.path.join(args.src, 'gt_objects.json')))
    gt = [g for g in gt if max(g.get('size') or [1]) >= 0.05]
    K, W, H = B.camera_of(args.src)
    P = np.loadtxt(os.path.join(args.src, 'traj.txt')).reshape(-1, 4, 4)
    vis = np.zeros(len(gt), bool); det = np.zeros(len(gt), bool); geo = np.zeros(len(gt), bool)
    det_labels = {}
    Gw = np.array([g['position'] for g in gt])
    for fp in sorted(glob.glob(os.path.join(args.src, f'gsa_detections_ram_{args.suffix}', '*.pkl.gz'))):
        idx = int(os.path.basename(fp)[5:11])
        g = pickle.load(gzip.open(fp, 'rb'))
        Tcw = np.linalg.inv(P[idx])
        pc = (Tcw[:3, :3] @ Gw.T).T + Tcw[:3, 3]
        z = pc[:, 2]; u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]; v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
        depth = cv2.imread(os.path.join(args.src, 'results', f'depth{idx:06d}.png'), -1) / 6553.5
        inimg = (z > 0.2) & (z < 4.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
        sizes = np.array([max(x['size']) if x.get('size') else 0.5 for x in gt])
        seen = inimg & (np.abs(depth[vi, ui] - z) < np.maximum(0.15, sizes / 2)) if depth.max() > 0 else inimg
        vis |= seen
        pts_s, T_ws, T_cs = B.sensor_data(args.src, idx, P[idx])
        for i in range(len(g['xyxy'])):
            lab = g['classes'][g['class_id'][i]]; conf = float(g['confidence'][i]); x1, y1, x2, y2 = g['xyxy'][i]
            if conf < R.CONF_TH or lab in R.BG or (x2 - x1) * (y2 - y1) > R.MAX_BBOX * W * H:
                continue
            m = g['mask'][i]
            hit = seen & m[vi, ui]
            if not hit.any():
                continue
            det |= hit
            for j in np.flatnonzero(hit):
                det_labels.setdefault(gt[j]['gt_class'], []).append(lab)
            sel = largest_cluster(depth_gate(points_inside_mask(pts_s, m, T_cs, K), T_cs, **R.GATE), B.FUSION['obs_cluster_eps'])
            if len(sel) < 3 and B.FUSION['depth_fallback'] and depth.max() > 0:     # same fallback as the dataset builder
                sel = largest_cluster(voxel_downsample(depth_gate(depth_points_in_mask(depth, m, K, T_cs), T_cs, **R.GATE), R.VOXEL),
                                      B.FUSION['obs_cluster_eps'])
            if len(sel) >= 3:
                geo |= hit
    objs = json.load(open(os.path.join(args.ds, 'objects.json')))['objects']
    C = np.array([o['bbox_3d']['center'][:2] for o in objs]) if objs else np.zeros((0, 2))
    exp = np.array([any(np.linalg.norm(C[k] - Gw[j, :2]) <= 0.6 and B.compatible(objs[k]['semantic_tag'], gt[j]['gt_class'])
                        for k in range(len(objs))) for j in range(len(gt))])
    small = np.array([0 < max(x.get('size') or [1]) < 0.3 for x in gt])
    for name, m in (('small', small), ('large', ~small)):
        n = m.sum()
        print(f'{name:5s} GT={n:3d} | visible {vis[m].sum():3d} | detected {(det & m).sum():3d} | geometry>=3pts {(geo & m).sum():3d} '
              f'| exported+label {(exp & m).sum():3d}')
    lost_geo = sorted({gt[j]['gt_class'] for j in np.flatnonzero(det & ~geo & small)})
    lost_det = sorted({gt[j]['gt_class'] for j in np.flatnonzero(vis & ~det & small)})
    print('small: detected but no geometry:', lost_geo[:20]); print('small: visible but not detected:', lost_det[:20])
    print('labels CG-D gave to some GT classes:', {k: sorted(set(v))[:5] for k, v in list(det_labels.items())[:12]})


if __name__ == '__main__':
    main()
