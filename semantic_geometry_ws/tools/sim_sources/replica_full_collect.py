"""
Replica full release (multi-room apartments) -> common intermediate format, Tidybot rig geometry.

Mesh: <scene>/mesh.ply (z up, vertex colours). GT: habitat/mesh_semantic.ply (same frame, per-face object_id)
+ habitat/info_semantic.json (class names); each GT object = vertices of its faces (AABB centre / size).
Main floor = the floor level with the largest area (apartment_0 has two levels).
Robot spots: robot-free cells (base radius 0.37 m up to 0.95 m) on the main floor, largest connected region,
farthest-point sampled (count from area); N views per spot. RGB/depth rendered from vertex colours at the rig
camera (0.77 m, 640x480, D435i K); LiDAR = Mid-360 ray casting at the rig LiDAR pose, FAST-LIO body-cloud
emulation (blind 0.2 m, point_filter_num 3, 5 scans).

Usage (conceptgraph env): python replica_full_collect.py OUT_ROOT --scenes apartment_1 frl_apartment_0 ...
"""
import argparse, json, os, sys
import numpy as np, cv2, open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../replica_rig')); sys.path.insert(0, os.path.join(HERE, '../replica_feasibility'))
import rig_exp as R
from render_rig import cast, CAM_HEIGHT, ROBOT_RADIUS, ROBOT_TOP
from render_active_obs import render
from plyload import load_ply_tris

ROOT = os.path.expanduser('~/datasets/replica_full')
K = np.array([[602.312561, 0, 329.529297], [0, 602.256470, 240.361496], [0, 0, 1]]); W, H = 640, 480
STRUCT = {'wall', 'floor', 'ceiling', 'undefined', 'beam', 'vent', 'wall-plug', 'switch', 'panel', 'pipe', 'anonymize_text',
          'anonymize_picture', 'other-leaf', 'stair', 'handrail'}


def read_ply(path, extra_face_prop=False):
    raw = open(path, 'rb').read(); h = raw.index(b'end_header\n') + 11
    nv = int(raw[:h].decode().split('element vertex ')[1].split()[0])
    nf = int(raw[:h].decode().split('element face ')[1].split()[0])
    v = np.frombuffer(raw, np.dtype([('p', '<f4', 3), ('n', '<f4', 3), ('c', 'u1', 3)]), nv, h)
    off = h + nv * 27
    if extra_face_prop:     # semantic mesh: uint8 count, 4 x uint32 indices, uint16 object_id
        f = np.frombuffer(raw, np.dtype([('n', 'u1'), ('i', '<u4', 4), ('o', '<u2')]), nf, off)
        return v, f['i'], f['o']
    return v, None, None


def load(scene):
    V, F = load_ply_tris(os.path.join(ROOT, scene, 'mesh.ply'))
    v, _, _ = read_ply(os.path.join(ROOT, scene, 'mesh.ply'))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return sc, V, F, v['c'].astype(np.float32)


def main_floor(sc, V):
    """Floor height = the most populated level of first upward hits from below the mesh (lowest surface per xy)."""
    lo, hi = V[:, :2].min(0), V[:, :2].max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.2), np.arange(lo[1], hi[1], 0.2))
    o = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, V[:, 2].min() - 0.1)]
    t, _, _ = cast(sc, o, np.tile([0, 0, 1.0], (len(o), 1)))
    z = (o[:, 2] + t)[np.isfinite(t)]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.05, 0.05))
    mode = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    return float(np.median(z[np.abs(z - mode) < 0.15]))


def free_cells(sc, floor, V, step=0.1):
    lo, hi = V[:, :2].min(0), V[:, :2].max(0)
    xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
    gx, gy = np.meshgrid(xs, ys); cells = np.c_[gx.ravel(), gy.ravel()]
    tdn = cast(sc, np.c_[cells, np.full(len(cells), floor + 0.3)], np.tile([0, 0, -1.0], (len(cells), 1)))[0]
    on_floor = np.isfinite(tdn) & (np.abs(tdn - 0.3) < 0.12)   # scanned floors are uneven (~10 cm)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); heights = np.linspace(0.15, ROBOT_TOP, 8)          # above uneven scanned floors / rug edges
    dirs = np.tile(np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)], (len(heights), 1))
    free = np.zeros(len(cells), bool)
    for i in np.flatnonzero(on_floor):
        th = cast(sc, np.repeat([[cells[i, 0], cells[i, 1], floor + h] for h in heights], len(ang), 0), dirs)[0]
        free[i] = np.nanmin(np.where(np.isfinite(th), th, np.inf)) >= ROBOT_RADIUS   # rays through open doors are free
    grid = free.reshape(gy.shape).astype(np.uint8)
    n, lab = cv2.connectedComponents(grid)
    if n <= 1:
        return np.zeros((0, 2))
    big = 1 + int(np.argmax([np.sum(lab == k) for k in range(1, n)]))
    return cells[(lab == big).ravel()]


def gt_objects(scene):
    v, faces, oid = read_ply(os.path.join(ROOT, scene, 'habitat', 'mesh_semantic.ply'), True)
    info = json.load(open(os.path.join(ROOT, scene, 'habitat', 'info_semantic.json')))
    cls = {o['id']: o['class_name'] for o in info['objects']}
    out = []
    for i in np.unique(oid):
        name = cls.get(int(i), 'undefined')
        if name in STRUCT:
            continue
        p = v['p'][np.unique(faces[oid == i].ravel())]
        lo, hi = p.min(0), p.max(0)
        out.append(dict(id=int(i), gt_class=name, position=((lo + hi) / 2).round(4).tolist(), size=(hi - lo).round(4).tolist()))
    return out


def collect(scene, out, views, rng):
    sc, V, F, col = load(scene)
    R.scene, R.V, R.F = sc, V, F                    # Mid-360 ray casting on this mesh
    floor = main_floor(sc, V)
    cells = free_cells(sc, floor, V)
    if len(cells) == 0:
        print(scene, 'no robot-free cells'); return 0
    area = len(cells) * 0.01
    n_spots = int(np.clip(area / 2.5, 8, 30))
    spots = [cells[np.argmin(np.linalg.norm(cells - cells.mean(0), axis=1))]]
    while len(spots) < n_spots:
        d = np.min([np.linalg.norm(cells - s, axis=1) for s in spots], axis=0)
        if d.max() < 1.0:
            break
        spots.append(cells[int(np.argmax(d))])
    sd = os.path.join(out, f'replica_full_{scene}')
    for dd in ('results', 'lidar'):
        os.makedirs(os.path.join(sd, dd), exist_ok=True)
    traj, mp, mo, k = [], [], [], 0
    for sx, sy in spots:
        for vi in range(views):
            yaw = 2 * np.pi * vi / views
            fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0]); down = np.array([0, 0, -1.0])
            Tc = np.eye(4); Tc[:3, :3] = np.c_[np.cross(down, fwd), down, fwd]; Tc[:3, 3] = [sx, sy, floor + CAM_HEIGHT]
            rgb, depth = render(sc, F, col, Tc, K, W, H)
            if np.mean(depth > 0) < 0.6:          # unscanned holes / looking outside the mesh: no real camera sees this
                continue
            cv2.imwrite(os.path.join(sd, 'results', f'frame{k:06d}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(sd, 'results', f'depth{k:06d}.png'), np.clip(depth * 6553.5, 0, 65535).astype(np.uint16))
            T_wl = Tc @ R.T_CAM_LIDAR
            pts_b, T_wb, T_cb = R.sensor_cloud(R.mid360_scans(T_wl, R.N_SCANS), T_wl, 'fastlio_body')
            np.savez_compressed(os.path.join(sd, 'lidar', f'frame{k:06d}.npz'), points=pts_b.astype(np.float32),
                                T_map_sensor=T_wb, T_cam_sensor=T_cb)
            traj.append(Tc.reshape(-1))
            if vi % 2 == 0:
                mp.append((T_wb[:3, :3] @ pts_b.T).T + T_wb[:3, 3]); mo.append(np.repeat(T_wb[None, :3, 3], len(pts_b), 0))
            k += 1
    np.savetxt(os.path.join(sd, 'traj.txt'), np.array(traj))
    json.dump(dict(K=K.tolist(), width=W, height=H, distortion=[0] * 5), open(os.path.join(sd, 'camera.json'), 'w'), indent=1)
    mp, mo = np.concatenate(mp), np.concatenate(mo)
    key = np.floor(mp / 0.03).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    np.savez_compressed(os.path.join(sd, 'map_points.npz'), points=mp[first].astype(np.float32), origins=mo[first].astype(np.float32))
    gt = gt_objects(scene)
    json.dump(gt, open(os.path.join(sd, 'gt_objects.json'), 'w'), indent=1)
    json.dump(dict(source='replica_full', scene=scene, frame_id='map', floor_z=floor, robot_free_area_m2=round(area, 1),
                   spots=len(spots), views=views, keyframes=k,
                   lidar_input='FAST-LIO /cloud_registered_body emulation from Mid-360 ray casting (blind 0.2 m, point_filter_num 3, 5 scans)',
                   pose_source='ground truth (rendered)', camera='640x480, D435i calibrated K'),
              open(os.path.join(sd, 'source.json'), 'w'), indent=1)
    print(f'{scene}: floor {floor:.2f} free area {area:.1f} m2 spots {len(spots)} keyframes {k} gt objects {len(gt)}', flush=True)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--scenes', nargs='+', required=True); ap.add_argument('--views', type=int, default=6)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for s in args.scenes:
        collect(s, args.out, args.views, rng)


if __name__ == '__main__':
    main()
