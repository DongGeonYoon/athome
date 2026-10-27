"""
Render Replica room0 from the real Tidybot camera rig, in Replica/CG-D dataset format.

Rig (Camera_pose.md / calib.json / fast_lio_ws): D435i color 640x480 with the calibrated K,
camera level at 0.77 m above the floor, Mid-360 0.445 m (from T_cam_lidar).
Robot xy + heading follow the Replica room0 handheld trajectory projected to the floor;
poses where the robot body would intersect geometry are skipped.

Output (CG-D ReplicaDataset layout):
  <out>/replica_rig/room0_rig/results/frame%06d.jpg, depth%06d.png (scale 6553.5)
  <out>/replica_rig/room0_rig/traj.txt   (camera-to-world, OpenCV, row-major 4x4 per line)
  <out>/replica_rig/rig_replica.yaml     (CG-D dataset config)
  <out>/videos/room0_rig_rgb.mp4
"""
import argparse, os, sys
import numpy as np, cv2, open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../replica_feasibility'))
from plyload import load_ply_tris

REPLICA = os.path.expanduser('~/conceptgraph_ws/Replica/')
K = np.array([[602.312561, 0, 329.529297], [0, 602.256470, 240.361496], [0, 0, 1]])
W, H = 640, 480
CAM_HEIGHT = 0.77
ROBOT_RADIUS = 0.37          # circumscribed base radius (mppi_nav footprint)
ROBOT_TOP = 0.95


def load_scene(name='room0'):
    raw = open(REPLICA + f'{name}_mesh.ply', 'rb').read()
    h = raw.index(b'end_header\n') + len(b'end_header\n')
    nv = int(raw[:h].decode().split('element vertex ')[1].split()[0])
    col = np.frombuffer(raw, np.dtype([('p', '<f4', 3), ('n', '<f4', 3), ('c', 'u1', 3)]), nv, h)['c']
    V, F = load_ply_tris(REPLICA + f'{name}_mesh.ply')
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return sc, F, col.astype(np.float32)


def cast(sc, origins, dirs):
    r = sc.cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(origins, dirs.shape), dirs].astype(np.float32)))
    return r['t_hit'].numpy(), r['primitive_ids'].numpy(), r['primitive_uvs'].numpy()


def render(sc, F, col, T_wc):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    t, pid, uv = cast(sc, T_wc[:3, 3], d @ T_wc[:3, :3].T)  # d has z=1 -> t is z-depth
    ok = np.isfinite(t)
    rgb = np.zeros((H * W, 3), np.float32)
    tri = F[pid[ok]]
    w1, w2 = uv[ok, 0:1], uv[ok, 1:2]
    rgb[ok] = (1 - w1 - w2) * col[tri[:, 0]] + w1 * col[tri[:, 1]] + w2 * col[tri[:, 2]]
    depth = np.where(ok, t, 0.0)
    return rgb.reshape(H, W, 3).astype(np.uint8), depth.reshape(H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/exp1'))
    ap.add_argument('--step', type=int, default=5, help='Replica trajectory subsampling')
    ap.add_argument('--scene', default='room0')
    args = ap.parse_args()
    scene_dir = os.path.join(args.out, 'replica_rig', f'{args.scene}_rig')
    os.makedirs(os.path.join(scene_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'videos'), exist_ok=True)

    sc, F, col = load_scene(args.scene)
    P = np.loadtxt(REPLICA + f'{args.scene}/traj.txt').reshape(-1, 4, 4)

    # Floor height: lowest cluster of downward hits along the trajectory.
    xy = P[::args.step, :3, 3]
    t, *_ = cast(sc, np.zeros(3), np.tile([0, 0, -1.0], (len(xy), 1)))
    hits = xy[:, 2] - cast(sc, xy, np.tile([0, 0, -1.0], (len(xy), 1)))[0]
    floor = float(np.median(hits[hits < hits.min() + 0.1]))

    # Free-space grid for the robot body: horizontal rays (16 dirs x 8 heights up to ROBOT_TOP)
    # must not hit geometry within ROBOT_RADIUS (floor/ceiling not counted). Cells must also
    # have floor directly below (inside the room).
    xs = np.arange(P[:, 0, 3].min() - 1.5, P[:, 0, 3].max() + 1.5, 0.1)
    ys = np.arange(P[:, 1, 3].min() - 1.5, P[:, 1, 3].max() + 1.5, 0.1)
    gx, gy = np.meshgrid(xs, ys); cells = np.c_[gx.ravel(), gy.ravel()]
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    heights = np.linspace(0.08, ROBOT_TOP, 8)
    dirs1 = np.tile(np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)], (len(heights), 1))
    free = np.zeros(len(cells), bool)
    for ci, (cx_, cy_) in enumerate(cells):
        o = np.repeat([[cx_, cy_, floor + h] for h in heights], len(ang), axis=0)
        t_h = cast(sc, o, dirs1)[0]
        t_dn = cast(sc, np.array([cx_, cy_, floor + 0.3]), np.array([[0, 0, -1.0]]))[0][0]
        free[ci] = np.isfinite(t_h).all() and t_h.min() >= ROBOT_RADIUS and abs(t_dn - 0.3) < 0.05
    free_xy = cells[free]

    poses, kept_src = [], []
    for i in range(0, len(P), args.step):
        p = P[i, :3, 3]
        fwd = P[i, :3, 2].copy(); fwd[2] = 0
        if np.linalg.norm(fwd) < 1e-3 or len(free_xy) == 0:
            continue
        fwd /= np.linalg.norm(fwd)
        dist = np.linalg.norm(free_xy - p[:2], axis=1)
        if dist.min() > 1.0:   # no reachable free spot near this trajectory point
            continue
        q = free_xy[np.argmin(dist)]
        down = np.array([0, 0, -1.0]); right = np.cross(down, fwd)
        T = np.eye(4); T[:3, :3] = np.c_[right, down, fwd]; T[:3, 3] = [q[0], q[1], floor + CAM_HEIGHT]
        poses.append(T); kept_src.append(i)
    print(f'free cells: {free.sum()} / {len(cells)}')

    writer = cv2.VideoWriter(os.path.join(args.out, 'videos', f'{args.scene}_rig_rgb.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
    for k, T in enumerate(poses):
        rgb, depth = render(sc, F, col, T)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(scene_dir, 'results', f'frame{k:06d}.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(scene_dir, 'results', f'depth{k:06d}.png'), np.clip(depth * 6553.5, 0, 65535).astype(np.uint16))
        writer.write(bgr)
    writer.release()

    np.savetxt(os.path.join(scene_dir, 'traj.txt'), np.array([T.reshape(-1) for T in poses]))
    np.savetxt(os.path.join(scene_dir, 'source_replica_index.txt'), np.array(kept_src, int), fmt='%d')
    with open(os.path.join(args.out, 'replica_rig', 'rig_replica.yaml'), 'w') as f:
        f.write(f"""dataset_name: 'replica'
camera_params:
  image_height: {H}
  image_width: {W}
  fx: {K[0, 0]}
  fy: {K[1, 1]}
  cx: {K[0, 2]}
  cy: {K[1, 2]}
  png_depth_scale: 6553.5
  crop_edge: 0
""")
    print(f'floor z={floor:.3f} | trajectory poses tried={len(range(0, len(P), args.step))} '
          f'kept={len(poses)} (skipped for robot clearance={len(range(0, len(P), args.step)) - len(poses)})')


if __name__ == '__main__':
    main()
