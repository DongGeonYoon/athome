"""
Render Active Observation (in-place rotation) views of Replica room0 from the Tidybot rig.

Search locations: robot-free cells (base radius 0.37 m, height 0.95 m), farthest-point sampled.
Configs (require.md item 1):
  A6 : 1280x720, K estimated from the 640x480 calibration (fx x1.5), 6 views @ 60 deg
  B8 : 640x480,  calibrated K,                                         8 views @ 45 deg
Each (location, config) is a CG-D Replica-format scene dir:
  <out>/active_obs/loc<j>_<cfg>/results/frame%06d.jpg, depth%06d.png ; traj.txt ; <cfg>.yaml
"""
import argparse, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render_rig import load_scene, cast, REPLICA, CAM_HEIGHT, ROBOT_RADIUS, ROBOT_TOP

K_640 = np.array([[602.312561, 0, 329.529297], [0, 602.256470, 240.361496], [0, 0, 1]])
S = 1.5  # 640x480 = 1440x1080 centre crop x 4/9 ; 1280x720 = 1920x1080 x 2/3  ->  scale 1.5
K_1280 = np.array([[K_640[0, 0] * S, 0, (K_640[0, 2] - 320) * S + 640],
                   [0, K_640[1, 1] * S, (K_640[1, 2] - 240) * S + 360], [0, 0, 1]])
CONFIGS = {
    'A6': dict(W=1280, H=720, K=K_1280, n_views=6),
    'B8': dict(W=640, H=480, K=K_640, n_views=8),
}


def render(sc, F, col, T_wc, K, W, H):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    t, pid, uv = cast(sc, T_wc[:3, 3], d @ T_wc[:3, :3].T)
    ok = np.isfinite(t)
    rgb = np.zeros((H * W, 3), np.float32)
    tri = F[pid[ok]]
    w1, w2 = uv[ok, 0:1], uv[ok, 1:2]
    rgb[ok] = (1 - w1 - w2) * col[tri[:, 0]] + w1 * col[tri[:, 1]] + w2 * col[tri[:, 2]]
    return rgb.reshape(H, W, 3).astype(np.uint8), np.where(ok, t, 0.0).reshape(H, W)


def free_cells(sc, floor, P):
    xs = np.arange(P[:, 0, 3].min() - 1.5, P[:, 0, 3].max() + 1.5, 0.1)
    ys = np.arange(P[:, 1, 3].min() - 1.5, P[:, 1, 3].max() + 1.5, 0.1)
    gx, gy = np.meshgrid(xs, ys); cells = np.c_[gx.ravel(), gy.ravel()]
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); heights = np.linspace(0.08, ROBOT_TOP, 8)
    dirs = np.tile(np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)], (len(heights), 1))
    free = []
    for cx_, cy_ in cells:
        th = cast(sc, np.repeat([[cx_, cy_, floor + h] for h in heights], len(ang), axis=0), dirs)[0]
        tdn = cast(sc, np.array([cx_, cy_, floor + 0.3]), np.array([[0, 0, -1.0]]))[0][0]
        if np.isfinite(th).all() and th.min() >= ROBOT_RADIUS and abs(tdn - 0.3) < 0.05:
            free.append((cx_, cy_))
    return np.array(free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/exp1'))
    ap.add_argument('--n-locations', type=int, default=3)
    args = ap.parse_args()
    sc, F, col = load_scene()
    P = np.loadtxt(REPLICA + 'room0/traj.txt').reshape(-1, 4, 4)
    xy = P[::5, :3, 3]
    hits = xy[:, 2] - cast(sc, xy, np.tile([0, 0, -1.0], (len(xy), 1)))[0]
    floor = float(np.median(hits[hits < hits.min() + 0.1]))
    free = free_cells(sc, floor, P)

    # Farthest-point sampling, seeded at the free cell closest to the free-space centroid.
    locs = [free[np.argmin(np.linalg.norm(free - free.mean(0), axis=1))]]
    while len(locs) < args.n_locations:
        d = np.min([np.linalg.norm(free - l, axis=1) for l in locs], axis=0)
        locs.append(free[np.argmax(d)])

    root = os.path.join(args.out, 'active_obs')
    meta = dict(floor_z=floor, free_cells=int(len(free)), locations=[l.tolist() for l in locs],
                configs={k: dict(W=c['W'], H=c['H'], K=c['K'].tolist(), n_views=c['n_views'],
                                 hfov_deg=float(2 * np.degrees(np.arctan(c['W'] / 2 / c['K'][0, 0]))))
                         for k, c in CONFIGS.items()})
    for j, loc in enumerate(locs):
        for name, c in CONFIGS.items():
            sd = os.path.join(root, f'loc{j}_{name}')
            os.makedirs(os.path.join(sd, 'results'), exist_ok=True)
            poses = []
            for k in range(c['n_views']):
                yaw = 2 * np.pi * k / c['n_views']
                fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0]); down = np.array([0, 0, -1.0])
                T = np.eye(4); T[:3, :3] = np.c_[np.cross(down, fwd), down, fwd]
                T[:3, 3] = [loc[0], loc[1], floor + CAM_HEIGHT]
                rgb, depth = render(sc, F, col, T, c['K'], c['W'], c['H'])
                cv2.imwrite(os.path.join(sd, 'results', f'frame{k:06d}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(sd, 'results', f'depth{k:06d}.png'), np.clip(depth * 6553.5, 0, 65535).astype(np.uint16))
                poses.append(T.reshape(-1))
            np.savetxt(os.path.join(sd, 'traj.txt'), np.array(poses))
            K = c['K']
            with open(os.path.join(root, f'{name}.yaml'), 'w') as f:
                f.write(f"dataset_name: 'replica'\ncamera_params:\n  image_height: {c['H']}\n  image_width: {c['W']}\n"
                        f"  fx: {K[0, 0]}\n  fy: {K[1, 1]}\n  cx: {K[0, 2]}\n  cy: {K[1, 2]}\n"
                        f"  png_depth_scale: 6553.5\n  crop_edge: 0\n")
    json.dump(meta, open(os.path.join(root, 'meta.json'), 'w'), indent=2)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
