"""
ProcTHOR (AI2-THOR) house -> common intermediate format, Tidybot rig geometry.

Cameras are AI2-THOR third-party cameras placed at robot poses (no agent physics):
  RGB  : camera 0.77 m above the floor, level, 640x480, fx = fy = 602.3 (vertical FOV 43.4 deg), principal point centred
  LiDAR: depth camera at the Mid-360 position (0.445 m, 0.32 m below the RGB camera), wide FOV; Mid-360-like
         rays (0.94 pts/deg^2 per 0.1 s scan, elevation >= -7 deg, 2 cm range noise, 5 scans) sampled from its depth.
Robot spots: reachable positions with free neighbours (clearance), farthest-point sampled; N views per spot.
Unity (x right, y up, z forward, left-handed) -> map (x = z_u, y = -x_u, z = y_u).

Output <out>/<house_id>/: results/, traj.txt, lidar/frame%06d.npz, camera.json, map_points.npz,
gt_objects.json (type, AABB centre/size in map), gt_rooms.json, source.json

Usage (thor env): python thor_collect.py OUT_ROOT --houses 0 1 2 [--spots 10 --views 6]
"""
import argparse, json, math, os
import numpy as np, cv2
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

W, H, FX = 640, 480, 602.312561
CAM_H, LIDAR_H = 0.77, 0.445
LIDAR_FWD = 0.31 - 0.29061             # lidar is ~2 cm ahead of the camera along the heading
RGB_VFOV = math.degrees(2 * math.atan(H / 2 / FX))
LIDAR_VFOV = 100.0                     # depth camera used to emulate the LiDAR (covers the RGB frustum + -7 deg limit)
DENSITY, N_SCANS, NOISE = 0.94, 5, 0.02
STRUCT = {'Floor', 'Wall', 'Ceiling', 'Doorway', 'Doorframe', 'Window'}


def u2m(p):
    return np.array([p['z'], -p['x'], p['y']], dtype=np.float64)


def T_map_cam(x, y, yaw, h):
    """Camera optical frame (x right, y down, z forward) at map (x, y, h) looking along yaw (map frame)."""
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0]); r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    T = np.eye(4); T[:3, :3] = np.c_[r, [0, 0, -1.0], f]; T[:3, 3] = [x, y, h]
    return T


def unity_yaw_deg(yaw):
    # map forward (cos, sin, 0) = (z_u, -x_u)  ->  unity forward (x_u, z_u) = (-sin, cos)  ->  unity yaw = atan2(x_u, z_u)
    return math.degrees(math.atan2(-math.sin(yaw), math.cos(yaw)))


def lidar_points(depth, vfov_deg, rng):
    """Mid-360-like sampling of a (W x H) z-depth image of a level camera; returns points in that camera's optical frame."""
    fy = (H / 2) / math.tan(math.radians(vfov_deg) / 2); fx = fy
    hfov = 2 * math.atan((W / 2) / fx); vf = math.radians(vfov_deg)
    solid_deg2 = math.degrees(hfov) * math.degrees(vf)
    n = int(DENSITY * solid_deg2 * N_SCANS)
    az = rng.uniform(-hfov / 2, hfov / 2, n); el = rng.uniform(-vf / 2, vf / 2, n)
    el_ok = el >= math.radians(-7.0)                       # Mid-360 lower FOV limit (level sensor)
    az, el = az[el_ok], el[el_ok]
    d = np.c_[np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)]   # optical frame directions
    u = (fx * d[:, 0] / d[:, 2] + W / 2).astype(int); v = (fy * d[:, 1] / d[:, 2] + H / 2).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d, u, v = d[ok], u[ok], v[ok]
    z = depth[v, u]
    good = (z > 0.1) & (z < 20)
    rng_ = z[good] / d[good, 2] + rng.normal(0, NOISE, good.sum())       # range along the ray + noise
    return d[good] * rng_[:, None]


def free_spots(positions, step=0.25, n=10, clear_steps=2):
    P = np.array([[p['x'], p['z']] for p in positions])
    key = set(map(tuple, np.round(P / step).astype(int)))
    ok = [all((i + a, j + b) in key for a in range(-clear_steps, clear_steps + 1) for b in range(-clear_steps, clear_steps + 1))
          for i, j in np.round(P / step).astype(int)]
    P = P[np.array(ok)]
    if len(P) == 0:
        return []
    xy = np.c_[P[:, 1], -P[:, 0]]                                         # unity (x, z) -> map (x, y)
    spots = [xy[np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1))]]
    while len(spots) < min(n, len(xy)):
        d = np.min([np.linalg.norm(xy - s, axis=1) for s in spots], axis=0)
        if d.max() < 1.0:
            break
        spots.append(xy[int(np.argmax(d))])
    return spots


def collect_house(c, house, hid, out, spots_n, views, rng):
    c.reset(scene=house)
    reach = c.step(action='GetReachablePositions').metadata['actionReturn']
    spots = free_spots(reach, n=spots_n)
    if not spots:
        return 0
    sd = os.path.join(out, f'procthor_{hid:05d}')
    for d in ('results', 'lidar'):
        os.makedirs(os.path.join(sd, d), exist_ok=True)
    c.step(action='AddThirdPartyCamera', position=dict(x=0, y=1, z=0), rotation=dict(x=0, y=0, z=0), fieldOfView=RGB_VFOV)
    c.step(action='AddThirdPartyCamera', position=dict(x=0, y=1, z=0), rotation=dict(x=0, y=0, z=0), fieldOfView=LIDAR_VFOV)
    T_cam_lidarcam = None
    traj, mp, mo, k = [], [], [], 0
    for sx, sy in spots:
        for vi in range(views):
            yaw = 2 * math.pi * vi / views
            Tc = T_map_cam(sx, sy, yaw, CAM_H)
            lx, ly = sx + LIDAR_FWD * math.cos(yaw), sy + LIDAR_FWD * math.sin(yaw)
            Tl = T_map_cam(lx, ly, yaw, LIDAR_H)
            for cam_id, (x, y, h) in enumerate([(sx, sy, CAM_H), (lx, ly, LIDAR_H)]):
                ev = c.step(action='UpdateThirdPartyCamera', thirdPartyCameraId=cam_id,
                            position=dict(x=-y, y=h, z=x), rotation=dict(x=0, y=unity_yaw_deg(yaw), z=0))
            rgb = ev.third_party_camera_frames[0]; depth_l = ev.third_party_depth_frames[1]
            cv2.imwrite(os.path.join(sd, 'results', f'frame{k:06d}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(sd, 'results', f'depth{k:06d}.png'),
                        np.clip(ev.third_party_depth_frames[0] * 6553.5, 0, 65535).astype(np.uint16))
            pts = lidar_points(depth_l, LIDAR_VFOV, rng).astype(np.float32)          # lidar-camera optical frame
            T_cam_sensor = np.linalg.inv(Tc) @ Tl
            np.savez_compressed(os.path.join(sd, 'lidar', f'frame{k:06d}.npz'), points=pts, T_map_sensor=Tl, T_cam_sensor=T_cam_sensor)
            traj.append(Tc.reshape(-1))
            mp.append((Tl[:3, :3] @ pts.T).T + Tl[:3, 3]); mo.append(np.repeat(Tl[None, :3, 3], len(pts), 0))
            k += 1
    np.savetxt(os.path.join(sd, 'traj.txt'), np.array(traj))
    K = [[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]]
    json.dump(dict(K=K, width=W, height=H, distortion=[0] * 5), open(os.path.join(sd, 'camera.json'), 'w'), indent=1)
    mp, mo = np.concatenate(mp), np.concatenate(mo)
    key = np.floor(mp / 0.03).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    np.savez_compressed(os.path.join(sd, 'map_points.npz'), points=mp[first].astype(np.float32), origins=mo[first].astype(np.float32))
    gt = []
    for o in c.last_event.metadata['objects']:
        if o['objectType'] in STRUCT:
            continue
        bb = o.get('axisAlignedBoundingBox') or {}
        cen = u2m(bb['center']) if bb.get('center') else u2m(o['position'])
        sz = bb.get('size') or dict(x=0, y=0, z=0)
        gt.append(dict(id=o['objectId'], gt_class=o['objectType'], position=cen.round(4).tolist(),
                       size=[round(sz['z'], 4), round(sz['x'], 4), round(sz['y'], 4)],
                       parent=(o.get('parentReceptacles') or [None])[0]))
    json.dump(gt, open(os.path.join(sd, 'gt_objects.json'), 'w'), indent=1)
    rooms = [dict(id=r['id'], room_type=r['roomType'],
                  polygon=[[p['z'], -p['x']] for p in r['floorPolygon']]) for r in house['rooms']]
    json.dump(rooms, open(os.path.join(sd, 'gt_rooms.json'), 'w'), indent=1)
    json.dump(dict(source='procthor_10k', house_index=hid, frame_id='map', floor_z=0.0, rooms=len(rooms),
                   spots=len(spots), views=views, keyframes=k,
                   lidar_input=f'Mid-360 emulated from a depth camera at the LiDAR mount ({N_SCANS} scans, {NOISE} m noise, el >= -7 deg)',
                   pose_source='ground truth (no localization error)',
                   camera='640x480, fx=fy=602.3, principal point at image centre (sim camera)'),
              open(os.path.join(sd, 'source.json'), 'w'), indent=1)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--houses', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--spots', type=int, default=10); ap.add_argument('--views', type=int, default=6)
    ap.add_argument('--min-rooms', type=int, default=3)
    ap.add_argument('--max-houses', type=int, default=0, help='stop after this many accepted houses (0 = all listed)')
    args = ap.parse_args()
    d = prior.load_dataset('procthor-10k')['train']
    c = Controller(platform=CloudRendering, width=W, height=H, renderDepthImage=True, quality='High')
    rng = np.random.default_rng(0)
    accepted = 0
    for hid in args.houses:
        if args.max_houses and accepted >= args.max_houses:
            break
        house = d[hid]
        if len(house['rooms']) < args.min_rooms:
            print(f'house {hid}: {len(house["rooms"])} rooms < {args.min_rooms}, skipped', flush=True); continue
        k = collect_house(c, house, hid, args.out, args.spots, args.views, rng)
        accepted += k > 0
        print(f'house {hid}: rooms={len(house["rooms"])} keyframes={k}', flush=True)
    c.stop()


if __name__ == '__main__':
    main()
