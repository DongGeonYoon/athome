"""
Real-robot pre-mapping rosbag -> common intermediate format (same layout as the simulation sources), so the
offline Scene Graph features are built with exactly the same tools:
  robot_bag_extract.py -> run_cgd_sources.sh -> build_dataset.py (or ros_replay.sh with the real node)

Needs in the bag: /tf (camera_init -> body), /tf_static or --map-to-cinit, /cloud_registered_body, colour image
(sensor_msgs/Image or CompressedImage) + camera_info; optional aligned depth (16UC1 mm or 32FC1 m) and
/obs_keyframe markers. The camera <- body extrinsic is taken from robot.yaml (calibration), not from TF, because
the robot does not publish a base_link -> camera TF.

Keyframes: /obs_keyframe markers if recorded, otherwise automatic: the robot is nearly still (|v| < 0.15 m/s,
|w| < 0.3 rad/s, no motion blur / sync error) and has moved >= --kf-dist m or turned >= --kf-yaw deg since the
last keyframe.

Usage (system python with ROS sourced):
  python3 robot_bag_extract.py BAG_DIR OUT_DIR [--image-topic ... --depth-topic ... --kf-dist 0.5 --kf-yaw 30]
"""
import argparse, json, os
import numpy as np, cv2, yaml
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation as Rot
from gazebo_extract import PoseSeries, read_bag, tf_mat, stamp

ROBOT_YAML = os.path.expanduser('~/semantic_geometry_ws/install/semantic_geometry_fusion/share/semantic_geometry_fusion/config/robot.yaml')
T_BODY_BASE_DEFAULT = np.eye(4); T_BODY_BASE_DEFAULT[:3, 3] = [-0.321, -0.023, -0.401]   # mppi_nav static TF


def decode_image(m):
    if hasattr(m, 'format'):                                   # CompressedImage
        return cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding.lower() == 'rgb8' else img.copy()


def decode_depth_m(m):
    if m.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width) * 0.001
    return np.nan_to_num(np.frombuffer(m.data, np.float32).reshape(m.height, m.width), nan=0, posinf=0, neginf=0)


def auto_keyframes(body, dist, yaw_deg, gap=1.0, max_speed=0.15, max_turn=0.3):
    """Stamps where the robot is slow enough (|v| <= max_speed m/s, |w| <= max_turn rad/s: no blur, small sync
    error) and has moved/turned enough since the previous keyframe."""
    out, last = [], None
    for t in np.arange(body.t[0] + 0.5, body.t[-1] - 0.5, 0.1):
        a, b = body.at(t - 0.25), body.at(t + 0.25)
        v = np.linalg.norm(b[:2, 3] - a[:2, 3]) / 0.5
        w = abs(Rot.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()[2]) / 0.5
        if v > max_speed or w > max_turn:
            continue
        T = body.at(t)
        if last is not None:
            moved = np.linalg.norm(T[:2, 3] - last[1][:2, 3])
            turned = np.degrees(abs(Rot.from_matrix(last[1][:3, :3].T @ T[:3, :3]).as_rotvec()[2]))
            if t - last[0] < gap or (moved < dist and turned < yaw_deg):
                continue
        out.append(t); last = (t, T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('out')
    ap.add_argument('--image-topic', default='/camera/camera/color/image_raw/compressed')
    ap.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    ap.add_argument('--depth-topic', default='/camera/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--cloud-topic', default='/cloud_registered_body')
    ap.add_argument('--robot-yaml', default=ROBOT_YAML)
    ap.add_argument('--window', type=float, default=0.5)
    ap.add_argument('--kf-dist', type=float, default=0.5); ap.add_argument('--kf-yaw', type=float, default=30.0)
    ap.add_argument('--name', default='robot_premap')
    ap.add_argument('--max-speed', type=float, default=0.15, help='keyframes only below this speed [m/s]')
    ap.add_argument('--max-turn', type=float, default=0.3, help='keyframes only below this turn rate [rad/s]')
    ap.add_argument('--kf-gap', type=float, default=1.0, help='minimum seconds between keyframes')
    ap.add_argument('--ignore-markers', action='store_true', help='select keyframes automatically even if /obs_keyframe exists')
    args = ap.parse_args()
    T_cam_body = np.array(yaml.safe_load(open(args.robot_yaml))['geometry_fusion_node']['ros__parameters']['T_cam_lidar']).reshape(4, 4)

    # pass 1: poses, calibration, markers, cloud stamps (images and clouds are large: read them in pass 2)
    statics, body, mcin, kfs, info, cl_t = {}, PoseSeries(), PoseSeries(), [], None, []
    for topic, m in read_bag(args.bag, ['/tf', '/tf_static', '/obs_keyframe', args.info_topic, args.cloud_topic]):
        if topic in ('/tf_static', '/tf'):
            for t in m.transforms:
                if topic == '/tf_static':
                    statics[(t.header.frame_id, t.child_frame_id)] = tf_mat(t)
                elif t.header.frame_id == 'camera_init' and t.child_frame_id == 'body':
                    body.add(stamp(t.header), tf_mat(t))
                elif t.header.frame_id == 'map' and t.child_frame_id == 'camera_init':   # localization correction
                    mcin.add(stamp(t.header), tf_mat(t))
        elif topic == '/obs_keyframe' and not args.ignore_markers:
            kfs.append(json.loads(m.data)['stamp'])
        elif topic == args.info_topic and info is None:
            info = m
        elif topic == args.cloud_topic:
            cl_t.append(stamp(m.header))
    if info is None or not body.t or not cl_t:
        raise SystemExit(f'missing data: camera_info={info is not None} camera_init->body TF={len(body.t)} clouds={len(cl_t)}')
    body.finalize()
    if mcin.t:                       # FAST_LIO_LOCALIZATION2: map -> camera_init changes with every ICP correction
        mcin.finalize(); M = mcin.at
    else:
        M = lambda t, T0=statics.get(('map', 'camera_init'), np.eye(4)): T0
    T_body_base = statics.get(('body', 'base_link'), T_BODY_BASE_DEFAULT)
    kf_source = '/obs_keyframe markers' if kfs else (f'automatic (|v| <= {args.max_speed} m/s, |w| <= {args.max_turn} rad/s, '
                                                     f'>= {args.kf_dist} m or {args.kf_yaw} deg, >= {args.kf_gap} s apart)')
    if not kfs:
        kfs = auto_keyframes(body, args.kf_dist, args.kf_yaw, args.kf_gap, args.max_speed, args.max_turn)
    kfs = np.array(sorted(kfs)); cl_t = np.array(cl_t)
    print(f'pass 1: {len(body.t)} poses, {len(cl_t)} clouds, {len(kfs)} keyframes ({kf_source})', flush=True)

    # pass 2: nearest image / depth per keyframe, clouds in each keyframe window, thinned map points
    best_img, best_depth, kf_clouds = {}, {}, {}
    need = lambda t, tol: int(np.argmin(np.abs(kfs - t))) if len(kfs) and np.min(np.abs(kfs - t)) <= tol else None
    chunk, kept, n_cloud = [], [], 0

    def thin(arrs):                                                # 3 cm voxels, first point wins
        a = np.concatenate(arrs); _, first = np.unique(np.floor(a[:, :3] / 0.03).astype(np.int64), axis=0, return_index=True)
        return a[first]
    for topic, m in read_bag(args.bag, [args.image_topic, args.depth_topic, args.cloud_topic]):
        if topic == args.image_topic:
            t = stamp(m.header); k = need(t, 0.3)
            if k is not None and (k not in best_img or abs(t - kfs[k]) < abs(best_img[k][0] - kfs[k])):
                best_img[k] = (t, decode_image(m))
        elif topic == args.depth_topic:
            t = stamp(m.header); k = need(t, 0.3)
            if k is not None and (k not in best_depth or abs(t - kfs[k]) < abs(best_depth[k][0] - kfs[k])):
                best_depth[k] = (t, decode_depth_m(m))
        elif topic == args.cloud_topic:
            t = stamp(m.header); rec = point_cloud2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)
            p = np.stack([rec['x'], rec['y'], rec['z']], -1).astype(np.float64)
            for k in np.flatnonzero(np.abs(kfs - t) <= args.window / 2 + 0.3):
                kf_clouds.setdefault(int(k), []).append((t, p))
            n_cloud += 1
            if n_cloud % 2 == 0:                                   # map points for occupancy, 3 cm voxels
                T = M(t) @ body.at(t)
                chunk.append(np.c_[(T[:3, :3] @ p.T).T + T[:3, 3], np.repeat(T[None, :3, 3], len(p), 0)])
                if len(chunk) >= 50:
                    kept.append(thin(chunk)); chunk = []

    for d in ('results', 'lidar'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    K = np.array(info.k).reshape(3, 3); W, H = info.width, info.height
    traj, stamps, base_z, n_depth = [], [], [], 0
    for k in range(len(kfs)):
        if k not in best_img:
            continue
        t_img, img = best_img[k]; idx = len(traj)
        cv2.imwrite(os.path.join(args.out, 'results', f'frame{idx:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if k in best_depth and best_depth[k][1].shape == (H, W):
            dimg = np.clip(best_depth[k][1] * 6553.5, 0, 65535).astype(np.uint16); n_depth += 1
        else:
            dimg = np.zeros((H, W), np.uint16)
        cv2.imwrite(os.path.join(args.out, 'results', f'depth{idx:06d}.png'), dimg)
        T_map_body = M(t_img) @ body.at(t_img)
        traj.append((T_map_body @ np.linalg.inv(T_cam_body)).reshape(-1))
        Tinv = np.linalg.inv(T_map_body)
        pts = [(Tinv @ M(tc) @ body.at(tc) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
               for tc, p in kf_clouds.get(k, []) if abs(tc - t_img) <= args.window / 2]
        pts = np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32)
        np.savez_compressed(os.path.join(args.out, 'lidar', f'frame{idx:06d}.npz'), points=pts,
                            T_map_sensor=T_map_body, T_cam_sensor=T_cam_body, stamp=t_img)
        stamps.append(t_img); base_z.append((T_map_body @ T_body_base)[2, 3])
    if not traj:
        raise SystemExit('no keyframe has an image within 0.3 s: check --image-topic')
    np.savetxt(os.path.join(args.out, 'traj.txt'), np.array(traj))
    with open(os.path.join(args.out, 'keyframe_stamps.csv'), 'w') as f:
        f.write('filename,timestamp_sec\n')
        f.writelines(f'frame{i:06d}.pkl.gz,{t:.6f}\n' for i, t in enumerate(stamps))
    json.dump(dict(K=K.tolist(), width=W, height=H, distortion=list(info.d)), open(os.path.join(args.out, 'camera.json'), 'w'), indent=1)
    allp = thin(kept + chunk) if kept or chunk else np.zeros((0, 6))
    pm, po = allp[:, :3].astype(np.float32), allp[:, 3:].astype(np.float32)
    np.savez_compressed(os.path.join(args.out, 'map_points.npz'), points=pm, origins=po)
    floor_z = float(np.median(base_z))                           # base_link origin is on the floor (REP-105)
    json.dump(dict(source='robot', name=args.name, frame_id='map', floor_z=floor_z,
                   floor_note='median base_link z (base_link origin = base centre on the floor)',
                   lidar_input=f'FAST-LIO {args.cloud_topic}', accumulation_s=args.window,
                   pose_source='FAST-LIO / localization TF (map -> camera_init -> body)', keyframe_selection=kf_source,
                   T_cam_body_source=args.robot_yaml, keyframes=len(traj), keyframes_with_depth=n_depth,
                   depth_input='aligned D435i depth (fusion depth fallback only)' if n_depth else 'none recorded'),
              open(os.path.join(args.out, 'source.json'), 'w'), indent=1)
    print(f'keyframes {len(traj)} | lidar pts/frame median {np.median([len(np.load(os.path.join(args.out, "lidar", f"frame{i:06d}.npz"))["points"]) for i in range(len(traj))]):.0f} '
          f'| depth {n_depth}/{len(traj)} | map points {len(pm)} | floor z {floor_z:.3f}')


if __name__ == '__main__':
    main()
