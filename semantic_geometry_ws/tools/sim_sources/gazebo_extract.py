"""
Gazebo episode rosbag -> common intermediate format (same layout for every simulation source).

Uses only what the real robot has at runtime: camera image + camera_info, FAST-LIO /cloud_registered_body
and TF (map -> camera_init -> body -> base_link -> camera). Gazebo ground truth (/gt_odom, world model
poses) is written separately for evaluation only.

Output <out>/:
  results/frame%06d.jpg, depth%06d.png (aligned D435i depth if recorded, else zeros = CG-D loader placeholder),
  traj.txt (T_map_cam)
  lidar/frame%06d.npz   points (body frame at keyframe time, 0.5 s accumulated, each scan compensated with
                        its own TF), T_map_sensor, T_cam_sensor
  camera.json, map_points.npz (FAST-LIO body clouds in map + sensor origins), gt_objects.json,
  gt_robot_poses.json (GT base pose per keyframe, map frame), source.json

Usage (system python with ROS sourced):
  python3 gazebo_extract.py BAG_DIR GT_MODELS.json OUT_DIR [--window 0.5]
"""
import argparse, bisect, collections, json, os
import numpy as np, cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation as Rot, Slerp

T_BASE_IMU = np.eye(4); T_BASE_IMU[:3, 3] = [0.321, 0.023, 0.401]   # URDF imu_link (= FAST-LIO body)
GT_SKIP = ('Floor', 'HouseWall', 'RoomWall', 'RoomCeiling', 'RoomWindow', 'Light', 'Chandelier', 'Aircon', 'SecurityCamera', 'Handle')
GT_CLASS = {'Bed': 'bed', 'NightStand': 'nightstand', 'Chair': 'chair', 'Sofa': 'sofa', 'CoffeeTable': 'coffee table',
            'KitchenTable': 'dining table', 'BalconyTable': 'table', 'ReadingDesk': 'desk', 'CookingBench': 'kitchen counter',
            'KitchenCabinet': 'kitchen cabinet', 'Refrigerator': 'refrigerator', 'TVCabinet': 'tv cabinet', 'TV': 'television',
            'Trash': 'trash can', 'ShoeRack': 'shoe rack', 'FitnessEquipment': 'exercise machine', 'Dumbbell': 'dumbbell',
            'Ball': 'ball', 'Carpet': 'carpet', 'Curtain': 'curtain', 'FoldingDoor': 'door', 'Door': 'door',
            'DeskPortrait': 'picture frame', 'Portrait': 'picture frame', 'Tablet': 'tablet', 'Tableware': 'tableware',
            'Wardrobe': 'wardrobe', 'Rangehood': 'range hood', 'SeasoningBox': 'box', 'KitchenUtensils': 'kitchen utensils',
            'Board': 'board'}


def tf_mat(t):
    T = np.eye(4); q = t.transform.rotation; p = t.transform.translation
    T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix(); T[:3, 3] = [p.x, p.y, p.z]
    return T


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class PoseSeries:
    def __init__(self):
        self.t, self.T = [], []

    def add(self, t, T):
        self.t.append(t); self.T.append(T)

    def finalize(self):
        t = np.array(self.t); o = np.argsort(t, kind='stable')
        keep = np.r_[t[o][1:] > t[o][:-1], True]          # repeated stamps (sim clock steps): keep the last sample
        self.t = t[o][keep]; self.T = np.array(self.T)[o][keep]
        self.slerp = Slerp(self.t, Rot.from_matrix(self.T[:, :3, :3]))

    def at(self, t):
        t = float(np.clip(t, self.t[0], self.t[-1]))
        i = min(max(bisect.bisect_left(self.t, t), 1), len(self.t) - 1)
        a = (t - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9)
        T = np.eye(4); T[:3, :3] = self.slerp([t]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.T[i - 1][:3, 3] + a * self.T[i][:3, 3]
        return T


def read_bag(path, topics=None):
    """Yield (topic, msg). A file-compressed bag is decompressed next to itself, so never read the same
    compressed bag from two processes at once. `topics` limits deserialisation to the listed topics."""
    meta = open(os.path.join(path, 'metadata.yaml')).read()
    reader = rosbag2_py.SequentialCompressionReader() if 'compression_mode: FILE' in meta else rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, get_message(types[topic]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag'); ap.add_argument('gt_models'); ap.add_argument('out')
    ap.add_argument('--window', type=float, default=0.5)
    args = ap.parse_args()
    statics, body, gt_odom = {}, PoseSeries(), PoseSeries()
    kfs, images, clouds, info = [], [], [], None
    DEPTH = '/camera/camera/aligned_depth_to_color/depth/image_raw'
    recent_depth, depths = collections.deque(maxlen=8), []      # depth is large: keep only frames near keyframes
    for topic, m in read_bag(args.bag):
        if topic == '/tf_static':
            for t in m.transforms:
                statics[(t.header.frame_id, t.child_frame_id)] = tf_mat(t)
        elif topic == '/tf':
            for t in m.transforms:
                if t.header.frame_id == 'camera_init' and t.child_frame_id == 'body':
                    body.add(stamp(t.header), tf_mat(t))
        elif topic == '/obs_keyframe':
            kfs.append(json.loads(m.data))
            kept = {t for t, _ in depths}
            depths += [d for d in recent_depth if abs(d[0] - kfs[-1]['stamp']) <= 1.0 and d[0] not in kept]
        elif topic == DEPTH:
            d = (stamp(m.header), np.nan_to_num(np.frombuffer(m.data, np.float32).reshape(m.height, m.width), nan=0, posinf=0, neginf=0))
            recent_depth.append(d)
            if kfs and abs(d[0] - kfs[-1]['stamp']) <= 1.0:
                depths.append(d)
        elif topic == '/camera/camera/d435i_color/image_raw/compressed':
            images.append((stamp(m.header), m))
        elif topic == '/cloud_registered_body':
            rec = point_cloud2.read_points(m, field_names=('x', 'y', 'z'), skip_nans=True)
            clouds.append((stamp(m.header), np.stack([rec['x'], rec['y'], rec['z']], -1).astype(np.float32)))
        elif topic == '/camera/camera/color/camera_info' and info is None:
            info = m
        elif topic == '/gt_odom':
            p = m.pose.pose; T = np.eye(4)
            T[:3, :3] = Rot.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_matrix()
            T[:3, 3] = [p.position.x, p.position.y, p.position.z]
            gt_odom.add(stamp(m.header), T)
    body.finalize(); gt_odom.finalize()
    T_map_cinit = statics.get(('map', 'camera_init'), np.eye(4))
    T_body_base = statics[('body', 'base_link')]
    T_base_cam = statics[('base_link', 'camera_link')] @ statics[('camera_link', 'camera_color_optical_frame')]
    T_body_cam = T_body_base @ T_base_cam
    T_cam_body = np.linalg.inv(T_body_cam)
    K = np.array(info.k).reshape(3, 3); W, H = info.width, info.height

    for d in ('results', 'lidar'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    img_t = np.array([t for t, _ in images]); cl_t = np.array([t for t, _ in clouds])
    traj, gt_poses, n_depth = [], [], 0
    # world -> map: FAST-LIO camera_init = IMU frame when FAST-LIO started (robot still at spawn)
    T_world_cinit = gt_odom.at(body.t[0]) @ T_BASE_IMU
    T_map_world = T_map_cinit @ np.linalg.inv(T_world_cinit)
    for k, kf in enumerate(kfs):
        t = kf['stamp']
        j = int(np.argmin(np.abs(img_t - t)))
        if abs(img_t[j] - t) > 0.3:
            continue
        idx = len(traj)
        img = cv2.imdecode(np.frombuffer(images[j][1].data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(args.out, 'results', f'frame{idx:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dj = int(np.argmin([abs(dt - img_t[j]) for dt, _ in depths])) if depths else -1
        if dj >= 0 and abs(depths[dj][0] - img_t[j]) <= 0.3:       # robot is stopped at keyframes
            dimg = np.clip(depths[dj][1] * 6553.5, 0, 65535).astype(np.uint16); n_depth += 1
        else:
            dimg = np.zeros((H, W), np.uint16)
        cv2.imwrite(os.path.join(args.out, 'results', f'depth{idx:06d}.png'), dimg)
        t_img = img_t[j]
        T_map_body = T_map_cinit @ body.at(t_img)
        traj.append((T_map_body @ T_body_cam).reshape(-1))
        near = np.flatnonzero(np.abs(cl_t - t_img) <= args.window / 2)
        pts = [ (np.linalg.inv(T_map_body) @ T_map_cinit @ body.at(cl_t[i]) @ np.c_[clouds[i][1], np.ones(len(clouds[i][1]))].T).T[:, :3]
                for i in near ]
        pts = np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32)
        np.savez_compressed(os.path.join(args.out, 'lidar', f'frame{idx:06d}.npz'), points=pts,
                            T_map_sensor=T_map_body, T_cam_sensor=T_cam_body, stamp=t_img)
        T_map_base_gt = T_map_world @ gt_odom.at(t_img)
        gt_poses.append(dict(frame=idx, spot=kf['spot'], view=kf['view'], T_map_base_gt=T_map_base_gt.tolist(),
                             T_map_base_est=(T_map_body @ T_body_base).tolist()))
    np.savetxt(os.path.join(args.out, 'traj.txt'), np.array(traj))
    with open(os.path.join(args.out, 'keyframe_stamps.csv'), 'w') as f:      # for ROS replay (gsa_pkl_bridge format)
        f.write('filename,timestamp_sec\n')
        for i in range(len(traj)):
            f.write(f'frame{i:06d}.pkl.gz,{float(np.load(os.path.join(args.out, "lidar", f"frame{i:06d}.npz"))["stamp"]):.6f}\n')
    json.dump(dict(K=K.tolist(), width=W, height=H, distortion=list(info.d)), open(os.path.join(args.out, 'camera.json'), 'w'), indent=1)
    # map points for occupancy: every body cloud in map frame (+ sensor origins), voxel-thinned
    mp, mo = [], []
    for t, p in clouds[::2]:
        T = T_map_cinit @ body.at(t)
        mp.append((T[:3, :3] @ p.T).T + T[:3, 3]); mo.append(np.repeat(T[None, :3, 3], len(p), 0))
    mp, mo = np.concatenate(mp), np.concatenate(mo)
    key = np.floor(mp / 0.03).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    np.savez_compressed(os.path.join(args.out, 'map_points.npz'), points=mp[first].astype(np.float32), origins=mo[first].astype(np.float32))
    gt = []
    for g in json.load(open(args.gt_models)):
        if any(s in g['model'] for s in GT_SKIP):
            continue
        cls = next((v for k, v in GT_CLASS.items() if g['model'].startswith(k)), g['model'].split('_')[0].lower())
        Tw = np.eye(4); Tw[:3, :3] = Rot.from_euler('xyz', g['pose'][3:6]).as_matrix(); Tw[:3, 3] = g['pose'][:3]
        Tm = T_map_world @ Tw
        gt.append(dict(id=g['name'], gt_class=cls, position=Tm[:3, 3].round(4).tolist(), note='model origin (usually base centre)'))
    json.dump(gt, open(os.path.join(args.out, 'gt_objects.json'), 'w'), indent=1)
    json.dump(gt_poses, open(os.path.join(args.out, 'gt_robot_poses.json'), 'w'))
    floor_z = float((T_map_world @ np.array([0, 0, 0, 1.0]))[2])
    json.dump(dict(source='gazebo_aws_small_house', world='AWS RoboMaker small house (multi-room)', frame_id='map',
                   map_frame='FAST-LIO camera_init (IMU at start) with identity map->camera_init', floor_z=floor_z,
                   lidar_input='FAST-LIO /cloud_registered_body (point_filter_num 3, blind 0.5 m)', accumulation_s=args.window,
                   pose_source='FAST-LIO TF (estimated); ground truth only in gt_robot_poses.json',
                   keyframes=len(traj), keyframes_with_depth=n_depth,
                   depth_input='aligned D435i depth (fusion depth fallback only)' if n_depth else 'none recorded'), open(os.path.join(args.out, 'source.json'), 'w'), indent=1)
    err = [np.linalg.norm(np.array(p['T_map_base_gt'])[:2, 3] - np.array(p['T_map_base_est'])[:2, 3]) for p in gt_poses]
    print(f'keyframes {len(traj)} | lidar pts/frame median {np.median([len(np.load(os.path.join(args.out, "lidar", f"frame{i:06d}.npz"))["points"]) for i in range(len(traj))]):.0f} '
          f'| FAST-LIO xy error vs GT: median {np.median(err):.3f} m, max {np.max(err):.3f} m | depth {n_depth}/{len(traj)} | gt objects {len(gt)} | floor z {floor_z:.3f}')


if __name__ == '__main__':
    main()
