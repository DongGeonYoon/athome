"""Overlay LiDAR points (colored by depth) on the nearest camera image from a rosbag2.

Usage: python3 tools/bag_check/projection_overlay.py BAG_DIR OUT.png [--frame N]
Uses T_cam_lidar from config/fusion_params.yaml and K from the bag's CameraInfo.
"""
import argparse, os, sys
import numpy as np, cv2, yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../src/semantic_geometry_fusion'))
from semantic_geometry_fusion.geometry import project_lidar_to_image

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('out'); ap.add_argument('--frame', type=int, default=30)
ap.add_argument('--image-topic', default='/camera/camera/color/image_raw')
ap.add_argument('--info-topic', default='/camera/camera/color/camera_info')
ap.add_argument('--lidar-topic', default='/livox/lidar')
args = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join(HERE, '../../src/semantic_geometry_fusion/config/fusion_params.yaml')))
T_cam_lidar = np.array(cfg['geometry_fusion_node']['ros__parameters']['T_cam_lidar']).reshape(4, 4)

reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
images, clouds, info = [], [], None
while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic not in (args.image_topic, args.info_topic, args.lidar_topic):
        continue
    msg = deserialize_message(data, get_message(types[topic]))
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if topic == args.image_topic: images.append((t, msg))
    elif topic == args.lidar_topic: clouds.append((t, msg))
    elif info is None: info = msg

t_img, img_msg = images[min(args.frame, len(images) - 1)]
t_cloud, cloud = min(clouds, key=lambda c: abs(c[0] - t_img))
K = np.array(info.k).reshape(3, 3)
img = np.frombuffer(img_msg.data, np.uint8).reshape(img_msg.height, img_msg.width, -1).copy()
if img_msg.encoding == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
rec = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
pts = np.stack([rec['x'], rec['y'], rec['z']], -1).astype(np.float64)

uv, z, _ = project_lidar_to_image(pts, T_cam_lidar, K)
ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
col = cv2.applyColorMap(np.clip((z[ok] - 0.3) / 5.0 * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
for (u, v), c in zip(uv[ok].astype(int), col):
    cv2.circle(img, (u, v), 2, tuple(int(x) for x in c), -1)
cv2.imwrite(args.out, img)
print(f'image {img_msg.width}x{img_msg.height} enc={img_msg.encoding} | K fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} | '
      f'D={list(np.round(info.d, 4))} | cloud frame={cloud.header.frame_id} fields={[f.name for f in cloud.fields]}')
print(f'dt(camera-lidar)={t_img - t_cloud:+.4f}s | scan pts={len(pts)} in-image={ok.sum()} | '
      f'#images={len(images)} #clouds={len(clouds)} img-rate={len(images)/(images[-1][0]-images[0][0]):.1f}Hz '
      f'lidar-rate={len(clouds)/(clouds[-1][0]-clouds[0][0]):.1f}Hz')
dts = [min(abs(t - c[0]) for c in clouds) for t, _ in images]
print(f'nearest-lidar |dt| over all images: median={np.median(dts)*1e3:.1f}ms max={np.max(dts)*1e3:.1f}ms')
