#!/usr/bin/env python3
"""Camera-LiDAR timing on the robot: for each colour frame, |stamp - nearest /cloud_registered_body stamp|, and the
receive latency of both. Prints a recommended max_sync_dt_sec (p99 of the nearest-scan gap + 20 ms, at least 0.06 s).
  python3 check_sync.py [SECONDS=15] [OUT_JSON]"""
import json, sys, time, bisect
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2

st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
rclpy.init(); n = Node('check_sync'); img, cl, lat = [], [], {'image': [], 'cloud': []}
n.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', lambda m: (img.append(st(m.header)), lat['image'].append(time.time() - st(m.header))), qos_profile_sensor_data)
n.create_subscription(PointCloud2, '/cloud_registered_body', lambda m: (cl.append(st(m.header)), lat['cloud'].append(time.time() - st(m.header))), 20)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0; t0 = time.time()
while time.time() - t0 < T:
    rclpy.spin_once(n, timeout_sec=0.05)
cl = sorted(cl); gaps = []
for t in img:
    i = bisect.bisect_left(cl, t); c = [cl[j] for j in (i - 1, i) if 0 <= j < len(cl)]
    if c: gaps.append(min(abs(t - x) for x in c))
res = dict(seconds=T, image_hz=len(img) / T, cloud_hz=len(cl) / T)
if gaps:
    g = np.array(gaps); res.update(nearest_gap_median=float(np.median(g)), nearest_gap_p99=float(np.percentile(g, 99)), nearest_gap_max=float(g.max()),
                                  recommended_max_sync_dt_sec=round(max(0.06, float(np.percentile(g, 99)) + 0.02), 3))
for k, v in lat.items():
    if v: res[f'{k}_receive_latency_median'] = float(np.median(v))
print(json.dumps(res, indent=1))
if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], 'w'), indent=1)
