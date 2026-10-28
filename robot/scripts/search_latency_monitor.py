#!/usr/bin/env python3
"""
Record what the search server would see from live CG-D: receive time and image stamp of every /cg_d/objects_2d, every
camera image (clock offset of the replay), and the odometry yaw (camera_init -> body) for turn rates. JSON on Ctrl-C.
  python3 search_latency_monitor.py OUT.json [--image-topic /camera/camera/color/image_raw/compressed]
"""
import argparse, json, math, signal, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from tf2_msgs.msg import TFMessage
from semantic_geometry_msgs.msg import SemanticObject2DArray

st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--image-topic', default='/camera/camera/color/image_raw/compressed')
    a = ap.parse_args(); rclpy.init(); n = Node('search_latency_monitor')
    rec = dict(results=[], images=[], yaw=[])
    n.create_subscription(SemanticObject2DArray, '/cg_d/objects_2d', lambda m: rec['results'].append((time.time(), st(m.header), len(m.objects))), 10)
    n.create_subscription(CompressedImage if a.image_topic.endswith('compressed') else Image, a.image_topic,
                          lambda m: rec['images'].append((time.time(), st(m.header))), qos_profile_sensor_data)
    def tf(m):
        for t in m.transforms:
            if t.header.frame_id == 'camera_init' and t.child_frame_id == 'body':
                q = t.transform.rotation
                rec['yaw'].append((st(t.header), math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
    n.create_subscription(TFMessage, '/tf', tf, 100)
    stop = {'v': False}; signal.signal(signal.SIGINT, lambda *_: stop.update(v=True)); signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    while not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.1)
    json.dump(rec, open(a.out, 'w'))


if __name__ == '__main__':
    main()
