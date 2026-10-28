#!/usr/bin/env python3
"""
Tag check viewer (robot PC, plain python3 + ROS): newest camera image -> tag_check_server.py on the GPU server
(SSH tunnel) -> /tag_check/annotated (boxes + "label confidence") for rqt_image_view. Labels only, nothing for fusion.
Only the newest image is kept (depth-1 subscription), so the view is never behind the camera by more than one frame.
Reconnects when the server restarts. Log line per frame; --save-dir also writes tags.jsonl.
  python3 tag_check_node.py [--image-topic /camera/camera/color/image_raw] [--server 127.0.0.1:47001] [--save-dir DIR]
"""
import argparse, json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgd_wire as W

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage


class TagCheckNode(Node):
    def __init__(self, a):
        super().__init__('tag_check')
        self.a, self.latest, self.sock, self.n = a, None, None, 0
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        compressed = a.image_topic.endswith('/compressed')
        self.create_subscription(CompressedImage if compressed else Image, a.image_topic, self.on_image, newest)
        self.pub = self.create_publisher(Image, '/tag_check/annotated', 2)
        self.get_logger().info(f'tag check: {a.image_topic} -> {a.server} -> /tag_check/annotated')

    def on_image(self, msg):
        self.latest = msg

    @staticmethod
    def decode(msg):
        if isinstance(msg, CompressedImage):
            return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding.lower() == 'rgb8' else img.copy()

    def step(self):
        msg, self.latest = self.latest, None
        t0 = time.time()
        ok, png = cv2.imencode('.png', self.decode(msg), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        try:
            if self.sock is None:
                self.sock = W.connect(self.a.server, timeout=30)
            W.send(self.sock, dict(id=self.n), [png.tobytes()])
            h, (ann,) = W.recv(self.sock)
        except (OSError, ConnectionError) as e:
            self.get_logger().warning(f'server not reachable ({e}), retrying', throttle_duration_sec=5.0)
            self.sock = None; time.sleep(1.0); return
        out = cv2.imdecode(np.frombuffer(ann, np.uint8), cv2.IMREAD_COLOR)
        m = Image(); m.header = msg.header; m.height, m.width = out.shape[:2]; m.encoding = 'bgr8'
        m.step = int(out.shape[1] * 3); m.data = out.tobytes()
        self.pub.publish(m)
        labels = [f"{o['label']} {o['confidence']:.2f}" for o in h['objects']]
        self.get_logger().info(f'{time.time() - t0:.2f} s (server {h["seconds"]:.2f}): {", ".join(labels)}')
        if self.a.save_dir:
            os.makedirs(self.a.save_dir, exist_ok=True)
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with open(os.path.join(self.a.save_dir, 'tags.jsonl'), 'a') as f:
                f.write(json.dumps(dict(frame=self.n, stamp=st, seconds=round(time.time() - t0, 2), ram_tags=h['ram_tags'],
                                        objects=h['objects'])) + '\n')
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-topic', default='/camera/camera/color/image_raw')
    ap.add_argument('--server', default='127.0.0.1:47001')
    ap.add_argument('--save-dir', default='')
    a, _ = ap.parse_known_args()
    rclpy.init(); n = TagCheckNode(a)
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.02)
            if n.latest is not None:
                n.step()
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
