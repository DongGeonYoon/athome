#!/usr/bin/env python3
"""
Feed the live CG-D node (cgd_live_node.py) with saved keyframes one by one (next frame after its /cg_d/objects_2d
answer) to measure the online GPU use, first without and then with search targets on /athome/search/targets.
Run on a separate ROS_DOMAIN_ID so nothing reaches a running robot:
  python3 vram_live_test.py FRAME_DIR [--targets cup bottle ...] [--passes 1]
"""
import argparse, glob, os, time
import cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from semantic_geometry_msgs.msg import SemanticObject2DArray
from athome_interfaces.msg import SearchTarget, SearchTargets


class Feeder(Node):
    def __init__(self):
        super().__init__('vram_live_feeder')
        self.pub = self.create_publisher(Image, '/vram_test/image', 2)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.tpub = self.create_publisher(SearchTargets, '/athome/search/targets', latched)
        self.create_subscription(SemanticObject2DArray, '/cg_d/objects_2d', self.on_out, 10)
        self.answers = []

    def on_out(self, msg):
        self.answers.append(len(msg.objects))

    def send(self, path):
        img = cv2.imread(path); m = Image(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'camera'
        m.height, m.width = img.shape[:2]; m.encoding = 'bgr8'; m.step = img.shape[1] * 3; m.data = img.tobytes()
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('frames'); ap.add_argument('--targets', nargs='*', default=[])
    ap.add_argument('--passes', type=int, default=1); a = ap.parse_args()
    rclpy.init(); n = Feeder(); files = sorted(glob.glob(os.path.join(a.frames, 'frame*.jpg')))
    t = SearchTargets(); t.command_id = 1; t.targets = [SearchTarget(name=x, query_names=[x]) for x in a.targets]; n.tpub.publish(t)
    t_wait = time.time()
    while n.count_subscribers('/vram_test/image') == 0:          # wait for the node (model loading)
        rclpy.spin_once(n, timeout_sec=0.5)
        if time.time() - t_wait > 180:
            print('node not found on this ROS_DOMAIN_ID within 180 s', flush=True); return
    time.sleep(2.0)
    for p in files * a.passes:
        k = len(n.answers); n.send(p); t0 = time.time()
        while len(n.answers) == k and time.time() - t0 < 120:
            rclpy.spin_once(n, timeout_sec=0.1)
        if len(n.answers) == k:
            print(f'no answer for {os.path.basename(p)} (node dead?)', flush=True); break
        print(f'{os.path.basename(p)}: {n.answers[-1]} objects, {time.time() - t0:.1f} s', flush=True)
    print(f'DONE frames={len(n.answers)}', flush=True)


if __name__ == '__main__':
    main()
