#!/usr/bin/env python3
"""
Publish CG-D detections as /cg_d/objects_2d in sync with a replayed rosbag (sim time), so the real
geometry_fusion_node can be tested on recorded sensor data.

Each pkl (frameXXXXXX.pkl.gz) is published once /clock passes its keyframe stamp + DELAY (so the LiDAR
scans of the accumulation window are already in the node's queue). The header stamp is the keyframe stamp.

Usage: python3 semantic_replay.py GSA_DIR KEYFRAME_STAMPS.csv [--delay 0.6 --frame camera_color_optical_frame]
"""
import argparse, csv, os, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time as TimeMsg

sys.path.insert(0, os.path.expanduser('~/semantic_geometry_ws/src/semantic_geometry_fusion'))
from semantic_geometry_fusion.gsa_io import load_gsa, iter_filtered_detections
from semantic_geometry_fusion.gsa_pkl_bridge import GSAPklBridge
from semantic_geometry_msgs.msg import SemanticObject2D, SemanticObject2DArray


class Replay(Node):
    def __init__(self, gsa_dir, stamps, delay, frame):
        super().__init__('semantic_replay')
        self.items = sorted((t, os.path.join(gsa_dir, f)) for f, t in stamps.items() if os.path.exists(os.path.join(gsa_dir, f)))
        self.delay, self.frame, self.k = delay, frame, 0
        self.pub = self.create_publisher(SemanticObject2DArray, '/cg_d/objects_2d', 10)
        self.create_subscription(Clock, '/clock', self.on_clock, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

    def on_clock(self, msg):
        now = msg.clock.sec + msg.clock.nanosec * 1e-9
        while self.k < len(self.items) and now >= self.items[self.k][0] + self.delay:
            t, path = self.items[self.k]; self.k += 1
            out = SemanticObject2DArray()
            out.header.stamp = TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
            out.header.frame_id = self.frame
            for d in iter_filtered_detections(load_gsa(path)):
                o = SemanticObject2D(); o.header = out.header
                o.detection_id = int(d['detection_id']); o.bbox_xyxy = [float(v) for v in d['bbox_xyxy']]
                o.label = d['label']; o.confidence = float(d['confidence'])
                o.mask = GSAPklBridge.mask_msg(d['mask']); o.clip_feature = [float(v) for v in d['clip_feature']]
                out.objects.append(o)
            self.pub.publish(out)
            self.get_logger().info(f'{os.path.basename(path)} t={t:.2f} objects={len(out.objects)} ({self.k}/{len(self.items)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gsa_dir'); ap.add_argument('stamps_csv')
    ap.add_argument('--delay', type=float, default=0.6); ap.add_argument('--frame', default='camera_color_optical_frame')
    args = ap.parse_args()
    stamps = {r['filename']: float(r['timestamp_sec']) for r in csv.DictReader(open(args.stamps_csv))}
    rclpy.init()
    rclpy.spin(Replay(args.gsa_dir, stamps, args.delay, args.frame))


if __name__ == '__main__':
    main()
