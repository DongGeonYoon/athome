#!/usr/bin/env python3
"""
Observation-heading guide for the 6-direction stop-and-look scan (60 deg steps from the start heading).
Shows, once a second: current heading and sector, how long the robot has been still, and how many CG-D frames were
processed in each sector (a frame counts for the sector the robot faced at the frame's image stamp).
A sector is "done" with >= 2 processed frames taken while still.
  python3 heading_guide.py [--sectors 6] [--need 2]
"""
import argparse, math, time
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from semantic_geometry_msgs.msg import ObjectObservation3DArray


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Guide(Node):
    def __init__(self, a):
        super().__init__('heading_guide')
        self.a, self.start, self.hist, self.done = a, None, deque(maxlen=600), [0] * a.sectors
        self.still_since, self.last = None, None
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.create_subscription(ObjectObservation3DArray, '/athome/perception/objects', self.on_frame, 10)
        self.create_timer(0.2, self.on_pose); self.create_timer(1.0, self.show)

    def sector(self, yaw):
        d = math.degrees(yaw - self.start) % 360
        return int(((d + 180 / self.a.sectors) % 360) // (360 / self.a.sectors)), d

    def on_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time())
        except Exception:
            return
        y = yaw_of(t.transform.rotation); now = time.time()
        if self.start is None:
            self.start = y
        # still = heading changed < 3 deg over the last second (localization jitter is ~0.3 deg per update)
        ref = [h for h in self.hist if now - h[0] >= 1.0]
        moving = bool(ref) and abs(math.degrees(math.atan2(math.sin(y - ref[-1][1]), math.cos(y - ref[-1][1])))) > 3.0
        if moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = now
        self.last = (now, y); self.hist.append((now, y, self.still_since is not None))

    def on_frame(self, msg):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start is None or not self.hist:
            return
        # pose closest to the image stamp (host clock); count only frames taken while still
        tt, y, still = min(self.hist, key=lambda h: abs(h[0] - st))
        if abs(tt - st) < 2.0 and still:
            self.done[self.sector(y)[0]] += 1

    def show(self):
        if self.last is None:
            print('[guide] localization 대기 중 (RViz에서 2D Pose Estimate)', flush=True); return
        s, d = self.sector(self.last[1]); held = time.time() - self.still_since if self.still_since else 0.0
        need = self.a.need; cur = self.done[s]
        if self.still_since is None:
            act = '회전 중'
        elif cur >= need:
            act = f'>>> 다음 방향으로 {360 // self.a.sectors}° 회전 <<<' if min(self.done) < need else '>>> 6방향 완료: 종료해도 됨 <<<'
        else:
            act = f'정지 유지 (이 방향 {cur}/{need}장)'
        bar = ' '.join(f"{i + 1}:{'■' * min(c, need)}{'□' * max(0, need - c)}" for i, c in enumerate(self.done))
        print(f'[guide] 방향 {s + 1}/{self.a.sectors}  현재 {d:5.1f}° (목표 {s * 360 // self.a.sectors}°)  정지 {held:4.1f}s  | {bar} | {act}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sectors', type=int, default=6); ap.add_argument('--need', type=int, default=2)
    a = ap.parse_args(); rclpy.init(); n = Guide(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
