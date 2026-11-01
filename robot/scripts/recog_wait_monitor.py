#!/usr/bin/env python3
"""Recording helper: has CG-D seen enough of the current spot? (read only, commands nothing)
   python3 recog_wait_monitor.py [--frames 2]
Robot pose from TF map -> base_link (10 Hz). "Stopped" = |v| < 0.05 m/s and |w| < 0.05 rad/s.
Counts /cg_d/objects_2d results whose image was taken after the robot stopped; with --frames of them the spot
is done and the robot can move on (the live node also skips images taken while turning > 0.1 rad/s)."""
import argparse, math, time
from collections import Counter, deque
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from semantic_geometry_msgs.msg import SemanticObject2DArray

V_STILL, W_STILL = 0.05, 0.05
G, Y, R, B, N = '\033[1;32m', '\033[1;33m', '\033[1;31m', '\033[1m', '\033[0m'


class Monitor(Node):
    def __init__(self, need):
        super().__init__('recog_wait_monitor')
        self.need, self.hist = need, deque(maxlen=10)
        self.stopped_at, self.n, self.labels, self.last_frame, self.total = None, 0, Counter(), None, 0
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.create_subscription(SemanticObject2DArray, '/cg_d/objects_2d', self.on_frame, 10)
        self.create_timer(0.1, self.on_pose)
        self.create_timer(0.5, self.show)

    def on_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time())
        except Exception:
            return
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.hist.append((time.time(), t.transform.translation.x, t.transform.translation.y, yaw))
        if len(self.hist) < 5:
            return
        t0, x0, y0, a0 = self.hist[0]; t1, x1, y1, a1 = self.hist[-1]; dt = max(t1 - t0, 1e-3)
        self.v = math.hypot(x1 - x0, y1 - y0) / dt
        self.w = abs(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))) / dt
        self.yaw = a1
        still = self.v < V_STILL and self.w < W_STILL
        if still and self.stopped_at is None:
            self.stopped_at, self.n, self.labels = t0, 0, Counter()
        elif not still:
            self.stopped_at = None

    def on_frame(self, msg):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.total += 1; self.last_frame = time.time()
        if self.stopped_at is not None and st >= self.stopped_at:
            self.n += 1; self.labels.update(o.label for o in msg.objects)

    def show(self):
        out = ['\033[H\033[2J', f'{B}인식 대기 확인{N}  (멈춘 뒤 찍힌 사진 {self.need}장 처리되면 다음 위치로)\n']
        if not hasattr(self, 'v'):
            out.append(f'{R}위치추정 TF(map -> base_link) 대기 중{N}')
        else:
            out.append(f'속도 {self.v:.2f} m/s   회전 {math.degrees(self.w):.0f} deg/s   방향 {math.degrees(self.yaw):.0f} deg')
            if self.stopped_at is None:
                out.append(f'\n{Y}이동 중{N} — 멈추면 측정 시작')
            else:
                out.append(f'멈춘 시간 {time.time() - self.stopped_at:.0f} s   멈춘 뒤 처리된 사진 {self.n}장')
                out.append(f'\n{G}충분합니다. 이동해도 됩니다.{N}' if self.n >= self.need else
                           f'\n{Y}대기하세요 ({self.n}/{self.need}){N}')
                if self.labels:
                    out.append('이 위치에서 검출: ' + ', '.join(f'{k} {v}' for k, v in self.labels.most_common(10)))
        age = '없음' if self.last_frame is None else f'{time.time() - self.last_frame:.0f} s 전'
        out.append(f'\nCG-D 처리 사진 총 {self.total}장, 마지막 결과 {age}')
        if self.last_frame is not None and time.time() - self.last_frame > 15:
            out.append(f'{R}15 s 넘게 결과 없음: cgd_live.log 확인{N}')
        print('\n'.join(out), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', type=int, default=2)
    a = ap.parse_args(); rclpy.init(); n = Monitor(a.frames)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
