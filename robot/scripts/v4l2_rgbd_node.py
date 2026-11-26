#!/usr/bin/env python3
"""
RealSense D435 colour + depth straight from V4L2 (no librealsense), for USB 2.0 links where the librealsense node
receives no frames. Publishes the same topics as realsense2_camera so the perception scripts do not change:
  /camera/camera/color/image_raw (bgr8), .../color/image_raw/compressed (jpeg), .../color/camera_info
  /camera/camera/aligned_depth_to_color/image_raw (16UC1, mm) + camera_info
Depth is aligned to colour with the factory calibration in ~/real/config/calibration.yaml (depth intrinsics,
depth->colour extrinsic). Stamps are the host clock at frame arrival (same clock as the Livox driver).

  python3 v4l2_rgbd_node.py [--width 1280 --height 720 --fps 10] [--no-depth] [--calib ~/real/config/calibration.yaml]
Intrinsics come from calibration.yaml camera.color_<W>x<H> (factory values). Aligned depth needs 640x480.
"""
import argparse, os, re, subprocess, threading, time
import numpy as np, cv2, yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

FRAME = 'camera_color_optical_frame'


def find_devices():
    """(colour YUYV device, depth Z16 device) of the RealSense, by name and format (numbers change on replug)."""
    out = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True).stdout
    devs, cur = [], False
    for line in out.splitlines():
        if line and not line.startswith('\t'):
            cur = 'RealSense' in line
        elif cur and '/dev/video' in line:
            devs.append(line.strip())
    color = depth = None
    for d in devs:
        fmts = subprocess.run(['v4l2-ctl', '-d', d, '--list-formats'], capture_output=True, text=True).stdout
        if "'YUYV'" in fmts and color is None:
            color = d
        if "'Z16 '" in fmts and depth is None:
            depth = d
    return color, depth


def open_cap(dev, fourcc, fps, raw, w=640, h=480):
    c = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w); c.set(cv2.CAP_PROP_FRAME_HEIGHT, h); c.set(cv2.CAP_PROP_FPS, fps)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if raw:
        c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return c


class Aligner:
    def __init__(self, cal):
        d, c = cal['camera']['depth_640x480'], cal['camera']['color_640x480']
        self.Kd = (d['fx'], d['fy'], d['cx'], d['cy']); self.Kc = (c['fx'], c['fy'], c['cx'], c['cy'])
        R = np.array(cal['camera']['T_color_depth']['rotation_printed']).T      # printed column-major
        self.R, self.t = R, np.array(cal['camera']['T_color_depth']['translation_m'])
        v, u = np.mgrid[0:480, 0:640]
        self.rays = np.stack([(u - self.Kd[2]) / self.Kd[0], (v - self.Kd[3]) / self.Kd[1], np.ones_like(u, float)], -1).reshape(-1, 3)

    def __call__(self, depth_mm):
        z = depth_mm.reshape(-1).astype(np.float64) * 0.001
        ok = z > 0.1
        p = self.rays[ok] * z[ok, None]
        q = p @ self.R.T + self.t
        u = np.round(self.Kc[0] * q[:, 0] / q[:, 2] + self.Kc[2]).astype(int)
        v = np.round(self.Kc[1] * q[:, 1] / q[:, 2] + self.Kc[3]).astype(int)
        inb = (u >= 0) & (u < 640) & (v >= 0) & (v < 480) & (q[:, 2] > 0.1)
        out = np.full(480 * 640, 65535, np.uint16)
        np.minimum.at(out, v[inb] * 640 + u[inb], np.clip(q[inb, 2] * 1000, 0, 65534).astype(np.uint16))
        out[out == 65535] = 0
        return out.reshape(480, 640)


class RGBD(Node):
    def __init__(self, args):
        super().__init__('v4l2_rgbd')
        cal = yaml.safe_load(open(os.path.expanduser(args.calib)))
        self.W, self.H = args.width, args.height
        c = cal['camera'][f'color_{self.W}x{self.H}']
        self.info = CameraInfo(); self.info.header.frame_id = FRAME; self.info.width, self.info.height = self.W, self.H
        self.info.distortion_model = 'plumb_bob'; self.info.d = [0.0] * 5
        self.info.k = [float(v) for v in (c['fx'], 0.0, c['cx'], 0.0, c['fy'], c['cy'], 0.0, 0.0, 1.0)]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [float(v) for v in (c['fx'], 0.0, c['cx'], 0.0, 0.0, c['fy'], c['cy'], 0.0, 0.0, 0.0, 1.0, 0.0)]
        ns = '/camera/camera'
        self.p_img = self.create_publisher(Image, f'{ns}/color/image_raw', 5)
        self.p_jpg = self.create_publisher(CompressedImage, f'{ns}/color/image_raw/compressed', 5)
        self.p_info = self.create_publisher(CameraInfo, f'{ns}/color/camera_info', 5)
        self.p_dep = self.create_publisher(Image, f'{ns}/aligned_depth_to_color/image_raw', 5)
        self.p_dinfo = self.create_publisher(CameraInfo, f'{ns}/aligned_depth_to_color/camera_info', 5)
        color_dev, depth_dev = find_devices()
        if color_dev is None:
            raise SystemExit('RealSense colour device (YUYV) not found')
        self.get_logger().info(f'colour {color_dev}, depth {depth_dev if not args.no_depth else "off"}')
        self.col = open_cap(color_dev, 'YUYV', args.fps, False, self.W, self.H)
        aligned_ok = (self.W, self.H) == (640, 480)
        self.dep = None if args.no_depth or depth_dev is None or not aligned_ok else open_cap(depth_dev, 'Z16 ', args.fps, True)
        self.get_logger().info(f'colour {self.W}x{self.H} @ {args.fps} fps, fx {c["fx"]:.2f}, hfov '
                               f'{2 * np.degrees(np.arctan(self.W / (2 * c["fx"]))):.1f} deg')
        self.align = Aligner(cal)
        self.latest_depth, self.lock, self.n = None, threading.Lock(), [0, 0]
        if self.dep is not None:
            threading.Thread(target=self.depth_loop, daemon=True).start()
        threading.Thread(target=self.color_loop, daemon=True).start()
        self.create_timer(5.0, self.report)

    def depth_loop(self):
        while rclpy.ok():
            ok, b = self.dep.read()
            if not ok:
                time.sleep(0.01); continue
            t = self.get_clock().now()
            d = b.view(np.uint16).reshape(480, 640) if b.dtype != np.uint16 else b.reshape(480, 640)
            with self.lock:
                self.latest_depth = (t, d.copy())
            self.n[1] += 1

    def color_loop(self):
        while rclpy.ok():
            ok, img = self.col.read()
            if not ok:
                time.sleep(0.01); continue
            now = self.get_clock().now(); stamp = now.to_msg()
            m = Image(); m.header.stamp, m.header.frame_id = stamp, FRAME
            if img.shape[:2] != (self.H, self.W):
                self.get_logger().error(f'camera gave {img.shape[1]}x{img.shape[0]}, expected {self.W}x{self.H}', throttle_duration_sec=5.0)
                continue
            m.height, m.width, m.encoding, m.step = self.H, self.W, 'bgr8', self.W * 3; m.data = img.tobytes()
            self.p_img.publish(m)
            j = CompressedImage(); j.header = m.header; j.format = 'jpeg'
            j.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(); self.p_jpg.publish(j)
            self.info.header.stamp = stamp; self.p_info.publish(self.info)
            with self.lock:
                dl = self.latest_depth
            if dl is not None and (now - dl[0]).nanoseconds < 0.1e9:
                a = self.align(dl[1])
                dm = Image(); dm.header = m.header; dm.height, dm.width, dm.encoding, dm.step = 480, 640, '16UC1', 1280
                dm.data = a.tobytes(); self.p_dep.publish(dm); self.p_dinfo.publish(self.info)
            self.n[0] += 1

    def report(self):
        self.get_logger().info(f'last 5 s: colour {self.n[0] / 5:.1f} Hz, depth {self.n[1] / 5:.1f} Hz')
        self.n = [0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fps', type=int, default=10); ap.add_argument('--no-depth', action='store_true')
    ap.add_argument('--width', type=int, default=1280); ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--calib', default='~/real/config/calibration.yaml')
    args, _ = ap.parse_known_args()
    rclpy.init(); node = RGBD(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
