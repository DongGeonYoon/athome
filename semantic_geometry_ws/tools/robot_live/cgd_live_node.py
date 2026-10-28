"""
Live CG-D on the robot camera (CG-D code unchanged; same GPU-split runner and v2 vocabulary as the dataset).

Takes the newest colour image whenever the previous one is finished (CG-D needs ~4-6 s per frame on an 8 GB GPU),
runs RAM -> GroundingDINO -> SAM -> CLIP exactly like generate_gsa_results.py, applies the same detection filter as
the dataset (confidence >= 0.25, bbox <= 50 % of the image, no wall/floor/ceiling) and publishes
  /cg_d/objects_2d   SemanticObject2DArray (header = image stamp)  -> geometry_fusion_node
  /cg_d/annotated    sensor_msgs/Image (boxes, masks, labels)        -> rqt_image_view
Search targets from /athome/search/targets (athome_interfaces/SearchTargets): every query_names entry of every target
is appended to the DINO class list, so a target detection's label is one of its query_names verbatim (the search server
judges "found" by that label). An empty list returns to the plain vocabulary.

Run with the conceptgraph python after sourcing ROS (see start_perception.sh):
  python cgd_live_node.py [--image-topic /camera/camera/color/image_raw] [--period 0]
CG-D on the GPU server instead (cgd_server.py there, SSH tunnel to it; plain python3, no torch needed here):
  python3 cgd_live_node.py --server 127.0.0.1:47000 [...]
Search timing (require4): only the newest image is kept (depth-1 subscription), so a 5-7 s frame never leaves older
queued images to work through; images taken while the robot turns faster than --max-turn-rate (0.1 rad/s, odometry TF
camera_init -> body at the image stamp) are skipped, so the first image after the robot stops is the next one processed.
Server mode survives a lost connection (reconnect and retry; nothing is published meanwhile, which the search server
sees as a perception gap).
"""
import argparse, json, os, sys, time
import numpy as np, cv2

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CompressedImage
from semantic_geometry_msgs.msg import SemanticObject2D, SemanticObject2DArray
from athome_interfaces.msg import SearchTargets


def to_image_msg(img_bgr, header):
    m = Image(); m.header = header; m.height, m.width = img_bgr.shape[:2]
    m.encoding = 'bgr8'; m.step = int(img_bgr.shape[1] * 3); m.data = np.ascontiguousarray(img_bgr).tobytes()
    return m


def mask_msg(mask):
    m = Image(); m.height, m.width = mask.shape; m.encoding = 'mono8'; m.step = int(mask.shape[1])
    m.data = (mask.astype(np.uint8) * 255).tobytes()
    return m


class CGDLive(Node):
    def __init__(self, args):
        super().__init__('cgd_live')
        self.args, self.latest, self.targets, self.n = args, None, [], 0
        t0 = time.time()
        if args.server:                            # CG-D on the GPU server (cgd_server.py) through an SSH tunnel
            from cgd_wire import RemoteEngine
            self.engine = RemoteEngine(args.server)
            self.get_logger().info(f'CG-D runs on the GPU server via {args.server}')
        else:
            from cgd_engine import CGDEngine      # CG-D models + one-image inference (shared with cgd_server.py)
            self.engine = CGDEngine('cuda')
            self.get_logger().info(f'CG-D models loaded in {time.time() - t0:.0f} s')
        compressed = args.image_topic.endswith('/compressed')
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(CompressedImage if compressed else Image, args.image_topic, self.on_image, newest)
        self.tf = None
        if args.max_turn_rate > 0:                 # own node + thread: the TF buffer fills while a frame is processed
            self.tf_node = rclpy.create_node('cgd_live_tf')
            self.tf = Buffer(); self._tfl = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.n_skipped, self._tf_warned = 0, False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(SearchTargets, '/athome/search/targets', self.on_targets, latched)
        self.pub = self.create_publisher(SemanticObject2DArray, '/cg_d/objects_2d', 10)
        self.vis = self.create_publisher(Image, '/cg_d/annotated', 2)

    def on_image(self, msg):
        self.latest = msg

    def on_targets(self, msg):
        names = [q.strip() for t in msg.targets for q in t.query_names if q.strip()]
        self.targets = list(dict.fromkeys(names))          # query names in order, duplicates removed
        self.get_logger().info(f'search targets (command {msg.command_id}): '
                               f'{[(t.name, list(t.query_names)) for t in msg.targets]} -> query {self.targets}')

    def decode(self, msg):
        if isinstance(msg, CompressedImage):
            return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding.lower() == 'rgb8' else img.copy()

    def turn_rate(self, stamp):
        """|yaw rate| [rad/s] of the robot at the image stamp from the odometry TF (None: no TF)."""
        a = self.args
        try:
            t = Time.from_msg(stamp)
            T1 = self.tf.lookup_transform(a.odom_frame, a.robot_frame, t, timeout=Duration(seconds=0.3))
            T0 = self.tf.lookup_transform(a.odom_frame, a.robot_frame, t - Duration(seconds=a.turn_window),
                                          timeout=Duration(seconds=0.05))
        except Exception as e:
            if not self._tf_warned:
                self.get_logger().warning(f'no {a.odom_frame} -> {a.robot_frame} TF at the image stamp ({e}); '
                                          'turn filter inactive until it is available')
                self._tf_warned = True
            return None
        yaw = lambda q: np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        d = yaw(T1.transform.rotation) - yaw(T0.transform.rotation)
        return abs(np.arctan2(np.sin(d), np.cos(d))) / a.turn_window

    def step(self):
        msg, self.latest = self.latest, None
        if self.tf is not None:
            w = self.turn_rate(msg.header.stamp)
            if w is not None and w > self.args.max_turn_rate:
                self.n_skipped += 1
                self.get_logger().info(f'image skipped: turning {w:.2f} rad/s ({self.n_skipped} skipped)',
                                       throttle_duration_sec=2.0)
                return
        t0 = time.time()
        try:
            r = self.engine.run(self.decode(msg), self.targets)
        except (OSError, ConnectionError) as e:   # GPU server / tunnel gone: retry with the next image
            self.get_logger().warning(f'CG-D server not reachable ({e}); retrying', throttle_duration_sec=5.0)
            time.sleep(1.0)
            return
        out = SemanticObject2DArray(); out.header = msg.header
        for d in r['objects']:
            o = SemanticObject2D(); o.header = msg.header; o.detection_id = d['detection_id']
            o.bbox_xyxy = d['bbox']; o.label = d['label']; o.confidence = d['confidence']
            o.mask = mask_msg(d['mask']); o.clip_feature = d['clip'].tolist()
            out.objects.append(o)
        self.publish(msg, out, r, t0)

    def publish(self, msg, out, r, t0):
        """/cg_d/objects_2d, experiment record (--save-dir), /cg_d/annotated, log line."""
        annotated = r['annotated']
        self.pub.publish(out)
        if self.args.save_dir:                     # experiment record: annotated frame + detections per frame
            os.makedirs(os.path.join(self.args.save_dir, 'annotated'), exist_ok=True)
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cv2.imwrite(os.path.join(self.args.save_dir, 'annotated', f'{self.n:05d}_{st:.2f}.jpg'), annotated)
            with open(os.path.join(self.args.save_dir, 'detections.jsonl'), 'a') as f:
                f.write(json.dumps(dict(frame=self.n, stamp=st, seconds=round(time.time() - t0, 2), ram_tags=r['ram_tags'],
                                        query=self.targets, n_classes=r['n_classes'],
                                        objects=[dict(label=o.label, confidence=round(float(o.confidence), 3),
                                                      bbox=[round(float(v), 1) for v in o.bbox_xyxy]) for o in out.objects])) + '\n')
        cv2.putText(annotated, f'{len(out.objects)} objects  {time.time() - t0:.1f} s', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        self.vis.publish(to_image_msg(annotated, msg.header))
        self.n += 1
        labels = [f'{o.label} {o.confidence:.2f}' for o in out.objects]
        self.get_logger().info(f'frame {self.n}: {time.time() - t0:.1f} s, {len(out.objects)} objects: {", ".join(labels)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-topic', default='/camera/camera/color/image_raw')
    ap.add_argument('--period', type=float, default=0.0, help='minimum seconds between processed frames (0 = back to back)')
    ap.add_argument('--save-dir', default='', help='write annotated frames and detections.jsonl here')
    ap.add_argument('--server', default='', help='HOST:PORT of cgd_server.py (GPU server through an SSH tunnel); '
                                                 'empty = run CG-D on this PC')
    ap.add_argument('--max-turn-rate', type=float, default=0.1, help='skip images taken while |yaw rate| > this [rad/s], 0 = off')
    ap.add_argument('--turn-window', type=float, default=0.2, help='yaw rate over this window before the image [s]')
    ap.add_argument('--odom-frame', default='camera_init'); ap.add_argument('--robot-frame', default='body')
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = CGDLive(args)
    last = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest is not None and time.time() - last >= args.period:
                last = time.time(); node.step()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
