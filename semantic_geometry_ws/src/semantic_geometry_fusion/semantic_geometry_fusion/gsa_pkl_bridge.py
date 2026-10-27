from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image
from semantic_geometry_msgs.msg import SemanticObject2D, SemanticObject2DArray
from .gsa_io import load_gsa, iter_filtered_detections


def to_time_msg(t):
    sec = int(t)
    ns = int(round((t - sec) * 1e9))
    return TimeMsg(sec=sec, nanosec=ns)


class GSAPklBridge(Node):
    def __init__(self):
        super().__init__("gsa_pkl_bridge")
        self.declare_parameter("gsa_dir", "")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("timestamp_csv", "")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("mask_conf_threshold", 0.25)
        self.declare_parameter("max_bbox_area_ratio", 0.5)
        self.declare_parameter("skip_bg", False)

        self.gsa_dir = Path(self.get_parameter("gsa_dir").value).expanduser()
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.rate = float(self.get_parameter("publish_rate_hz").value)
        self.mask_conf = float(self.get_parameter("mask_conf_threshold").value)
        self.max_bbox = float(self.get_parameter("max_bbox_area_ratio").value)
        self.skip_bg = bool(self.get_parameter("skip_bg").value)

        if not self.gsa_dir.exists():
            raise RuntimeError(f"gsa_dir does not exist: {self.gsa_dir}")
        self.files = sorted(self.gsa_dir.glob("frame*.pkl.gz"))
        if not self.files:
            raise RuntimeError(f"No frame*.pkl.gz under {self.gsa_dir}")

        self.timestamps = {}
        csv_path = str(self.get_parameter("timestamp_csv").value).strip()
        if csv_path:
            with open(Path(csv_path).expanduser(), newline="") as f:
                for row in csv.DictReader(f):
                    self.timestamps[row["filename"]] = float(row["timestamp_sec"])
        else:
            self.get_logger().warning(
                "timestamp_csv is empty. Current-time fallback is for message inspection only, "
                "not LiDAR-camera synchronization validation."
            )

        self.pub = self.create_publisher(SemanticObject2DArray, "/cg_d/objects_2d", 10)
        self.idx = 0
        self.timer = self.create_timer(1.0 / max(self.rate, 1e-6), self.tick)

    @staticmethod
    def mask_msg(mask):
        mask = np.asarray(mask, dtype=np.uint8) * 255
        m = Image()
        m.height = int(mask.shape[0])
        m.width = int(mask.shape[1])
        m.encoding = "mono8"
        m.is_bigendian = 0
        m.step = int(mask.shape[1])
        m.data = mask.tobytes()
        return m

    def tick(self):
        if self.idx >= len(self.files):
            self.get_logger().info("Finished GSA pkl files")
            self.timer.cancel()
            return

        path = self.files[self.idx]
        gobs = load_gsa(path)
        out = SemanticObject2DArray()
        if path.name in self.timestamps:
            out.header.stamp = to_time_msg(self.timestamps[path.name])
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.camera_frame

        for d in iter_filtered_detections(
            gobs,
            mask_conf_threshold=self.mask_conf,
            max_bbox_area_ratio=self.max_bbox,
            skip_bg=self.skip_bg,
        ):
            o = SemanticObject2D()
            o.header = out.header
            o.detection_id = int(d["detection_id"])
            o.bbox_xyxy = [float(v) for v in d["bbox_xyxy"]]
            o.label = d["label"]
            o.confidence = float(d["confidence"])
            o.mask = self.mask_msg(d["mask"])
            o.clip_feature = [float(v) for v in d["clip_feature"]]
            out.objects.append(o)

        self.pub.publish(out)
        dims = sorted(set(len(o.clip_feature) for o in out.objects))
        self.get_logger().info(f"{path.name}: objects={len(out.objects)}, clip_dims={dims}")
        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = GSAPklBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
