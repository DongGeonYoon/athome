from __future__ import annotations

from collections import deque
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy._rclpy_pybind11 import RCLError
import time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from std_srvs.srv import Trigger
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException

from semantic_geometry_msgs.msg import (
    SemanticObject2DArray,
    ObjectObservation3D,
    ObjectObservation3DArray,
)
from athome_interfaces.msg import ObservedObject, ObservedObjectArray, SearchTargets

from .geometry import (
    points_inside_mask,
    transform_points,
    compute_geometry,
    depth_gate,
    erode_mask,
    largest_cluster,
    remove_outliers,
    depth_points_in_mask,
    voxel_downsample,
)
from .feature_fusion import (Observation3D, ObjectFeatureMap, dedup_detections, denoise_object, apply_size_prior,
                             merge_fragments)
from .scene_graph_export import save_scene_graph_json
from .prior_map import load_prior_map
from .association import l2_normalize
from .ros_utils import stamp_to_sec, transform_stamped_to_matrix


class GeometryFusionNode(Node):
    def __init__(self):
        super().__init__("geometry_fusion_node")

        # Topics/frame names are implementation parameters; proposal equations are independent of names.
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("semantic_topic", "/cg_d/objects_2d")
        self.declare_parameter("observation_topic", "/athome/perception/objects")   # agreed with Scene Graph
        self.declare_parameter("static_map_topic", "/semantic_geometry/static_map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("dynamic_label", "person")

        # These thresholds are proposal parameters and are intentionally NOT assigned a research value here.
        self.declare_parameter("max_sync_dt_sec", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("nn_distance_threshold", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("association_threshold", rclpy.Parameter.Type.DOUBLE)

        # LiDAR -> camera optical extrinsic, row-major 4x4. Must come from calibration.
        self.declare_parameter("T_cam_lidar", rclpy.Parameter.Type.DOUBLE_ARRAY)

        # Replica + simulated Mid-360 feasibility study defaults (tools/replica_feasibility).
        # Scans within +-accumulation_window_sec/2 of the camera stamp are merged; 0 = nearest scan only.
        self.declare_parameter("accumulation_window_sec", 0.5)
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("depth_gate_k_mad", 2.5)
        self.declare_parameter("depth_gate_min_band", 0.05)
        self.declare_parameter("mask_erosion_px", 0)

        # Object-Centric Feature JSON for the (offline) Scene Graph; empty disables. Rebuilding it costs
        # ~0.7 s per 300 map objects, so it is rewritten at most every scene_graph_export_period_sec (0 = after
        # every frame), on the ~/save_scene_graph service and always on shutdown.
        self.declare_parameter("scene_graph_json_path", "")
        self.declare_parameter("scene_graph_export_period_sec", 10.0)
        self.declare_parameter("static_map_publish_period_sec", 2.0)   # full map (all points) topic, 0 = every frame
        self.declare_parameter("scene_graph_min_observations", 2)
        self.declare_parameter("scene_graph_excluded_tags", ["other item", "furniture", "pad", "object", "item", "thing",
                                                             "tile wall", "tile", "wall", "floor", "ceiling", "pillar", "pole",
                                                             "lamp post", "balustrade", "tile floor", "wood floor"])

        # Over-merge fixes (scene_graph_dataset v0.3): duplicate masks in one frame and object-level
        # cleanup (largest eps-connected component) before the JSON export. growth_ratio and
        # frame_one_to_one are off by default (no gain / fragmentation on Replica room0). 0 disables.
        self.declare_parameter("dedup_mask_iou", 0.8)
        self.declare_parameter("obs_cluster_eps", 0.15)   # per-observation DBSCAN (proposal 2-3), 0 disables
        self.declare_parameter("frame_one_to_one", False)
        self.declare_parameter("growth_ratio", 0.0)
        self.declare_parameter("growth_margin", 0.3)
        self.declare_parameter("denoise_eps", 0.20)
        # Stray points before every bbox (per observation and at export): statistical outlier removal restricted to
        # isolated points (geometry.remove_outliers). outlier_k 0 disables.
        self.declare_parameter("outlier_k", 8)
        self.declare_parameter("outlier_std_ratio", 2.0)
        self.declare_parameter("outlier_min_gap", 0.08)
        self.declare_parameter("outlier_min_points", 20)
        # Camera-depth fallback for objects the LiDAR does not see (small items on table tops above the
        # Mid-360). Aligned depth (same intrinsics as colour); "" disables. 16UC1 in mm or 32FC1 in m.
        # CG-D results arrive seconds after the image: keep LiDAR scans / TF that long, and hold a detection
        # until the scans of its accumulation window have arrived instead of dropping it.
        self.declare_parameter("sensor_buffer_sec", 30.0)
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("size_prior_slack", 1.5)   # export label check vs SIZE_PRIOR_MAX, 0 disables
        self.declare_parameter("merge_fragments_cos", 0.9)  # export re-merge of one object split in two, 0 disables
        self.declare_parameter("depth_fallback_stride", 4)
        # Search targets (/athome/search/targets, athome_interfaces/SearchTargets): every query_names entry of every
        # target is a target label. A target label seen once wins the object's label vote, and target detections need
        # at least target_min_confidence (VLFM-style stricter gate for the goal object, 0.4 agreed with the server).
        self.declare_parameter("search_targets_topic", "/athome/search/targets")
        self.declare_parameter("target_min_confidence", 0.4)
        # athome_interfaces/ObservedObjectArray contract with the search server: clip_model names the CLIP the map
        # was built with; object_id is the ID in the pre-mapped objects.json (prior_objects_json) when the
        # observation associates with that object (same phi_sem + phi_geo >= association_threshold), else -1.
        self.declare_parameter("clip_model", "ViT-H-14/laion2b_s32b_b79k")
        self.declare_parameter("prior_objects_json", "")

        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.semantic_topic = self.get_parameter("semantic_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.dynamic_label = self.get_parameter("dynamic_label").value

        self.max_sync_dt = self._required_positive_double("max_sync_dt_sec")
        nn_threshold = self._required_positive_double("nn_distance_threshold")
        association_threshold = self._required_double("association_threshold")

        extrinsic = np.asarray(self.get_parameter("T_cam_lidar").value, dtype=np.float64)
        if extrinsic.size != 16:
            raise RuntimeError("Parameter T_cam_lidar must contain 16 row-major values")
        self.T_cam_lidar = extrinsic.reshape(4, 4)

        self.accumulation_window = float(self.get_parameter("accumulation_window_sec").value)
        self.gate_k_mad = float(self.get_parameter("depth_gate_k_mad").value)
        self.gate_min_band = float(self.get_parameter("depth_gate_min_band").value)
        self.mask_erosion_px = int(self.get_parameter("mask_erosion_px").value)
        self.scene_graph_json_path = str(self.get_parameter("scene_graph_json_path").value)
        self.scene_graph_min_obs = int(self.get_parameter("scene_graph_min_observations").value)
        self.scene_graph_excluded = set(self.get_parameter("scene_graph_excluded_tags").value)
        self.dedup_iou = float(self.get_parameter("dedup_mask_iou").value)
        self.obs_cluster_eps = float(self.get_parameter("obs_cluster_eps").value)
        self.denoise_eps = float(self.get_parameter("denoise_eps").value)
        self.outliers = dict(k=int(self.get_parameter("outlier_k").value),
                             std_ratio=float(self.get_parameter("outlier_std_ratio").value),
                             min_gap=float(self.get_parameter("outlier_min_gap").value),
                             min_points=int(self.get_parameter("outlier_min_points").value))
        self.frame_one_to_one = bool(self.get_parameter("frame_one_to_one").value)
        growth_ratio = float(self.get_parameter("growth_ratio").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.size_prior_slack = float(self.get_parameter("size_prior_slack").value)
        self.merge_cos = float(self.get_parameter("merge_fragments_cos").value)
        self.depth_stride = int(self.get_parameter("depth_fallback_stride").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.export_period = float(self.get_parameter("scene_graph_export_period_sec").value)
        self.map_period = float(self.get_parameter("static_map_publish_period_sec").value)
        self.target_min_conf = float(self.get_parameter("target_min_confidence").value)
        self.targets = set()
        self.clip_model = str(self.get_parameter("clip_model").value)
        prior_path = str(self.get_parameter("prior_objects_json").value)
        self.prior_map = None
        if prior_path:
            self.prior_map = load_prior_map(prior_path, nn_threshold, association_threshold,
                                            float(self.get_parameter("voxel_size").value))
            self.get_logger().info(f"prior map: {len(self.prior_map.objects)} objects from {prior_path}")
        self._map_dirty = self._json_dirty = False

        self.feature_map = ObjectFeatureMap(
            nn_distance_threshold=nn_threshold,
            association_threshold=association_threshold,
            voxel_size=float(self.get_parameter("voxel_size").value),
            growth_ratio=growth_ratio if growth_ratio > 0 else None,
            growth_margin=float(self.get_parameter("growth_margin").value),
        )

        self.K = None
        self.bridge = CvBridge()
        self.buffer_sec = float(self.get_parameter("sensor_buffer_sec").value)
        self.lidar_queue = deque()
        self.depth_queue = deque()
        self.pending = deque()      # semantic messages waiting for LiDAR scans after their stamp
        # Sensor callbacks only buffer (own thread, never blocked by the ~1 s per-frame fusion); fusion, timers and
        # services run in a second group. Without this, scans arriving during fusion overflowed the subscription
        # queue and frames were dropped (Gazebo replay under CPU load: 9 of 108).
        self._lock = threading.Lock()
        self._stats = []            # (seconds, accumulate seconds) per processed frame, logged every 20 frames
        sensors, proc = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()

        # TF history must outlive the longest wait of a pending detection (sensor buffer may grow beyond buffer_sec)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0 * self.buffer_sec))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PointCloud2, self.lidar_topic, self._lidar_cb, 100, callback_group=sensors)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 10, callback_group=sensors)
        self.create_subscription(SemanticObject2DArray, self.semantic_topic, self._semantic_cb, 50, callback_group=proc)
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self._depth_cb, 20, callback_group=sensors)
        self.create_timer(0.05, self._drain_pending, callback_group=proc)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(SearchTargets, self.get_parameter("search_targets_topic").value, self._targets_cb, latched,
                                 callback_group=proc)
        self.create_service(Trigger, "~/save_scene_graph", self._save_srv, callback_group=proc)
        if self.map_period > 0:
            self.create_timer(self.map_period, self._publish_map_if_dirty, callback_group=proc)
        if self.scene_graph_json_path and self.export_period > 0:
            self.create_timer(self.export_period, self._export_if_dirty, callback_group=proc)

        self.obs_pub = self.create_publisher(
            ObservedObjectArray,
            self.get_parameter("observation_topic").value,
            10,
        )
        self.map_pub = self.create_publisher(
            ObjectObservation3DArray,
            self.get_parameter("static_map_topic").value,
            10,
        )

    def _targets_cb(self, msg: SearchTargets):
        # empty list = not searching: plain vocabulary and plain majority vote again
        self.targets = {q.strip() for t in msg.targets for q in t.query_names if q.strip()}
        self.feature_map.set_priority_labels(self.targets)
        self.get_logger().info(f"search targets (command {msg.command_id}): "
                               f"{[(t.name, list(t.query_names)) for t in msg.targets]}")
        self._map_dirty = self._json_dirty = True

    def _publish_map_if_dirty(self):
        if self._map_dirty:
            self._publish_map()

    def _publish_map(self):
        self._map_dirty = False
        map_array = ObjectObservation3DArray()
        map_array.header.stamp = self.get_clock().now().to_msg()
        map_array.header.frame_id = self.map_frame
        for obj in self.feature_map.objects:
            map_array.objects.append(self._map_obj_to_msg(obj, map_array.header))
        self.map_pub.publish(map_array)

    def _export_if_dirty(self):
        if self._json_dirty:
            self.export_scene_graph()

    def export_scene_graph(self) -> int:
        """Write the Scene Graph JSON from the current map; returns the number of exported objects."""
        self._json_dirty = False
        exported = [denoise_object(o, self.denoise_eps, outliers=self.outliers) if self.denoise_eps > 0 else o
                    for o in self.feature_map.objects]
        if self.merge_cos > 0:
            exported = merge_fragments(exported, self.voxel_size, min_cos=self.merge_cos)
        if self.size_prior_slack > 0:
            exported = [apply_size_prior(o, slack=self.size_prior_slack) for o in exported]
        exported = [o for o in exported if o.label not in self.scene_graph_excluded]
        data = save_scene_graph_json(
            self.scene_graph_json_path,
            exported,
            frame_id=self.map_frame,
            min_observations=self.scene_graph_min_obs,
        )
        return len(data["objects"])

    def _save_srv(self, request, response):
        if not self.scene_graph_json_path:
            response.success, response.message = False, "scene_graph_json_path is empty"
            return response
        n = self.export_scene_graph()
        response.success, response.message = True, f"{n} objects -> {self.scene_graph_json_path}"
        return response

    def _required_positive_double(self, name: str) -> float:
        p = self.get_parameter(name)
        if p.type_ == rclpy.Parameter.Type.NOT_SET:
            raise RuntimeError(f"Required parameter '{name}' was not set")
        value = float(p.value)
        if value <= 0.0:
            raise RuntimeError(f"Parameter '{name}' must be > 0")
        return value

    def _required_double(self, name: str) -> float:
        p = self.get_parameter(name)
        if p.type_ == rclpy.Parameter.Type.NOT_SET:
            raise RuntimeError(f"Required parameter '{name}' was not set")
        return float(p.value)

    def _camera_info_cb(self, msg: CameraInfo):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def _trim(queue, newest, horizon):
        while queue and queue[0][0] < newest - horizon:
            queue.popleft()

    def _lidar_cb(self, msg: PointCloud2):
        t = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self.lidar_queue.append((t, msg))
            self._trim(self.lidar_queue, t, self._horizon(t))

    def _horizon(self, newest: float) -> float:
        """Keep sensor_buffer_sec of data, and never drop scans a waiting detection still needs."""
        try:
            oldest = stamp_to_sec(self.pending[0].header.stamp)    # read from the sensor thread: may race, then retry
        except IndexError:
            return self.buffer_sec
        return max(self.buffer_sec, newest - oldest + self.accumulation_window + 1.0)

    def _drain_pending(self):
        # The frame stays at pending[0] while it is processed so the sensor thread keeps the scans it needs.
        while self.pending and self._lidar_window_complete(stamp_to_sec(self.pending[0].header.stamp)):
            if self._process_semantic(self.pending[0]) == "retry":
                return                           # TF for the scan not received yet: try again on the next tick
            self.pending.popleft()

    def _snapshot(self, queue):
        with self._lock:
            return list(queue)

    def _depth_cb(self, msg: Image):
        t = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self.depth_queue.append((t, msg))
            self._trim(self.depth_queue, t, self._horizon(t))

    def _lidar_window_complete(self, camera_stamp_sec: float) -> bool:
        """True once a scan later than the accumulation window (or the sync tolerance) has arrived."""
        horizon = max(0.5 * self.accumulation_window, self.max_sync_dt)
        with self._lock:
            return bool(self.lidar_queue) and self.lidar_queue[-1][0] > camera_stamp_sec + horizon

    def _depth_at(self, camera_stamp_sec: float):
        """Aligned depth image (metres) closest to the camera stamp, or None."""
        queue = self._snapshot(self.depth_queue)
        if not queue:
            return None
        t, msg = min(queue, key=lambda item: abs(item[0] - camera_stamp_sec))
        if abs(t - camera_stamp_sec) > self.max_sync_dt:
            return None
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float64)
        return d * 0.001 if msg.encoding in ("16UC1", "mono16") else d

    def _nearest_lidar(self, camera_stamp_sec: float):
        queue = self._snapshot(self.lidar_queue)
        if not queue:
            return None
        best = min(queue, key=lambda item: abs(item[0] - camera_stamp_sec))
        if abs(best[0] - camera_stamp_sec) > self.max_sync_dt:
            return None
        return best[1]

    def _window_scans(self, camera_stamp_sec: float, nearest: PointCloud2):
        """Scans within +-accumulation_window/2 of the camera stamp (always includes the nearest scan)."""
        half = 0.5 * self.accumulation_window
        scans = [msg for t, msg in self._snapshot(self.lidar_queue) if abs(t - camera_stamp_sec) <= half]
        if not any(msg is nearest for msg in scans):
            scans.append(nearest)
        return scans

    def _accumulated_points(self, camera_stamp_sec: float, nearest: PointCloud2):
        """
        Merge scans around the camera stamp, each compensated with its own map pose,
        expressed in the nearest scan's LiDAR frame. Returns (points_ref_lidar, T_map_lidar_ref).
        """
        T_ref = self._lookup_T_map_lidar(nearest)
        if T_ref is None:
            return None, None

        chunks = []
        for scan in self._window_scans(camera_stamp_sec, nearest):
            T_scan = T_ref if scan is nearest else self._lookup_T_map_lidar(scan)
            if T_scan is None:
                continue
            chunks.append(transform_points(self._cloud_to_numpy(scan), T_scan))

        points_map = np.concatenate(chunks, axis=0)
        return transform_points(points_map, np.linalg.inv(T_ref)), T_ref

    def _lookup_T_map_lidar(self, cloud_msg: PointCloud2):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                cloud_msg.header.frame_id,
                Time.from_msg(cloud_msg.header.stamp),
            )   # non-blocking: a frame whose TF is not there yet stays in the pending queue
        except TransformException as exc:
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return None
        return transform_stamped_to_matrix(tf_msg)

    def _cloud_to_numpy(self, cloud_msg: PointCloud2) -> np.ndarray:
        # read_points_numpy requires every field to share one dtype; Livox PointCloud2
        # mixes float32 xyz with uint8 tag/line and float64 timestamp, so read structured.
        pts = point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        return np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float64).reshape(-1, 3)

    def _pc2(self, header, points_xyz: np.ndarray) -> PointCloud2:
        return point_cloud2.create_cloud_xyz32(header, points_xyz.astype(np.float32).tolist())

    def _to_msg(self, obs: Observation3D, header, object_id=-1, num_observations=1):
        msg = ObjectObservation3D()
        msg.header = header
        msg.detection_id = int(obs.detection_id)
        msg.object_id = int(object_id)
        msg.label = obs.label
        msg.confidence = float(obs.confidence)
        msg.points = self._pc2(header, obs.points_map)
        msg.centroid = Point(
            x=float(obs.centroid[0]),
            y=float(obs.centroid[1]),
            z=float(obs.centroid[2]),
        )
        msg.size = Vector3(
            x=float(obs.size[0]),
            y=float(obs.size[1]),
            z=float(obs.size[2]),
        )
        msg.yaw = float(obs.yaw)
        msg.clip_feature = np.asarray(obs.clip_feature, dtype=np.float32).tolist()
        msg.num_observations = int(num_observations)
        center = obs.bbox_center if obs.bbox_center is not None else obs.centroid
        msg.bbox_center = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        return msg

    def _observed_msg(self, obs: Observation3D) -> ObservedObject:
        msg = ObservedObject()
        msg.is_static = obs.label != self.dynamic_label
        msg.object_id = -1
        if msg.is_static and self.prior_map is not None:
            k = self.prior_map.associate(obs)[0]
            if k is not None:
                msg.object_id = int(self.prior_map.objects[k].object_id)
        msg.label = obs.label
        msg.confidence = float(obs.confidence)
        msg.centroid = Point(x=float(obs.centroid[0]), y=float(obs.centroid[1]), z=float(obs.centroid[2]))
        center = obs.bbox_center if obs.bbox_center is not None else obs.centroid
        half = 0.5 * float(obs.yaw)
        msg.bbox_center = Pose(position=Point(x=float(center[0]), y=float(center[1]), z=float(center[2])),
                               orientation=Quaternion(z=float(np.sin(half)), w=float(np.cos(half))))
        msg.bbox_size = Vector3(x=float(obs.size[0]), y=float(obs.size[1]), z=float(obs.size[2]))
        msg.clip_feature = l2_normalize(obs.clip_feature).astype(np.float32).tolist()
        return msg

    def _map_obj_to_msg(self, obj, header):
        obs = Observation3D(
            detection_id=-1,
            label=obj.label,
            confidence=obj.confidence,
            points_map=obj.points_map,
            clip_feature=obj.clip_feature,
            centroid=obj.centroid,
            size=obj.size,
            yaw=obj.yaw,
            bbox_center=obj.bbox_center,
        )
        return self._to_msg(
            obs,
            header,
            object_id=obj.object_id,
            num_observations=obj.num_observations,
        )

    def _semantic_cb(self, msg: SemanticObject2DArray):
        if self.K is None:
            self.get_logger().warning("CameraInfo not received yet")
            return
        self.pending.append(msg)                # processed in arrival order by _drain_pending
        while len(self.pending) > 50:
            self.pending.popleft()
            self.get_logger().warning("Pending detections overflow (no LiDAR?): dropping the oldest")
        self._drain_pending()

    def _process_semantic(self, msg: SemanticObject2DArray):
        t0 = time.perf_counter()
        result = self._process_semantic_impl(msg)
        if result != "retry":
            self._stats.append(time.perf_counter() - t0)
            if len(self._stats) >= 20:
                self.get_logger().info(f"fusion: {1000 * np.mean(self._stats):.0f} ms/frame (max {1000 * max(self._stats):.0f}), "
                                       f"{len(self.feature_map.objects)} map objects, {len(self.pending)} waiting")
                self._stats = []
        return result

    def _process_semantic_impl(self, msg: SemanticObject2DArray):
        camera_t = stamp_to_sec(msg.header.stamp)
        queue = self._snapshot(self.lidar_queue)
        lag = queue[-1][0] - camera_t if queue else 0.0
        if lag > self.buffer_sec:
            self.get_logger().warning(f"fusion is {lag:.1f} s behind the sensors (CPU overloaded or replay too fast)",
                                      throttle_duration_sec=10.0)
        cloud_msg = self._nearest_lidar(camera_t)
        if cloud_msg is None:
            queue = self._snapshot(self.lidar_queue)
            oldest = queue[0][0] if queue else float("nan")
            self.get_logger().warning(
                f"No LiDAR scan within max_sync_dt_sec of t={camera_t:.2f} (buffer starts at {oldest:.2f})"
            )
            return

        lidar_points, T_map_lidar = self._accumulated_points(camera_t, cloud_msg)
        if T_map_lidar is None:
            # TF of the scan can arrive after the scan itself: wait until the sensors are 2 s past the frame
            queue = self._snapshot(self.lidar_queue)
            if queue and queue[-1][0] - camera_t < 2.0 + self.accumulation_window:
                return "retry"
            self.get_logger().warning(f"TF for the scan at t={camera_t:.2f} never arrived: frame dropped")
            return

        obs_array = ObservedObjectArray()
        obs_array.header = msg.header
        obs_array.header.frame_id = self.map_frame
        obs_array.clip_model = self.clip_model

        masks = [self.bridge.imgmsg_to_cv2(o.mask, desired_encoding="mono8") for o in msg.objects]
        keep = range(len(masks))
        if self.dedup_iou > 0 and masks:
            keep = dedup_detections(masks, [o.confidence for o in msg.objects], self.dedup_iou)

        static_obs = []
        depth = False   # loaded on first fallback
        for k in keep:
            obj2d, mask = msg.objects[k], masks[k]
            if obj2d.label in self.targets and obj2d.confidence < self.target_min_conf:
                continue
            points_obj_lidar = points_inside_mask(
                lidar_points,
                erode_mask(mask, self.mask_erosion_px),
                self.T_cam_lidar,
                self.K,
            )

            # Camera-depth gate removes background/foreground points that project inside the mask,
            # then per-observation DBSCAN keeps the largest cluster.
            points_obj_lidar = depth_gate(
                points_obj_lidar,
                self.T_cam_lidar,
                self.gate_k_mad,
                self.gate_min_band,
            )
            if self.obs_cluster_eps > 0:
                points_obj_lidar = largest_cluster(points_obj_lidar, self.obs_cluster_eps)
            if len(points_obj_lidar) < 3 and self.depth_topic:
                if depth is False:
                    depth = self._depth_at(camera_t)
                if depth is not None:
                    points_obj_lidar = depth_gate(
                        depth_points_in_mask(depth, mask, self.K, self.T_cam_lidar, stride=self.depth_stride),
                        self.T_cam_lidar,
                        self.gate_k_mad,
                        self.gate_min_band,
                    )
                    points_obj_lidar = voxel_downsample(points_obj_lidar, self.voxel_size)
                    if self.obs_cluster_eps > 0:
                        points_obj_lidar = largest_cluster(points_obj_lidar, self.obs_cluster_eps)
            if len(points_obj_lidar) < 3:
                continue
            points_obj_lidar = remove_outliers(points_obj_lidar, **self.outliers)

            points_obj_map = transform_points(points_obj_lidar, T_map_lidar)
            geom = compute_geometry(points_obj_map)

            clip_feature = np.asarray(obj2d.clip_feature, dtype=np.float64)
            obs = Observation3D(
                detection_id=obj2d.detection_id,
                label=obj2d.label,
                confidence=obj2d.confidence,
                points_map=points_obj_map,
                clip_feature=clip_feature,
                centroid=geom.centroid,
                size=geom.size,
                yaw=geom.yaw,
                bbox_center=geom.bbox_center,
            )

            # Publish every 3D Object Observation.
            obs_array.objects.append(self._observed_msg(obs))

            # Proposal: person -> dynamic tracking; all others -> static multi-view fusion.
            if obs.label != self.dynamic_label:
                static_obs.append(obs)

        if self.frame_one_to_one:
            results = self.feature_map.update_frame(static_obs)
        else:
            results = [self.feature_map.update(o) for o in static_obs]
        for map_obj, created, score, phi_sem, phi_geo in results:
            self.get_logger().debug(
                f"object={map_obj.object_id} created={created} "
                f"score={score:.3f} sem={phi_sem:.3f} geo={phi_geo:.3f}"
            )

        # Online observation O_k for Target Matching: published for every processed frame (empty if nothing seen).
        self.obs_pub.publish(obs_array)

        self._map_dirty = self._json_dirty = True
        if self.map_period <= 0:
            self._publish_map()
        if self.scene_graph_json_path and self.export_period <= 0:
            self.export_scene_graph()


def main(args=None):
    rclpy.init(args=args)
    node = GeometryFusionNode()
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):   # RCLError: context shut down by SIGINT
        pass
    finally:
        # Offline mapping: the final map is what the Scene Graph uses, so always write it on exit.
        if node.scene_graph_json_path and node.feature_map.objects:
            n = node.export_scene_graph()
            print(f"[geometry_fusion_node] final scene graph: {n} objects -> {node.scene_graph_json_path}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
