from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Geometry3D:
    points_map: np.ndarray      # (N, 3)
    centroid: np.ndarray        # (3,)
    size: np.ndarray            # length, width, height
    yaw: float                  # BEV principal-axis yaw [rad]
    bbox_center: np.ndarray     # 3D box center (differs from the point-mean centroid)


def transform_points(points_xyz: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to Nx3 points."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if points_xyz.size == 0:
        return points_xyz.copy()

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    return (T_dst_src @ points_h.T).T[:, :3]


def project_lidar_to_image(
    points_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
):
    """
    Proposal equations:
      p_C = T_C_L p_L
      u = fx * x_C / z_C + cx
      v = fy * y_C / z_C + cy

    Returns:
      uv:       (N, 2) floating pixel coordinates
      z_cam:    (N,) camera-frame depth
      points_c: (N, 3) camera-frame points
    """
    points_c = transform_points(points_lidar, T_cam_lidar)
    z = points_c[:, 2]

    uv = np.full((points_c.shape[0], 2), np.nan, dtype=np.float64)
    valid_z = z > 0.0
    if np.any(valid_z):
        x = points_c[valid_z, 0]
        y = points_c[valid_z, 1]
        zv = z[valid_z]
        uv[valid_z, 0] = K[0, 0] * (x / zv) + K[0, 2]
        uv[valid_z, 1] = K[1, 1] * (y / zv) + K[1, 2]
    return uv, z, points_c


def points_inside_mask(
    points_lidar: np.ndarray,
    mask: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Implements A_{t,i}: LiDAR points whose projected pixels lie inside
    the object mask, with positive camera-frame depth and valid image bounds.
    No DBSCAN/clustering is applied.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")

    H, W = mask.shape
    uv, z_cam, _ = project_lidar_to_image(points_lidar, T_cam_lidar, K)

    finite = np.isfinite(uv).all(axis=1)
    u = np.zeros(len(points_lidar), dtype=np.int64)
    v = np.zeros(len(points_lidar), dtype=np.int64)
    u[finite] = np.rint(uv[finite, 0]).astype(np.int64)
    v[finite] = np.rint(uv[finite, 1]).astype(np.int64)

    valid = (
        finite
        & (z_cam > 0.0)
        & (u >= 0) & (u < W)
        & (v >= 0) & (v < H)
    )

    selected = np.zeros_like(valid)
    idx = np.flatnonzero(valid)
    selected[idx] = mask[v[idx], u[idx]] > 0
    return points_lidar[selected]


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink the mask boundary by `pixels` to drop background points at mask edges."""
    if pixels <= 0:
        return mask
    from scipy.ndimage import binary_erosion
    size = 2 * int(pixels) + 1
    return binary_erosion(mask > 0, structure=np.ones((size, size), dtype=bool))


def depth_gate(
    points_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    k_mad: float,
    min_band: float,
) -> np.ndarray:
    """
    Keep points whose camera-frame depth lies within median +- max(k_mad * MAD, min_band).
    Removes background/foreground points that fall inside the mask (lightweight
    replacement for DBSCAN, which is unreliable on sparse single-scan LiDAR points).
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if len(points_lidar) == 0:
        return points_lidar
    z = transform_points(points_lidar, T_cam_lidar)[:, 2]
    median = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - median))
    return points_lidar[np.abs(z - median) <= max(k_mad * mad, min_band)]


def depth_points_in_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_sensor: np.ndarray,
    stride: int = 4,
    erosion: int = 2,
    z_range: tuple = (0.2, 4.0),
) -> np.ndarray:
    """
    Fallback geometry for objects the LiDAR cannot see (e.g. small items on a table top that lies
    above the Mid-360): back-project the aligned camera depth inside the (eroded) mask, every
    `stride` pixels, and return the points in the sensor frame so the usual gate / clustering applies.
    """
    m = erode_mask(mask, erosion) if erosion > 0 else mask > 0
    sub = np.zeros_like(m, dtype=bool)
    sub[::stride, ::stride] = True
    v, u = np.nonzero(m & sub)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth_m[v, u].astype(np.float64)
    ok = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    u, v, z = u[ok], v[ok], z[ok]
    pc = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
    return transform_points(pc, np.linalg.inv(T_cam_sensor))


def largest_cluster(points: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    """
    Proposal 2-3 per-observation clustering (DBSCAN semantics): core points have >= min_samples
    neighbours within eps; clusters are eps-connected core points plus their border points.
    Returns the points of the largest cluster (empty if none). Removes background / neighbour
    points that leak into a mask and would otherwise bridge separate objects during fusion.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < min_samples:
        return pts[:0]
    tree = cKDTree(pts)
    core = np.array([len(n) >= min_samples for n in tree.query_ball_point(pts, eps)])
    if not core.any():
        return pts[:0]
    idx = np.flatnonzero(core)
    pairs = cKDTree(pts[idx]).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
    _, lab_core = connected_components(graph, directed=False)
    labels = np.full(len(pts), -1)
    labels[idx] = lab_core
    # border points join the cluster of their nearest core point within eps
    d, j = cKDTree(pts[idx]).query(pts[~core], distance_upper_bound=eps)
    ok = np.isfinite(d)
    border = np.flatnonzero(~core)
    labels[border[ok]] = lab_core[j[ok]]
    valid = labels >= 0
    largest = np.bincount(labels[valid]).argmax()
    return pts[labels == largest]


def remove_outliers(points: np.ndarray, k: int = 8, std_ratio: float = 2.0, min_gap: float = 0.08,
                    min_points: int = 20) -> np.ndarray:
    """
    Stray-point removal before the bbox (require3 problem 1: one point 11 cm above a desk top made the box 14 cm
    too tall). Statistical outlier removal (mean distance to the k nearest neighbours > mean + std_ratio * std,
    as Open3D remove_statistical_outlier) restricted to isolated points (nearest neighbour farther than min_gap,
    i.e. beyond the 5 cm voxel spacing), so surface edges and thin parts stay. Clouds with fewer than min_points
    points are returned unchanged (too few for statistics: cups, remotes). k <= 0 disables.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if k <= 0 or len(pts) < max(min_points, k + 1):
        return pts
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    bad = (mean_d > mean_d.mean() + std_ratio * mean_d.std()) & (d[:, 1] > min_gap)
    return pts[~bad]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep the first point per voxel; voxel_size <= 0 disables downsampling."""
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def compute_geometry(points_map: np.ndarray) -> Geometry3D:
    """
    Proposal-aligned geometry:
      - centroid = mean of map-frame object points
      - L/W/H from point spatial extent in principal-axis frame
      - yaw from BEV principal axis
    """
    pts = np.asarray(points_map, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_map must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("At least 3 points are required for geometry estimation")

    centroid = pts.mean(axis=0)
    xy = pts[:, :2] - centroid[:2]

    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = float(np.arctan2(principal[1], principal[0]))

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy_local = xy @ R

    min_xy = xy_local.min(axis=0)
    max_xy = xy_local.max(axis=0)
    length, width = max_xy - min_xy
    height = pts[:, 2].max() - pts[:, 2].min()

    center_xy = centroid[:2] + R @ (0.5 * (min_xy + max_xy))
    center_z = 0.5 * (pts[:, 2].max() + pts[:, 2].min())

    return Geometry3D(
        points_map=pts,
        centroid=centroid,
        size=np.array([length, width, height], dtype=np.float64),
        yaw=yaw,
        bbox_center=np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64),
    )
