"""Pre-mapped objects (Scene Graph objects.json) for giving online observations their persistent object_id."""
from __future__ import annotations

import json
import os

import numpy as np

from .association import l2_normalize
from .feature_fusion import MapObject, ObjectFeatureMap


def box_fill_points(center, size, yaw, spacing: float, max_points: int = 20000) -> np.ndarray:
    """Grid points filling an oriented box (stand-in for P_j when objects.json comes without points/)."""
    size = np.maximum(np.asarray(size, dtype=np.float64), 1e-3)
    spacing = max(spacing, float(np.prod(size) / max_points) ** (1.0 / 3.0))
    axes = [np.linspace(-s / 2, s / 2, max(2, int(np.ceil(s / spacing)) + 1)) for s in size]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.asarray(center, dtype=np.float64)


def load_prior_map(path: str, nn_distance_threshold: float, association_threshold: float,
                   voxel_size: float) -> ObjectFeatureMap:
    """
    objects.json (v0.8 schema) as a read-only ObjectFeatureMap: online observations are scored against it with the
    same phi_sem + phi_geo as mapping. P_j is points/obj_<id>.npy next to the JSON (offline dataset layout) when
    present, else the bbox filled at voxel spacing (online node export has no points).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "points")
    spacing = voxel_size if voxel_size > 0 else nn_distance_threshold
    prior = ObjectFeatureMap(nn_distance_threshold, association_threshold, voxel_size=voxel_size)
    for o in data["objects"]:
        box = o["bbox_3d"]
        npy = os.path.join(points_dir, f"obj_{int(o['object_id']):04d}.npy")
        pts = np.load(npy) if os.path.isfile(npy) else box_fill_points(box["center"], box["size"], box["yaw"], spacing)
        c = o["centroid"]
        prior.objects.append(MapObject(
            object_id=int(o["object_id"]),
            label=o["semantic_tag"],
            points_map=np.asarray(pts, dtype=np.float64).reshape(-1, 3),
            clip_feature=l2_normalize(o["clip_feature"]),
            num_observations=int(o.get("num_observations", 1)),
            centroid=np.array([c["x"], c["y"], c["z"]], dtype=np.float64),
            size=np.asarray(box["size"], dtype=np.float64),
            yaw=float(box["yaw"]),
            bbox_center=np.asarray(box["center"], dtype=np.float64),
            confidence=float(o.get("confidence", 1.0)),
            label_counts={o["semantic_tag"]: 1},
        ))
    return prior
