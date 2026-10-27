"""Export fused static map objects as the Object-Centric Feature JSON consumed by the Scene Graph."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable

from .feature_fusion import MapObject


def _r(x, nd=4):
    return round(float(x), nd)


def map_object_to_dict(obj: MapObject) -> dict:
    return {
        "object_id": int(obj.object_id),
        "semantic_tag": obj.label,
        "confidence": _r(obj.confidence),
        "centroid": {"x": _r(obj.centroid[0]), "y": _r(obj.centroid[1]), "z": _r(obj.centroid[2])},
        "bbox_3d": {
            "center": [_r(v) for v in obj.bbox_center],
            "size": [_r(v) for v in obj.size],
            "yaw": _r(obj.yaw),
        },
        "clip_feature": [_r(v, 6) for v in obj.clip_feature],
        "num_observations": int(obj.num_observations),
    }


def scene_graph_features(objects: Iterable[MapObject], frame_id: str = "map", min_observations: int = 1) -> dict:
    return {
        "frame_id": frame_id,
        "objects": [map_object_to_dict(o) for o in objects if o.num_observations >= min_observations],
    }


def save_scene_graph_json(path: str, objects: Iterable[MapObject], frame_id: str = "map", min_observations: int = 1):
    """Atomic write so a reader never sees a half-written file."""
    data = scene_graph_features(objects, frame_id, min_observations)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o644)    # mkstemp creates 0600; the scene graph process may run as another user
    os.replace(tmp, path)
    return data
