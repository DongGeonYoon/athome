from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .association import (
    l2_normalize,
    semantic_similarity,
    geometric_similarity,
    association_score,
)
from .geometry import compute_geometry, remove_outliers, voxel_downsample


@dataclass
class Observation3D:
    detection_id: int
    label: str
    confidence: float
    points_map: np.ndarray
    clip_feature: np.ndarray
    centroid: np.ndarray
    size: np.ndarray
    yaw: float
    bbox_center: Optional[np.ndarray] = None  # computed from points_map when omitted


@dataclass
class MapObject:
    object_id: int
    label: str                  # majority vote over label_counts (a current search target wins if present)
    points_map: np.ndarray
    clip_feature: np.ndarray
    num_observations: int
    centroid: np.ndarray
    size: np.ndarray
    yaw: float
    bbox_center: np.ndarray
    confidence: float           # mean detection confidence of fused observations
    label_counts: Dict[str, int] = field(default_factory=dict)


class ObjectFeatureMap:
    """
    Proposal-aligned static-object association and fusion.

    - semantic similarity: shifted cosine similarity in [0,1]
    - geometric similarity: nearest-neighbor match ratio
    - association score: phi_sem + phi_geo
    - feature fusion: running average weighted by observation count, then L2 normalize
    - geometry fusion: concatenate current observation points with existing map points,
      then voxel downsample (proposal) so map point clouds stay bounded

    Person observations are intentionally not inserted here; the proposal sends them
    to dynamic-object tracking instead.
    """

    def __init__(
        self,
        nn_distance_threshold: float,
        association_threshold: float,
        voxel_size: float = 0.0,
        growth_ratio: Optional[float] = None,
        growth_margin: float = 0.3,
        min_semantic_cos: Optional[float] = None,
    ):
        """
        growth_ratio: reject a match if the merged bbox diagonal exceeds
            growth_ratio * max(diag_obj, diag_obs) + growth_margin (stops adjacent objects
            from being absorbed into one growing object). None disables the gate.
        """
        if nn_distance_threshold <= 0.0:
            raise ValueError("nn_distance_threshold must be > 0")
        self.nn_distance_threshold = float(nn_distance_threshold)
        self.association_threshold = float(association_threshold)
        self.voxel_size = float(voxel_size)
        self.growth_ratio = None if growth_ratio is None else float(growth_ratio)
        self.growth_margin = float(growth_margin)
        # Semantic gate: never merge when the raw CLIP cosine is below this, however large phi_geo is
        # (stops cushion + sofa, chair + pillar merges driven by geometric contact alone).
        self.min_semantic_cos = None if min_semantic_cos is None else float(min_semantic_cos)
        self.objects: List[MapObject] = []
        self._next_id = 0
        self.priority_labels = set()

    def _pair_score(self, obs: Observation3D, obj: MapObject) -> Tuple[float, float, float]:
        phi_sem = semantic_similarity(obs.clip_feature, obj.clip_feature)
        phi_geo = geometric_similarity(
            obs.points_map,
            obj.points_map,
            self.nn_distance_threshold,
        )
        return association_score(phi_sem, phi_geo), phi_sem, phi_geo

    def _growth_ok(self, obs: Observation3D, obj: MapObject) -> bool:
        if self.growth_ratio is None:
            return True
        corners = np.vstack([_box_corners(obj.bbox_center, obj.size, obj.yaw),
                             _box_corners(_obs_center(obs), obs.size, obs.yaw)])
        merged = float(np.linalg.norm(corners.max(0) - corners.min(0)))
        largest = max(float(np.linalg.norm(obj.size)), float(np.linalg.norm(obs.size)))
        return merged <= self.growth_ratio * largest + self.growth_margin

    def associate(self, obs: Observation3D, exclude=()) -> Tuple[Optional[int], float, float, float]:
        """Best-scoring map object above the threshold that passes the size gate (indices in exclude skipped)."""
        best = (None, float("-inf"), 0.0, 0.0)
        reach = 0.5 * float(np.linalg.norm(obs.size)) + self.nn_distance_threshold
        for k, obj in enumerate(self.objects):
            if k in exclude:
                continue
            # phi_geo is exactly 0 when the two boxes cannot be within delta_nn: score <= 1 < threshold.
            if np.linalg.norm(obj.bbox_center - _obs_center(obs)) > reach + 0.5 * float(np.linalg.norm(obj.size)):
                if self.association_threshold > 1.0:
                    continue
            score, phi_sem, phi_geo = self._pair_score(obs, obj)
            if self.min_semantic_cos is not None and 2.0 * phi_sem - 1.0 < self.min_semantic_cos:
                continue
            if score > best[1]:
                if score >= self.association_threshold and not self._growth_ok(obs, obj):
                    continue
                best = (k, score, phi_sem, phi_geo)
        if best[0] is None or best[1] < self.association_threshold:
            return None, best[1], best[2], best[3]
        return best

    def _create(self, obs: Observation3D) -> MapObject:
        bbox_center = obs.bbox_center
        if bbox_center is None:
            bbox_center = compute_geometry(obs.points_map).bbox_center
        obj = MapObject(
            object_id=self._next_id,
            label=obs.label,
            points_map=voxel_downsample(obs.points_map, self.voxel_size).copy(),
            clip_feature=l2_normalize(obs.clip_feature),
            num_observations=1,
            centroid=np.asarray(obs.centroid, dtype=np.float64).copy(),
            size=np.asarray(obs.size, dtype=np.float64).copy(),
            yaw=float(obs.yaw),
            bbox_center=np.asarray(bbox_center, dtype=np.float64).copy(),
            confidence=float(obs.confidence),
            label_counts={obs.label: 1},
        )
        self._next_id += 1
        self.objects.append(obj)
        return obj

    def _fuse(self, obj: MapObject, obs: Observation3D) -> MapObject:
        n = obj.num_observations

        # Proposal feature fusion:
        # f_bar = (n * f_map + f_obs) / (n + 1)
        # f_map_new = f_bar / ||f_bar||_2
        fused = (n * obj.clip_feature + l2_normalize(obs.clip_feature)) / (n + 1)
        obj.clip_feature = l2_normalize(fused)

        # Proposal geometry fusion: accumulate the current observation's map-frame points,
        # then voxel downsample to remove duplicates from repeated observations.
        merged = np.concatenate([obj.points_map, obs.points_map], axis=0)
        obj.points_map = voxel_downsample(merged, self.voxel_size)
        obj.num_observations = n + 1

        # Semantic tag: CG-D labels the same object with synonyms (dresser / cabinet / drawer),
        # so keep per-label counts and use the majority (ties keep the earlier label).
        obj.label_counts[obs.label] = obj.label_counts.get(obs.label, 0) + 1
        obj.label = self._vote(obj)
        obj.confidence = (n * obj.confidence + float(obs.confidence)) / (n + 1)

        # Geometry from the merged cloud: a coarse voxel grid can leave < 3 points on small objects.
        geometry = compute_geometry(merged)
        obj.centroid = geometry.centroid
        obj.size = geometry.size
        obj.yaw = geometry.yaw
        obj.bbox_center = geometry.bbox_center
        return obj

    def _vote(self, obj: MapObject) -> str:
        """Majority label; ties keep the current label. A search target seen even once wins, so the target
        name is not outvoted by RAM synonyms (e.g. 'remote control' vs 'remote')."""
        counts = obj.label_counts
        targets = [lab for lab in counts if lab in self.priority_labels]
        pool = targets if targets else list(counts)
        best = max(counts[lab] for lab in pool)
        if obj.label in pool and counts[obj.label] == best:
            return obj.label
        return next(lab for lab in pool if counts[lab] == best)

    def set_priority_labels(self, labels) -> None:
        """Current search targets (/athome/search/targets); re-votes every object. Empty restores plain majority."""
        self.priority_labels = set(labels)
        for obj in self.objects:
            obj.label = self._vote(obj)

    def update(self, obs: Observation3D, exclude=()):
        match_index, score, phi_sem, phi_geo = self.associate(obs, exclude)
        if match_index is None:
            obj = self._create(obs)
            return obj, True, score, phi_sem, phi_geo

        obj = self._fuse(self.objects[match_index], obs)
        return obj, False, score, phi_sem, phi_geo

    def update_frame(self, observations: List[Observation3D]):
        """
        One frame of detections with one-to-one matching: a map object updated by one detection
        is not matched again in the same frame (two adjacent chairs in one image stay two objects).
        Higher-confidence detections are processed first.
        """
        results, used = [], set()
        for obs in sorted(observations, key=lambda o: -float(o.confidence)):
            obj, created, score, phi_sem, phi_geo = self.update(obs, exclude=used)
            used.add(self.objects.index(obj))
            results.append((obj, created, score, phi_sem, phi_geo))
        return results


def _obs_center(obs: Observation3D) -> np.ndarray:
    if obs.bbox_center is None:
        obs.bbox_center = compute_geometry(obs.points_map).bbox_center
    return np.asarray(obs.bbox_center, dtype=np.float64)


def _box_corners(center, size, yaw) -> np.ndarray:
    l, w, h = np.asarray(size, dtype=np.float64) / 2.0
    c, s_ = np.cos(yaw), np.sin(yaw)
    xy = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) @ np.array([[c, s_], [-s_, c]])
    xy = xy + np.asarray(center[:2])
    z = np.array([center[2] - h, center[2] + h])
    return np.array([[x, y, zz] for x, y in xy for zz in z])


def dedup_detections(masks, confidences, iou_threshold: float = 0.8) -> List[int]:
    """
    Indices of detections to keep within one frame: CG-D often returns the same object twice
    with different labels (dresser / cabinet). Masks overlapping by IoU >= threshold keep the
    higher-confidence one.
    """
    order = np.argsort(-np.asarray(confidences, dtype=np.float64))
    flat = [np.asarray(m, dtype=bool).ravel() for m in masks]
    area = [f.sum() for f in flat]
    keep: List[int] = []
    for i in order:
        if area[i] == 0:
            continue
        dup = False
        for j in keep:
            inter = np.logical_and(flat[i], flat[j]).sum()
            if inter / float(area[i] + area[j] - inter) >= iou_threshold:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return sorted(keep)


def denoise_object(obj: MapObject, eps: float, min_points: int = 3, outliers: Optional[dict] = None) -> MapObject:
    """
    Object-level cleanup before export: keep the largest eps-connected component of the fused
    point cloud (removes disconnected chunks absorbed by mistake), then drop isolated stray points
    (geometry.remove_outliers with the `outliers` kwargs, None skips) and recompute centroid / bbox.
    Returns a new MapObject; the map itself is not modified.
    """
    from copy import deepcopy
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    out = deepcopy(obj)
    pts = np.asarray(obj.points_map, dtype=np.float64)
    if len(pts) < 2 * min_points:
        return out
    pairs = cKDTree(pts).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    _, labels = connected_components(graph, directed=False)
    largest = np.bincount(labels).argmax()
    kept = pts[labels == largest]
    if outliers:
        kept = remove_outliers(kept, **outliers)
    if len(kept) < min_points:
        return out
    g = compute_geometry(kept)
    out.points_map, out.centroid, out.size, out.yaw, out.bbox_center = kept, g.centroid, g.size, g.yaw, g.bbox_center
    return out


# Upper bound of the largest bbox extent [m] for hand-sized / small categories. GroundingDINO sometimes
# grounds a small-object phrase on a large surface (e.g. a door labelled "remote control"); such a label
# cannot describe the fused geometry. Categories not listed are unconstrained.
SIZE_PRIOR_MAX = {
    'remote control': 0.35, 'remote': 0.35, 'cell phone': 0.3, 'phone': 0.3, 'mobile phone': 0.3, 'watch': 0.2,
    'keys': 0.2, 'key': 0.2, 'pen': 0.25, 'pencil': 0.25, 'spoon': 0.3, 'fork': 0.3, 'knife': 0.4,
    'cup': 0.25, 'coffee cup': 0.25, 'mug': 0.25, 'glass': 0.3, 'wine glass': 0.3, 'egg': 0.12, 'apple': 0.15,
    'potato': 0.2, 'tomato': 0.15, 'fruit': 0.3, 'candle': 0.4, 'soap': 0.3, 'spray bottle': 0.4,
    'bottle': 0.5, 'wine bottle': 0.5, 'water bottle': 0.5, 'bowl': 0.45, 'plate': 0.5, 'toilet paper': 0.3,
    'tissue box': 0.4, 'book': 0.6, 'clock': 0.7, 'alarm clock': 0.3, 'kettle': 0.45, 'teapot': 0.45,
    'pot': 0.6, 'vase': 0.9, 'toy': 0.7, 'laptop': 0.7, 'shoe': 0.45, 'door handle': 0.35, 'knob': 0.15,
    'video camera': 0.5, 'camera': 0.4, 'backpack': 0.8, 'basket': 0.9, 'box': 1.2, 'pillow': 1.0, 'cushion': 1.0,
    'computer monitor': 1.0, 'monitor': 1.0,
}


def apply_size_prior(obj: MapObject, priors=None, slack: float = 1.5, fallback: str = 'other item') -> MapObject:
    """
    Export-time label check: if the fused bbox is far larger than the voted category allows, use the
    most frequent label in label_counts that fits; if none fits, `fallback` (excluded from export).
    Returns a new MapObject; the map itself is not modified.
    """
    from copy import deepcopy
    priors = SIZE_PRIOR_MAX if priors is None else priors
    extent = float(np.max(obj.size))
    fits = lambda lab: lab.lower() not in priors or extent <= priors[lab.lower()] * slack
    if fits(obj.label):
        return obj
    out = deepcopy(obj)
    ok = sorted((c, lab) for lab, c in obj.label_counts.items() if fits(lab))
    out.label = ok[-1][1] if ok else fallback
    return out


def _inside_box(p, obj: MapObject, margin: float) -> bool:
    c, s, yaw = np.asarray(obj.bbox_center), np.asarray(obj.size), float(obj.yaw)
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    d = R @ (np.asarray(p[:2]) - c[:2])
    return bool(np.all(np.abs(d) <= s[:2] / 2 + margin) and abs(p[2] - c[2]) <= s[2] / 2 + margin)


def merge_fragments(objects: List[MapObject], voxel_size: float = 0.05, min_cos: float = 0.8, nn: float = 0.10,
                    min_overlap: float = 0.3, max_centre_dist: float = 0.2, max_growth: float = 1.25) -> List[MapObject]:
    """
    Export-time re-merge of one physical object that fusion kept as two map objects (seen from different sides,
    thin partial slices whose points never come within delta_nn, or a synonym label lowering phi_sem), in the
    spirit of ConceptGraphs' periodic merge of overlapping objects. Two objects are merged when their CLIP features
    agree (cosine >= min_cos) AND they share space: >= min_overlap of the smaller cloud lies within nn of the other
    cloud, or the smaller box centre lies inside the other box, or the box centres are <= max_centre_dist apart, AND
    the merged box stays the size of the larger part (largest extent <= max_growth x, +0.1 m): two fragments of one
    object barely grow it, two identical chairs side by side (same CLIP feature) double it (real run2: 1.6 m "chair").
    Most similar pair first, repeated until nothing merges. Returns new MapObjects; the map is not modified.
    """
    from copy import deepcopy
    from scipy.spatial import cKDTree
    objs = [deepcopy(o) for o in objects]
    while True:
        best = None
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                reach = 0.5 * (np.linalg.norm(a.size) + np.linalg.norm(b.size)) + nn
                if np.linalg.norm(np.asarray(a.bbox_center) - np.asarray(b.bbox_center)) > reach:
                    continue
                cos = float(np.dot(l2_normalize(a.clip_feature), l2_normalize(b.clip_feature)))
                if cos < min_cos:
                    continue
                small, big = (a, b) if len(a.points_map) <= len(b.points_map) else (b, a)
                overlap = float(np.mean(cKDTree(big.points_map).query(small.points_map)[0] <= nn)) if len(small.points_map) else 0.0
                close = np.linalg.norm(np.asarray(a.bbox_center) - np.asarray(b.bbox_center)) <= max_centre_dist
                if overlap >= min_overlap or _inside_box(small.bbox_center, big, 0.05) or close:
                    g = compute_geometry(np.concatenate([a.points_map, b.points_map]))
                    if max(g.size[:2]) > max_growth * max(max(a.size[:2]), max(b.size[:2])) + 0.1:
                        continue
                    if best is None or cos > best[0]:
                        best = (cos, i, j)
        if best is None:
            return objs
        _, i, j = best
        a, b = objs[i], objs[j]
        n = a.num_observations + b.num_observations
        a.clip_feature = l2_normalize(a.num_observations * a.clip_feature + b.num_observations * b.clip_feature)
        a.confidence = (a.num_observations * a.confidence + b.num_observations * b.confidence) / n
        for lab, c in b.label_counts.items():
            a.label_counts[lab] = a.label_counts.get(lab, 0) + c
        a.label = max(a.label_counts, key=lambda lab: (a.label_counts[lab], lab == a.label))
        a.points_map = voxel_downsample(np.concatenate([a.points_map, b.points_map]), voxel_size)
        a.num_observations, a.object_id = n, min(a.object_id, b.object_id)
        g = compute_geometry(a.points_map)
        a.centroid, a.size, a.yaw, a.bbox_center = g.centroid, g.size, g.yaw, g.bbox_center
        del objs[j]
