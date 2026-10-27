from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(f)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero CLIP feature")
    return f / norm


def semantic_similarity(feature_i: np.ndarray, feature_j: np.ndarray) -> float:
    """
    Proposal:
      phi_sem(i,j) = (f_i^T f_j + 1) / 2, range [0,1]
    assuming L2-normalized CLIP features.
    """
    fi = l2_normalize(feature_i)
    fj = l2_normalize(feature_j)
    cosine = float(np.clip(np.dot(fi, fj), -1.0, 1.0))
    return 0.5 * (cosine + 1.0)


def geometric_similarity(
    points_i: np.ndarray,
    points_j: np.ndarray,
    nn_distance_threshold: float,
) -> float:
    """
    Proposal:
      phi_geo(i,j) = fraction of points in current object i whose nearest
      point in map object j is within delta_nn.

    No DBSCAN/clustering is used.
    """
    pi = np.asarray(points_i, dtype=np.float64)
    pj = np.asarray(points_j, dtype=np.float64)
    if len(pi) == 0 or len(pj) == 0:
        return 0.0
    if nn_distance_threshold <= 0.0:
        raise ValueError("nn_distance_threshold must be > 0")

    tree = cKDTree(pj)
    distances, _ = tree.query(pi, k=1)
    return float(np.mean(distances < nn_distance_threshold))


def association_score(phi_sem: float, phi_geo: float) -> float:
    """Proposal text uses the sum of semantic and geometric similarity."""
    return float(phi_sem + phi_geo)
