import numpy as np
from .feature_fusion import Observation3D, ObjectFeatureMap, apply_size_prior
import json
from .geometry import compute_geometry, depth_gate, largest_cluster, depth_points_in_mask, transform_points
from .scene_graph_export import scene_graph_features

NN_DISTANCE_THRESHOLD = 0.05
ASSOCIATION_THRESHOLD = 1.2
VOXEL_SIZE = 0.05


def make_obs(det_id, label, conf, points, feature):
    g = compute_geometry(points)
    return Observation3D(
        detection_id=det_id,
        label=label,
        confidence=conf,
        points_map=points,
        clip_feature=feature,
        centroid=g.centroid,
        size=g.size,
        yaw=g.yaw,
    )


def sparse_face_scan(rng, dist, noise=0.02):
    """Front face (0.4 m x 0.4 m) of a box seen by one Mid-360 scan (~0.94 pts/deg^2)."""
    deg = np.degrees(0.4 / dist)
    n = max(3, int(0.94 * deg * deg))
    pts = np.c_[np.full(n, dist), rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n)]
    pts[:, 0] += rng.normal(0.0, noise, n)
    return pts


def main():
    rng = np.random.default_rng(7)
    p1 = rng.normal([1.0, 2.0, 0.5], [0.006, 0.006, 0.006], size=(500, 3))
    p2 = p1 + rng.normal(0.0, 0.003, size=p1.shape)
    p3 = p1 + np.array([1.0, 0.0, 0.0])

    f1 = rng.normal(size=1024); f1 /= np.linalg.norm(f1)
    f2 = f1 + rng.normal(0.0, 0.01, size=1024); f2 /= np.linalg.norm(f2)
    f3 = rng.normal(size=1024); f3 /= np.linalg.norm(f3)

    fmap = ObjectFeatureMap(NN_DISTANCE_THRESHOLD, ASSOCIATION_THRESHOLD, VOXEL_SIZE)
    a = make_obs(0, "chair", 0.90, p1, f1)
    b = make_obs(1, "chair", 0.88, p2, f2)
    c = make_obs(2, "table", 0.91, p3, f3)

    obj_a, created_a, *_ = fmap.update(a)
    obj_b, created_b, score_b, sem_b, geo_b = fmap.update(b)
    obj_c, created_c, score_c, sem_c, geo_c = fmap.update(c)

    print(f"same: score={score_b:.4f} sem={sem_b:.4f} geo={geo_b:.4f} new={created_b}")
    print(f"far : score={score_c:.4f} sem={sem_c:.4f} geo={geo_c:.4f} new={created_c}")
    print("objects:", len(fmap.objects))
    print("fused norm:", np.linalg.norm(obj_b.clip_feature))

    assert created_a is True
    assert created_b is False
    assert created_c is True
    assert len(fmap.objects) == 2
    assert abs(np.linalg.norm(obj_b.clip_feature) - 1.0) < 1e-6

    # Sparse LiDAR: two independent Mid-360-like scans of the same object at 2.5 m must associate.
    smap = ObjectFeatureMap(NN_DISTANCE_THRESHOLD, ASSOCIATION_THRESHOLD, VOXEL_SIZE)
    s1, s2 = sparse_face_scan(rng, 2.5), sparse_face_scan(rng, 2.5)
    smap.update(make_obs(0, "dresser", 0.8, s1, f1))
    _, created_s, score_s, sem_s, geo_s = smap.update(make_obs(1, "dresser", 0.8, s2, f2))
    print(f"sparse: pts={len(s1)} score={score_s:.4f} sem={sem_s:.4f} geo={geo_s:.4f} new={created_s}")
    assert created_s is False

    # Voxel downsampling keeps repeated observations of one object bounded.
    raw_total = len(s1) + len(s2)
    for i in range(20):
        scan = sparse_face_scan(rng, 2.5)
        raw_total += len(scan)
        smap.update(make_obs(2 + i, "dresser", 0.8, scan, f2))
    n_pts = len(smap.objects[0].points_map)
    print(f"voxel: objects={len(smap.objects)} points after 22 obs={n_pts} (raw {raw_total})")
    assert len(smap.objects) == 1
    assert n_pts < 0.3 * raw_total

    # Depth gate drops a background point projected inside the mask (identity extrinsic: z = x_cam).
    obj_pts = np.c_[rng.normal(0, 0.05, (30, 2)), rng.normal(2.0, 0.01, 30)]
    with_bg = np.vstack([obj_pts, [[0.0, 0.0, 3.5]]])
    kept = depth_gate(with_bg, np.eye(4), k_mad=2.5, min_band=0.05)
    print(f"depth gate: {len(with_bg)} -> {len(kept)}")
    assert len(kept) == 30

    # Label majority vote + mean confidence + Scene Graph JSON schema.
    vmap = ObjectFeatureMap(NN_DISTANCE_THRESHOLD, ASSOCIATION_THRESHOLD, VOXEL_SIZE)
    for k, (lab, conf) in enumerate([("cabinet", 0.4), ("dresser", 0.6), ("dresser", 0.8)]):
        vmap.update(make_obs(k, lab, conf, p1 + rng.normal(0, 0.003, p1.shape), f1))
    v = vmap.objects[0]
    print(f"label vote: {v.label_counts} -> {v.label}, mean conf={v.confidence:.3f}")
    assert v.label == "dresser" and abs(v.confidence - 0.6) < 1e-9
    sg = json.loads(json.dumps(scene_graph_features(vmap.objects)))
    o = sg["objects"][0]
    assert sg["frame_id"] == "map"
    assert set(o) == {"object_id", "semantic_tag", "confidence", "centroid", "bbox_3d", "clip_feature", "num_observations"}
    assert set(o["centroid"]) == {"x", "y", "z"} and set(o["bbox_3d"]) == {"center", "size", "yaw"}
    assert len(o["bbox_3d"]["center"]) == 3 and len(o["bbox_3d"]["size"]) == 3 and len(o["clip_feature"]) == 1024
    print("scene graph json: schema OK")

    # Per-observation DBSCAN keeps the object and drops a separate background chunk in the same mask.
    chunk = obj_pts + np.array([0.0, 0.0, 0.6])
    kept = largest_cluster(np.vstack([obj_pts, chunk[:10]]), eps=0.15)
    print(f"largest cluster: {len(obj_pts) + 10} -> {len(kept)}")
    assert len(kept) == len(obj_pts)

    # Depth fallback: a flat depth of 1.5 m inside a mask back-projects to points at z=1.5 in the camera
    # frame, returned in the sensor frame (round trip through T_cam_sensor).
    Kd = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    T_cs = np.eye(4); T_cs[:3, 3] = [0.0, 0.3, -0.05]
    depth = np.full((480, 640), 1.5); m = np.zeros((480, 640), np.uint8); m[200:260, 300:340] = 1
    ps = depth_points_in_mask(depth, m, Kd, T_cs)
    pc = transform_points(ps, T_cs)
    print(f"depth fallback: {len(ps)} points, z={pc[:, 2].min():.2f}..{pc[:, 2].max():.2f}")
    assert len(ps) > 20 and np.allclose(pc[:, 2], 1.5)
    assert len(depth_points_in_mask(np.zeros_like(depth), m, Kd, T_cs)) == 0

    # Size prior: a 2 m "remote control" takes the next plausible voted label; nothing plausible -> other item.
    from copy import deepcopy
    big = deepcopy(v); big.size = np.array([0.9, 0.2, 2.0]); big.label = "remote control"
    big.label_counts = {"remote control": 5, "door": 3}
    assert apply_size_prior(big).label == "door"
    big.label_counts = {"remote control": 5}
    assert apply_size_prior(big).label == "other item" and big.label == "remote control"
    print("size prior: OK")

    # Search-target priority: a target label seen once beats a majority synonym; clearing targets restores majority.
    tmap = ObjectFeatureMap(0.05, 1.2, voxel_size=0.05)
    feat_t = np.zeros(1024); feat_t[0] = 1.0
    for lab in ["remote", "remote", "remote control", "remote"]:
        tmap.update(Observation3D(0, lab, 0.5, obj_pts, feat_t, obj_pts.mean(0), np.array([0.2, 0.05, 0.03]), 0.0))
    assert len(tmap.objects) == 1 and tmap.objects[0].label == "remote"
    tmap.set_priority_labels(["remote control"])
    assert tmap.objects[0].label == "remote control"
    tmap.set_priority_labels([])
    assert tmap.objects[0].label == "remote"
    print("target priority: OK")

    # Fragment merge: two slices of one object with the same feature merge; a neighbour with another feature does not.
    from .feature_fusion import merge_fragments
    fm = ObjectFeatureMap(0.05, 1.1, voxel_size=0.05)
    fa = np.zeros(1024); fa[0] = 1.0; fb = np.zeros(1024); fb[1] = 1.0
    front = np.c_[rng.uniform(0, 0.5, (40, 1)), np.zeros((40, 1)), rng.uniform(0, 0.8, (40, 1))]
    back = front + np.array([0.08, 0.10, 0.0])     # second slice of the same object (run2 splits: 0.05-0.12 m)
    other = front + np.array([0.0, 0.9, 0.0])
    for pts, f, lab in ((front, fa, "chair"), (back, fa, "swivel chair"), (other, fb, "desk")):
        g = compute_geometry(pts); fm._create(Observation3D(0, lab, 0.5, pts, f, g.centroid, g.size, g.yaw, g.bbox_center))
    merged = merge_fragments(fm.objects, 0.05, min_cos=0.9)
    print(f"fragment merge: {len(fm.objects)} -> {len(merged)} ({[o.label for o in merged]})")
    assert len(merged) == 2 and len(fm.objects) == 3 and merged[0].num_observations == 2
    # Two identical chairs side by side (same feature, touching): the merged box would double -> kept apart.
    fm2 = ObjectFeatureMap(0.05, 1.1, voxel_size=0.05)
    chair = np.c_[rng.uniform(0, 0.5, (60, 1)), rng.uniform(0, 0.5, (60, 1)), rng.uniform(0, 0.8, (60, 1))]
    for pts in (chair, chair + np.array([0.55, 0.0, 0.0])):
        g = compute_geometry(pts); fm2._create(Observation3D(0, "chair", 0.5, pts, fa, g.centroid, g.size, g.yaw, g.bbox_center))
    assert len(merge_fragments(fm2.objects, 0.05, min_cos=0.9)) == 2
    print("fragment merge growth gate: OK")

    # Stray point (require3 problem 1, run4 desk): one point 11 cm above a desk top on the 5 cm voxel grid makes the
    # box 14 cm too tall; remove_outliers drops it, keeps the surface (edges included) and leaves small clouds alone.
    from .geometry import remove_outliers
    from .feature_fusion import denoise_object
    gx, gy = np.meshgrid(np.arange(0, 1.6, 0.05), np.arange(0, 0.9, 0.05))
    top = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 0.30)] + rng.normal(0, 0.005, (gx.size, 3))
    desk = np.vstack([top, [[0.80, 0.45, 0.44]]])
    clean = remove_outliers(desk)
    assert len(clean) == len(top) and abs(clean[:, 2].max() - top[:, 2].max()) < 1e-9
    small = np.vstack([top[:10], [[0.80, 0.45, 0.44]]])
    assert len(remove_outliers(small)) == len(small)
    dm = ObjectFeatureMap(0.05, 1.1, voxel_size=0.05)
    g = compute_geometry(desk); dm._create(Observation3D(0, "desk", 0.5, desk, fa, g.centroid, g.size, g.yaw, g.bbox_center))
    out = denoise_object(dm.objects[0], 0.20, outliers=dict(k=8, std_ratio=2.0, min_gap=0.08, min_points=20))
    print(f"stray point: box height {g.size[2]:.2f} -> {out.size[2]:.2f} m")
    assert out.size[2] < 0.05 and denoise_object(dm.objects[0], 0.20).size[2] > 0.1
    print("stray point removal: OK")

    print("SYNTHETIC TEST: PASS")


if __name__ == "__main__":
    main()
