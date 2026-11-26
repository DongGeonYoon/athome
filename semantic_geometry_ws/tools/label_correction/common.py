"""Shared helpers for the label-correction step (keyframe source folder of robot_bag_extract.py + CG-D detections)."""
import gzip, json, os, pickle, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../src/semantic_geometry_fusion'))
sys.path.insert(0, os.path.join(HERE, '../scene_graph_dataset'))


def read_labels(path):
    """Corrected tags, one line per keyframe:  `<keyframe>: <detection_id>=<tag> ...`. Detections not listed are removed."""
    labels = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        k, rest = line.split(':', 1)
        labels[int(k)] = {int(a): b.strip() for a, b in re.findall(r'(\d+)=([a-z ]+?)(?=\s+\d+=|\s*$)', rest)}
    return labels


def read_config(path):
    import yaml
    return yaml.safe_load(open(path))


class Keyframes:
    """One scene folder: results/frameNNNNNN.jpg, traj.txt (camera poses), camera.json, lidar/frameNNNNNN.npz, gsa_detections_ram_<suffix>/."""

    def __init__(self, scene_dir, suffix='v2'):
        self.dir = scene_dir; self.suffix = suffix
        self.T = np.loadtxt(os.path.join(scene_dir, 'traj.txt')).reshape(-1, 4, 4)
        cam = json.load(open(os.path.join(scene_dir, 'camera.json')))
        self.K = np.array(cam['K'], float); self.Ki = np.linalg.inv(self.K); self.W, self.H = cam['width'], cam['height']
        self._det, self._cloud = {}, {}

    def det(self, k):
        if k not in self._det:
            self._det[k] = pickle.load(gzip.open(os.path.join(self.dir, f'gsa_detections_ram_{self.suffix}', f'frame{k:06d}.pkl.gz'), 'rb'))
        return self._det[k]

    def cloud(self, k):
        """LiDAR points of the keyframe in the map frame."""
        if k not in self._cloud:
            d = np.load(os.path.join(self.dir, 'lidar', f'frame{k:06d}.npz')); P = d['points'].astype(float); T = d['T_map_sensor']
            self._cloud[k] = (T[:3, :3] @ P.T).T + T[:3, 3]
        return self._cloud[k]

    def project(self, k, P):
        """Pixel coordinates (int), in-image flag and depth of map points P in keyframe k."""
        Ti = np.linalg.inv(self.T[k]); C = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]; z = C[:, 2]
        uv = np.full((len(P), 2), -1.0); ok = z > .15; uv[ok] = (self.K @ C[ok].T).T[:, :2] / z[ok, None]
        inn = ok & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        return uv.astype(int), inn, z

    def in_mask(self, k, mask, P):
        uv, inn, _ = self.project(k, P); h = np.zeros(len(P), bool); h[inn] = mask[uv[inn, 1], uv[inn, 0]]
        return h, inn

    def clip(self, k, i):
        f = np.asarray(self.det(k)['image_feats'][i], float).reshape(-1)
        return f / np.linalg.norm(f)

    def record(self, k, i):
        g = self.det(k)
        return dict(keyframe=k, detection_id=i, confidence=round(float(g['confidence'][i]), 4), bbox_xyxy=[round(float(x), 1) for x in g['xyxy'][i]])


def mean_unit(F):
    c = np.mean(F, axis=0)
    return c / np.linalg.norm(c)


def largest_cluster(P, radius):
    """Largest connected set of points (neighbours within radius)."""
    from scipy.spatial import cKDTree
    if len(P) < 3:
        return P
    par = list(range(len(P)))

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in cKDTree(P).query_pairs(radius):
        par[root(a)] = root(b)
    lab = np.array([root(i) for i in range(len(P))]); u, cnt = np.unique(lab, return_counts=True)
    return P[lab == u[np.argmax(cnt)]]


def box_points(center_xy, z0, w, d, h):
    """Grid points of an upright box standing on z0 (used where no LiDAR point exists: the box is a placeholder, its size is not measured)."""
    return np.array([[center_xy[0] + a, center_xy[1] + b, z0 + e] for a in np.linspace(-w / 2, w / 2, 7)
                     for b in np.linspace(-d / 2, d / 2, 5) for e in np.linspace(0, h, 4)])
