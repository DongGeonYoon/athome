"""
Compare the real ROS node's scene-graph JSON (rosbag replay) with the offline dataset builder on the same episode,
and evaluate both against the Gazebo ground truth with the dataset's GT metric.

Usage (conceptgraph env): python compare_ros_offline.py ROS_JSON DATASET_SCENE_DIR SOURCE_DIR [--radius 0.3]
"""
import argparse, json, os, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../scene_graph_dataset'))
import build_dataset as B


def as_objs(objects):
    return [SimpleNamespace(label=o['semantic_tag'], bbox_center=np.array(o['bbox_3d']['center']),
                            size=np.array(o['bbox_3d']['size']), yaw=o['bbox_3d']['yaw']) for o in objects]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ros_json'); ap.add_argument('ds_scene'); ap.add_argument('src'); ap.add_argument('--radius', type=float, default=0.3)
    a = ap.parse_args()
    ros = json.load(open(a.ros_json))['objects']
    off = json.load(open(os.path.join(a.ds_scene, 'objects.json')))['objects']
    gtf = os.path.join(a.src, 'gt_objects.json')
    gt = json.load(open(gtf)) if os.path.exists(gtf) else None      # real-robot runs have no ground truth
    C = lambda objs: np.array([o['bbox_3d']['center'] for o in objs]).reshape(-1, 3)
    cr, co = C(ros), C(off)
    D = np.linalg.norm(cr[:, None, :2] - co[None, :, :2], axis=2) if len(ros) and len(off) else np.zeros((len(ros), len(off)))
    pairs, used = [], set()
    for i in np.argsort(D.min(1) if D.size else []):
        j = int(np.argmin(np.where([k in used for k in range(len(off))], np.inf, D[i])))
        if D[i, j] <= a.radius:
            used.add(j); pairs.append((i, j))
    same_tag = sum(ros[i]['semantic_tag'] == off[j]['semantic_tag'] for i, j in pairs)
    cos = [float(np.dot(ros[i]['clip_feature'], off[j]['clip_feature'])) for i, j in pairs]
    dc = [float(np.linalg.norm(cr[i] - co[j])) for i, j in pairs]
    res = dict(ros_objects=len(ros), offline_objects=len(off), matched_within_m=a.radius, matched=len(pairs),
               matched_same_tag=same_tag, centre_dist_median_m=round(float(np.median(dc)), 3) if dc else None,
               clip_cos_median=round(float(np.median(cos)), 3) if cos else None,
               gt_ros=B.gt_evaluation(as_objs(ros), gt) if gt else None, gt_offline=B.gt_evaluation(as_objs(off), gt) if gt else None)
    for k in ('gt_ros', 'gt_offline'):
        if res[k]:
            res[k].pop('missed_classes', None)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(a.ros_json), 'compare_ros_offline.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
