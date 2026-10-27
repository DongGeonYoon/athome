from __future__ import annotations
import gzip
import pickle
import numpy as np

BG_CLASSES = {"wall", "floor", "ceiling"}


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load_gsa(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def iter_filtered_detections(
    gobs,
    mask_conf_threshold=0.25,
    max_bbox_area_ratio=0.5,
    skip_bg=False,
):
    masks = _np(gobs["mask"])
    xyxy = _np(gobs["xyxy"])
    conf = _np(gobs["confidence"])
    class_ids = _np(gobs["class_id"]).astype(int)
    classes = list(gobs["classes"])
    image_feats = gobs["image_feats"]

    for i in range(len(xyxy)):
        cid = int(class_ids[i])
        if cid < 0 or cid >= len(classes):
            continue

        label = str(classes[cid])
        confidence = float(conf[i])
        if confidence < float(mask_conf_threshold):
            continue

        mask = _np(masks[i]).astype(bool)
        h, w = mask.shape[:2]
        box = _np(xyxy[i]).astype(np.float32).reshape(4)
        x1, y1, x2, y2 = [float(v) for v in box]
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        image_area = float(h * w)

        # Official CG-D mapping behavior: large-box filter does not apply to BG classes.
        if label not in BG_CLASSES and image_area > 0.0:
            if bbox_area > float(max_bbox_area_ratio) * image_area:
                continue

        if skip_bg and label in BG_CLASSES:
            continue

        feat = _np(image_feats[i]).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-12:
            feat = feat / norm

        yield {
            "detection_id": i,
            "bbox_xyxy": box,
            "label": label,
            "confidence": confidence,
            "mask": mask,
            "clip_feature": feat,
        }
