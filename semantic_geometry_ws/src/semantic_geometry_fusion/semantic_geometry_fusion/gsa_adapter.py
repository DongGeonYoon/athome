from __future__ import annotations

import gzip
import pickle
from pathlib import Path


def load_gsa_result(path):
    """Load one CG-D generate_gsa_results.py frame output (.pkl.gz)."""
    path = Path(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def iter_gsa_objects(gobs):
    """
    Yield the fields needed by SemanticObject2D from an official CG-D output.

    Official generate_gsa_results.py stores, among others:
      mask, class_id, classes, confidence, image_feats, text_feats.

    This adapter intentionally uses image_feats as the proposal's CLIP semantic feature.
    """
    classes = gobs["classes"]
    for i in range(len(gobs["mask"])):
        class_id = int(gobs["class_id"][i])
        yield {
            "detection_id": i,
            "label": classes[class_id],
            "confidence": float(gobs["confidence"][i]),
            "mask": gobs["mask"][i],
            "clip_feature": gobs["image_feats"][i],
        }
