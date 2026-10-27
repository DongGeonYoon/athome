import argparse
import numpy as np
from .gsa_io import load_gsa, iter_filtered_detections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkl")
    p.add_argument("--mask-conf-threshold", type=float, default=0.25)
    p.add_argument("--max-bbox-area-ratio", type=float, default=0.5)
    args = p.parse_args()

    gobs = load_gsa(args.pkl)
    print("keys:", sorted(gobs.keys()))
    print("raw detections:", len(gobs["xyxy"]))

    dets = list(iter_filtered_detections(
        gobs,
        mask_conf_threshold=args.mask_conf_threshold,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        skip_bg=False,
    ))
    print("filtered detections:", len(dets))
    if not dets:
        return

    dims = sorted(set(len(d["clip_feature"]) for d in dets))
    norms = [float(np.linalg.norm(d["clip_feature"])) for d in dets]
    print("clip dims:", dims)
    print("norm range:", min(norms), max(norms))
    for d in dets[:10]:
        print(
            f"id={d['detection_id']:3d} label={d['label']:<25} "
            f"conf={d['confidence']:.3f} dim={len(d['clip_feature'])} "
            f"norm={np.linalg.norm(d['clip_feature']):.6f}"
        )


if __name__ == "__main__":
    main()
