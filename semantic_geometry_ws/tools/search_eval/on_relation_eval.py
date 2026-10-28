#!/usr/bin/env python3
"""
"Object on furniture" check with the search server's rule (same as its training data, HM3D boxes):
  |object box bottom - furniture box top| < 0.10 m  and  top-view overlap / object footprint > 0.5
Footprints: the yawed bbox rectangles (default), or with --aabb the axis-aligned box enclosing each yawed box, which is
how the search server builds its graph (its training data, HM3D, has axis-aligned object boxes; require4 5).
  python on_relation_eval.py [--aabb] OBJECTS_JSON [OBJECTS_JSON ...]   -> per pair table + count of connected objects
"""
import json, sys
import numpy as np

SUPPORT = ('desk', 'table', 'counter', 'nightstand', 'dresser', 'cabinet', 'shelf', 'stand')
MAX_DZ, MIN_OVERLAP = 0.10, 0.5


AABB = False


def rect(o):
    b = o['bbox_3d']; c, s, y = np.array(b['center'][:2]), np.array(b['size'][:2]) / 2, b.get('yaw', 0.0)
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    pts = [c + R @ np.array(p) for p in ((-s[0], -s[1]), (s[0], -s[1]), (s[0], s[1]), (-s[0], s[1]))]
    if AABB:                                       # axis-aligned box enclosing the yawed one
        lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
        pts = [np.array(p) for p in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    return pts


def area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def clip(subject, clipper):
    """Sutherland-Hodgman: subject polygon clipped by a convex counter-clockwise polygon."""
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(_cross(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(_cross(p, q, a, b))
        if not out:
            break
    return out


def _cross(p, q, a, b):
    d1, d2 = q - p, b - a
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    return p + t * d1


def zrange(o):
    b = o['bbox_3d']; return b['center'][2] - b['size'][2] / 2, b['center'][2] + b['size'][2] / 2


def pairs(objs):
    sup = [o for o in objs if set(o['semantic_tag'].split()) & set(SUPPORT)]     # word match: 'desktop computer' is no desk
    rows = []
    for o in objs:
        if o in sup:
            continue
        ro = rect(o); ao = area(ro)
        best = None
        for s in sup:
            ov = area(clip(ro, rect(s))) / ao if ao > 0 else 0.0
            if ov <= 0:
                continue
            dz = zrange(o)[0] - zrange(s)[1]
            if zrange(o)[1] < zrange(s)[1] - 0.05:      # object entirely below the furniture top: not "on" it
                continue
            cand = (abs(dz) < MAX_DZ and ov > MIN_OVERLAP, ov, dz, s)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        if best:
            rows.append((o, best[3], best[2], best[1], best[0]))
    return rows


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--aabb' in args:
        AABB = True; args.remove('--aabb')
    for p in args:
        objs = json.load(open(p))['objects']
        rows = pairs(objs)
        print(f"== {p} ({'axis-aligned enclosing boxes' if AABB else 'yawed boxes'})")
        for o, s, dz, ov, ok in sorted(rows, key=lambda r: (r[1]['object_id'], r[0]['object_id'])):
            print(f"  {o['semantic_tag']}_{o['object_id']:<4} on {s['semantic_tag']}_{s['object_id']:<4} dz {dz:+.3f} m  overlap {ov:.2f}  "
                  f"{'connected' if ok else '-'}")
        print(f'  connected {sum(r[4] for r in rows)} / candidates {len(rows)}')
