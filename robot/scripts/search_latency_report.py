#!/usr/bin/env python3
"""
Summarise search_latency_monitor.py recordings (search server's view of live CG-D), side by side:
  age               image published -> result available (image stamp mapped to its receive time)
  taken while turning   results whose image was taken with |yaw rate| > 0.1 rad/s (useless for the search server)
  stop -> result    for every stop (|yaw rate| < 0.1 rad/s for >= 2 s after turning): time from the stop until the
                    first result of an image taken after the stop (the search server waits up to 25 s)
  python3 search_latency_report.py NAME=DIR/monitor.json [NAME=...]
"""
import json, sys
import numpy as np

TURN, STILL = 0.1, 2.0


def analyse(p):
    r = json.load(open(p))
    img = np.array(r['images']); res = np.array(r['results']); yaw = np.array(sorted(r['yaw']))
    o = np.argsort(img[:, 1]); ist, iw = img[o, 1], img[o, 0]
    wall_of = lambda s: np.interp(s, ist, iw)                     # stamp -> time the image reached the node (any replay rate)
    age = res[:, 0] - wall_of(res[:, 1])
    t = yaw[:, 0]; y = np.unwrap(yaw[:, 1])
    def rate(s, w=0.2):
        return abs(np.interp(s, t, y) - np.interp(s - w, t, y)) / w
    turning = np.array([rate(s) > TURN for s in res[:, 1]])
    grid = np.arange(t[0] + 0.2, t[-1], 0.1); g = np.array([rate(s) for s in grid]) <= TURN
    stops, k = [], 0
    while k < len(g):                                              # still segments >= STILL s that follow a turn
        if g[k] and k > 0 and not g[k - 1]:
            j = k
            while j < len(g) and g[j]:
                j += 1
            if grid[min(j, len(g) - 1)] - grid[k] >= STILL:
                stops.append(grid[k])
            k = j
        else:
            k += 1
    delays = []
    for s in stops:
        after = res[res[:, 1] >= s]
        if len(after):
            delays.append(after[0, 0] - wall_of(s))
    return dict(results=len(res), age=age, turning=turning, stops=len(stops), delays=np.array(delays))


def q(a, f):
    return f'{f(a):.1f}' if len(a) else '-'


if __name__ == '__main__':
    print(f"{'':6s} {'results':>7s} | {'age median / p90 / max [s]':>27s} | {'taken while turning':>20s} | {'stops':>5s} | {'stop -> result median / max [s]':>32s} | >25 s")
    for a in sys.argv[1:]:
        name, p = a.split('=', 1); m = analyse(p); A, D = m['age'], m['delays']
        print(f"{name:6s} {m['results']:7d} | {q(A, np.median):>9s} / {q(A, lambda x: np.percentile(x, 90)):>5s} / {q(A, np.max):>5s} | "
              f"{int(m['turning'].sum()):>3d} ({100 * m['turning'].mean():4.0f}%)        | {m['stops']:5d} | "
              f"{q(D, np.median):>14s} / {q(D, np.max):>5s}           | {int((D > 25).sum())}")
