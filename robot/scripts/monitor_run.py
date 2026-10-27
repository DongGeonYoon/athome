#!/usr/bin/env python3
"""
Live monitor + report for one run (system python, ROS sourced). Answers three questions while the robot scans:
  1. Same-object association: are repeated views of one object merged into one map object at a consistent place?
     - each per-frame observation (/athome/perception/objects) is matched to the map object whose bbox contains it
     - per object: views, label agreement, spread of the observed centres (localisation + extrinsic consistency)
     - suspected splits: two map objects with compatible tags whose boxes overlap / centres < 0.5 m
     - suspected over-merges: small-object tag on a box > 1.5 m
  2. Information accumulation: objects over time, observations per object, heading coverage of the scan (TF map->base_link)
  3. Scene Graph readiness: objects.json schema, height above the floor (floor_z), objects above floor + 1.5 m, objects
     on free / unknown cells of /map, clip_feature norm
Writes <run>/monitor/timeline.jsonl every 10 s and, on Ctrl+C (or SIGINT from stop_experiment.sh), report.md,
summary.json, bev.png.
  python3 monitor_run.py RUN_DIR [--floor-z -0.41]
"""
import argparse, json, math, os, signal, time
from collections import Counter, defaultdict
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from semantic_geometry_msgs.msg import ObjectObservation3DArray

SMALL = {'cup', 'mug', 'bottle', 'pen', 'remote control', 'cell phone', 'book', 'bowl', 'plate', 'keys', 'toy', 'box',
         'laptop', 'candle', 'vase', 'clock', 'basket', 'shoe', 'apple', 'kettle'}
FAMILIES = [{'chair', 'computer chair', 'swivel chair', 'office chair', 'armchair', 'stool', 'seat'},
            {'desk', 'computer desk', 'table', 'office desk', 'dining table', 'coffee table', 'workbench'},
            {'cabinet', 'drawer', 'dresser', 'shelf', 'bookshelf', 'file cabinet', 'nightstand', 'locker'},
            {'trash can', 'garbage', 'bin', 'wastebasket'}, {'monitor', 'computer monitor', 'screen', 'television', 'tv'},
            {'sofa', 'couch'}, {'cup', 'mug', 'glass'}, {'box', 'carton', 'package'}]


def compatible(a, b):
    return a == b or any(a in f and b in f for f in FAMILIES)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Monitor(Node):
    def __init__(self, args):
        super().__init__('run_monitor')
        self.args, self.out = args, os.path.join(args.run, 'monitor')
        os.makedirs(self.out, exist_ok=True)
        self.frames, self.map_objs, self.grid, self.poses, self.t0 = [], [], None, [], time.time()
        self.create_subscription(ObjectObservation3DArray, '/athome/perception/objects', self.on_frame, 50)
        self.create_subscription(ObjectObservation3DArray, '/semantic_geometry/static_map', self.on_map, 5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_grid, latched)
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.create_timer(0.5, self.on_pose)
        self.create_timer(10.0, self.snapshot)

    def on_frame(self, msg):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.frames.append(dict(t=st, objs=[dict(label=o.label, conf=float(o.confidence),
                                                 c=[o.bbox_center.x, o.bbox_center.y, o.bbox_center.z],
                                                 size=[o.size.x, o.size.y, o.size.z]) for o in msg.objects]))

    def on_map(self, msg):
        self.map_objs = [dict(id=o.object_id, label=o.label, n=int(o.num_observations), conf=float(o.confidence),
                              c=np.array([o.bbox_center.x, o.bbox_center.y, o.bbox_center.z]),
                              size=np.array([o.size.x, o.size.y, o.size.z]), yaw=float(o.yaw)) for o in msg.objects]

    def on_grid(self, msg):
        self.grid = msg

    def on_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time())
        except Exception:
            return
        p, q = t.transform.translation, t.transform.rotation
        self.poses.append((time.time(), p.x, p.y, p.z, yaw_of(q)))

    # ---------------------------------------------------------------- analysis
    def inside(self, o, c, margin=0.10):
        R = np.array([[math.cos(o['yaw']), math.sin(o['yaw'])], [-math.sin(o['yaw']), math.cos(o['yaw'])]])
        d = R @ (np.asarray(c[:2]) - o['c'][:2])
        return bool(np.all(np.abs(d) <= o['size'][:2] / 2 + margin) and abs(c[2] - o['c'][2]) <= o['size'][2] / 2 + margin)

    def analyse(self):
        objs = self.map_objs
        per = defaultdict(lambda: dict(labels=[], centres=[]))
        unmatched = 0
        for f in self.frames:
            for ob in f['objs']:
                cand = [o for o in objs if self.inside(o, ob['c'])]
                if not cand:
                    unmatched += 1; continue
                o = min(cand, key=lambda o: np.linalg.norm(o['c'] - np.array(ob['c'])))
                per[o['id']]['labels'].append(ob['label']); per[o['id']]['centres'].append(ob['c'])
        rows = []
        for o in objs:
            p = per.get(o['id'], dict(labels=[], centres=[]))
            cs = np.array(p['centres']) if p['centres'] else np.zeros((0, 3))
            spread = float(np.sqrt(((cs[:, :2] - cs[:, :2].mean(0)) ** 2).sum(1)).mean()) if len(cs) >= 2 else None
            agree = (Counter(p['labels']).most_common(1)[0][1] / len(p['labels'])) if p['labels'] else None
            rows.append(dict(id=o['id'], label=o['label'], n=o['n'], views=len(cs), label_agreement=agree,
                             centre_spread_m=spread, size=o['size'].round(2).tolist(), centre=o['c'].round(2).tolist()))
        splits = []
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                if compatible(a['label'], b['label']) and np.linalg.norm(a['c'][:2] - b['c'][:2]) < 0.5:
                    splits.append((a['id'], b['id'], a['label'], b['label'], round(float(np.linalg.norm(a['c'][:2] - b['c'][:2])), 2)))
        overs = [(o['id'], o['label'], o['size'].round(2).tolist()) for o in objs if o['label'] in SMALL and max(o['size'][:2]) > 1.5]
        # scan coverage
        yaws = np.degrees([p[4] for p in self.poses]) % 360 if self.poses else np.zeros(0)
        cov = len(set((yaws // 10).astype(int))) * 10 if len(yaws) else 0
        moved = float(np.linalg.norm(np.array(self.poses[-1][1:3]) - np.array(self.poses[0][1:3]))) if len(self.poses) > 1 else 0.0
        # scene-graph checks on the exported file
        sg = self.check_json()
        return dict(elapsed_s=round(time.time() - self.t0, 1), frames=len(self.frames),
                    frame_observations=sum(len(f['objs']) for f in self.frames), unmatched_observations=unmatched,
                    map_objects=len(objs), multi_view_objects=sum(o['n'] >= 2 for o in objs),
                    observations_per_object_median=float(np.median([o['n'] for o in objs])) if objs else 0,
                    centre_spread_median_m=float(np.median([r['centre_spread_m'] for r in rows if r['centre_spread_m'] is not None]))
                    if any(r['centre_spread_m'] is not None for r in rows) else None,
                    label_agreement_median=float(np.median([r['label_agreement'] for r in rows if r['label_agreement']]))
                    if any(r['label_agreement'] for r in rows) else None,
                    suspected_splits=splits, suspected_overmerges=overs, heading_coverage_deg=cov, robot_moved_m=round(moved, 2),
                    tags=Counter(o['label'] for o in objs).most_common(25), objects=rows, scene_graph=sg)

    def check_json(self):
        path = os.path.join(self.args.run, 'objects.json')
        if not os.path.exists(path):
            return dict(exported=0, note='objects.json not written yet (every 10 s)')
        sg = json.load(open(path))
        keys = {'object_id', 'semantic_tag', 'confidence', 'centroid', 'bbox_3d', 'clip_feature', 'num_observations'}
        objs = sg.get('objects', [])
        bad = [o.get('object_id') for o in objs if set(o) != keys or len(o['clip_feature']) != 1024]
        fz = self.args.floor_z
        heights = [(o['object_id'], o['semantic_tag'], round(o['bbox_3d']['center'][2] - o['bbox_3d']['size'][2] / 2 - fz, 2),
                    round(o['bbox_3d']['center'][2] + o['bbox_3d']['size'][2] / 2 - fz, 2)) for o in objs]
        floating = [h for h in heights if h[2] > 1.5]          # bottom more than 1.5 m above the floor
        below = [h for h in heights if h[3] < 0.0]              # top below the floor
        cells = Counter()
        if self.grid is not None:
            g = self.grid; res = g.info.resolution; ox, oy = g.info.origin.position.x, g.info.origin.position.y
            data = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
            for o in objs:
                i = int((o['centroid']['y'] - oy) / res); j = int((o['centroid']['x'] - ox) / res)
                if not (0 <= i < g.info.height and 0 <= j < g.info.width):
                    cells['outside map'] += 1; continue
                win = data[max(0, i - 4):i + 5, max(0, j - 4):j + 5]          # +-0.2 m
                cells['near occupied' if (win >= 50).any() else 'free' if (win == 0).all() else 'free/unknown'] += 1
        norms = [float(np.linalg.norm(o['clip_feature'])) for o in objs]
        return dict(exported=len(objs), frame_id=sg.get('frame_id'), schema_errors=bad,
                    clip_norm_range=[round(min(norms), 4), round(max(norms), 4)] if norms else None,
                    floor_z=fz, height_above_floor_bottom_range=[min(h[2] for h in heights), max(h[2] for h in heights)] if heights else None,
                    above_1_5m=floating, below_floor=below, map_cells=dict(cells))

    def snapshot(self):
        s = self.analyse()
        line = {k: s[k] for k in ('elapsed_s', 'frames', 'map_objects', 'multi_view_objects', 'centre_spread_median_m',
                                  'label_agreement_median', 'heading_coverage_deg')}
        line['splits'] = len(s['suspected_splits']); line['exported'] = s['scene_graph'].get('exported')
        with open(os.path.join(self.out, 'timeline.jsonl'), 'a') as f:
            f.write(json.dumps(line) + '\n')
        self.get_logger().info(json.dumps(line))

    def report(self):
        s = self.analyse()
        json.dump(s, open(os.path.join(self.out, 'summary.json'), 'w'), indent=1, default=str)
        self.bev(s)
        sg = s['scene_graph']
        L = [f"# 실험 모니터 보고서: {os.path.basename(self.args.run)}", "",
             f"- 시간 {s['elapsed_s']} s, 처리 frame {s['frames']}, frame별 관측 {s['frame_observations']}",
             f"- 로봇 이동 {s['robot_moved_m']} m, 회전으로 본 방향 범위 {s['heading_coverage_deg']}° / 360°", "",
             "## 1. 같은 물체 정합", "",
             f"- map object {s['map_objects']}개, 2회 이상 관측 {s['multi_view_objects']}개, object당 관측 중앙값 {s['observations_per_object_median']}",
             f"- 같은 object로 모인 관측들의 중심 흩어짐(중앙값): **{s['centre_spread_median_m']} m** (정합 일관성: 작을수록 좋음)",
             f"- label 일치율(중앙값): {s['label_agreement_median']}",
             f"- 쪼개짐 의심(같은 계열 tag, 0.5 m 이내 두 object): {len(s['suspected_splits'])}개 {s['suspected_splits'][:10]}",
             f"- 과병합 의심(작은 물체 tag인데 1.5 m 초과): {s['suspected_overmerges']}",
             f"- map object에 들어가지 않은 frame 관측: {s['unmatched_observations']}", "",
             "## 2. 정보 누적", "", f"- tag 분포: {s['tags']}", "- 시간별 추이: `timeline.jsonl`", "",
             "## 3. Scene Graph 구축 가능성", "",
             f"- `objects.json` object {sg.get('exported')}개, frame_id `{sg.get('frame_id')}`, 형식 오류 {sg.get('schema_errors')}",
             f"- CLIP 노름 범위 {sg.get('clip_norm_range')}",
             f"- 바닥(z {sg.get('floor_z')}) 기준 object 아랫면 높이 범위 {sg.get('height_above_floor_bottom_range')} m",
             f"- 바닥보다 1.5 m 넘게 떠 있는 object(학습 규칙상 제외 대상): {sg.get('above_1_5m')}",
             f"- 바닥 아래 object(좌표 이상): {sg.get('below_floor')}",
             f"- /map에서 object 위치: {sg.get('map_cells')}", "",
             "## object 목록", "", "| id | tag | 관측 | 본 frame | label 일치 | 중심 흩어짐 m | 크기 m |", "|---|---|---|---|---|---|---|"]
        for r in sorted(s['objects'], key=lambda r: -r['n']):
            L.append(f"| {r['id']} | {r['label']} | {r['n']} | {r['views']} | "
                     f"{'' if r['label_agreement'] is None else round(r['label_agreement'], 2)} | "
                     f"{'' if r['centre_spread_m'] is None else round(r['centre_spread_m'], 3)} | {r['size']} |")
        open(os.path.join(self.out, 'report.md'), 'w').write('\n'.join(L) + '\n')
        self.get_logger().info(f'report -> {self.out}/report.md')

    def bev(self, s):
        if self.grid is None:
            return
        g = self.grid; res = g.info.resolution; ox, oy = g.info.origin.position.x, g.info.origin.position.y
        data = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
        img = np.full(data.shape + (3,), 160, np.uint8); img[data == 0] = 255; img[data >= 50] = 40
        img = cv2.resize(img[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        H = img.shape[0]
        px = lambda x, y: (int((x - ox) / res * 2), int(H - (y - oy) / res * 2))
        for o in self.map_objs:
            c, sz, yw = o['c'], o['size'], o['yaw']
            pts = np.array([[sz[0] / 2 * a, sz[1] / 2 * b] for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
            R = np.array([[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]])
            poly = np.array([px(*(R @ p + c[:2])) for p in pts], np.int32)
            col = (0, 160, 0) if o['n'] >= 2 else (0, 0, 220)
            cv2.polylines(img, [poly], True, col, 2)
            cv2.putText(img, f"{o['id']}:{o['label']}", px(*c[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        for p in self.poses[::4]:
            cv2.circle(img, px(p[1], p[2]), 2, (220, 0, 0), -1)
        cv2.imwrite(os.path.join(self.out, 'bev.png'), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--floor-z', type=float, default=-0.41)
    args = ap.parse_args()
    rclpy.init(); node = Monitor(args)
    stop = {'v': False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(v=True)); signal.signal(signal.SIGTERM, lambda *a: stop.update(v=True))
    while not stop['v']:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.report()


if __name__ == '__main__':
    main()
