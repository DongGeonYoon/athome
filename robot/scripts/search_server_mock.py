#!/usr/bin/env python3
"""
Stand-in for the search server's /athome/search/targets side (require3 section 3), for bag replays:
publishes the command's targets at start, drops a target once /athome/perception/objects carries one of its
query_names with confidence >= 0.4 (the server's "found" rule), publishes an empty list when everything is found
or on Ctrl-C (command end). Same QoS as the server: reliable, transient local, depth 1. Events -> stdout (JSON lines).
  python3 search_server_mock.py --command-id 1 --target "remote control:remote control" --target "cups:cup"
  (--target NAME:QUERY1,QUERY2,...)
"""
import argparse, json, signal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from athome_interfaces.msg import SearchTarget, SearchTargets, ObservedObjectArray

CONF = 0.4


class Mock(Node):
    def __init__(self, a):
        super().__init__('search_server_mock')
        self.cid, self.left = a.command_id, []
        for t in a.target:
            name, q = t.split(':', 1)
            self.left.append(SearchTarget(name=name, query_names=[x.strip() for x in q.split(',') if x.strip()]))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(SearchTargets, '/athome/search/targets', latched)
        self.create_subscription(ObservedObjectArray, '/athome/perception/objects', self.on_obs, 10)
        self.started = False
        self.create_timer(1.0, self.start_once)

    def log(self, **kw):
        print(json.dumps(dict(sim_time=self.get_clock().now().nanoseconds * 1e-9, **kw)), flush=True)

    def publish(self, event):
        m = SearchTargets(); m.header.stamp = self.get_clock().now().to_msg(); m.command_id = self.cid; m.targets = self.left
        self.pub.publish(m)
        self.log(event=event, targets=[(t.name, list(t.query_names)) for t in self.left])

    def start_once(self):
        if not self.started and self.get_clock().now().nanoseconds > 0:     # wait for /clock of the replay
            self.started = True; self.publish('command start')

    def on_obs(self, msg):
        if not self.started:
            return
        for t in list(self.left):
            hits = [o for o in msg.objects if o.label in t.query_names and o.confidence >= CONF]
            if hits:
                o = hits[0]
                self.left = [x for x in self.left if x is not t]
                self.log(event='found', target=t.name, label=o.label, confidence=round(o.confidence, 3),
                         centroid=[round(o.centroid.x, 2), round(o.centroid.y, 2), round(o.centroid.z, 2)],
                         image_stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                self.publish('target list changed')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--command-id', type=int, default=1)
    ap.add_argument('--target', action='append', required=True); a = ap.parse_args()
    rclpy.init(); n = Mock(a)
    stop = {'v': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True)); signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    while rclpy.ok() and not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.left = []; n.publish('command end')
    for _ in range(20):
        rclpy.spin_once(n, timeout_sec=0.05)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
