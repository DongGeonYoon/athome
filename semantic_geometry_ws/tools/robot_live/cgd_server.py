#!/usr/bin/env python3
"""
CG-D as a service on the GPU server (no ROS there): the robot PC's cgd_remote_node.py sends one image (+ search target
query names) through an SSH tunnel, this answers with the same detections cgd_live_node.py would publish
(label, confidence, bbox, SAM mask, CLIP feature) and the annotated image. Listens on 127.0.0.1 only; the SSH tunnel
(ssh -L PORT:127.0.0.1:PORT) is the only way in. One client at a time, frames processed in order.
  source /data/athome/activate.sh && python cgd_server.py [--port 47000]
"""
import argparse, json, os, socket, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgd_wire as W
from cgd_engine import CGDEngine


def serve(conn, engine, n_done):
    while True:
        try:
            req, (img_bytes,) = W.recv(conn)
        except (ConnectionError, ValueError):
            return n_done
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        r = engine.run(image, req.get('targets', []))
        objs = r['objects']; H, Wd = image.shape[:2]
        ok, ann = cv2.imencode('.jpg', r['annotated'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        masks = W.pack_masks(np.stack([o['mask'] for o in objs]) if objs else np.zeros((0, H, Wd), bool))
        clip = np.stack([o['clip'] for o in objs]).astype(np.float32).tobytes() if objs else b''
        W.send(conn, dict(id=req.get('id'), shape=[H, Wd], ram_tags=r['ram_tags'], n_classes=r['n_classes'],
                          seconds=r['seconds'],
                          objects=[{k: o[k] for k in ('detection_id', 'label', 'confidence', 'bbox')} for o in objs]),
               [masks, clip, ann.tobytes()])
        n_done += 1
        print(json.dumps(dict(t=time.strftime('%H:%M:%S'), frame=n_done, seconds=round(r['seconds'], 2), objects=len(objs),
                              query=req.get('targets', []))), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=47000); a = ap.parse_args()
    t0 = time.time(); engine = CGDEngine('cuda')
    print(f'[cgd_server] models loaded in {time.time() - t0:.0f} s, listening on 127.0.0.1:{a.port}', flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', a.port)); srv.listen(1)
    n = 0
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print('[cgd_server] client connected', flush=True)
        n = serve(conn, engine, n); conn.close()
        print('[cgd_server] client disconnected', flush=True)


if __name__ == '__main__':
    main()
