"""
Wire format between cgd_remote_node.py (robot PC, ROS) and cgd_server.py (GPU server, no ROS), over an SSH tunnel.
One message = u32 header length + JSON header + binary parts; the header lists the part sizes ("parts").
  request : {"id", "targets": [...], "encoding": "jpeg" | "png"}, parts [image bytes]
  response: {"id", "objects": [{detection_id, label, confidence, bbox}], "ram_tags", "n_classes", "seconds",
             "shape": [H, W]}, parts [masks: zlib(packbits(N x H x W bool)), clip: float32 N x 1024, annotated jpeg]
No pickle: only JSON and raw arrays cross the connection.
"""
import json, socket, struct, zlib
import numpy as np


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError('connection closed')
        buf += chunk
    return bytes(buf)


def send(sock, header, parts=()):
    header = dict(header, parts=[len(p) for p in parts])
    h = json.dumps(header).encode()
    sock.sendall(struct.pack('!I', len(h)) + h + b''.join(parts))


def recv(sock):
    (n,) = struct.unpack('!I', _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    return header, [_recv_exact(sock, k) for k in header['parts']]


def pack_masks(masks):
    return zlib.compress(np.packbits(np.asarray(masks, bool).ravel()).tobytes(), 1)


def unpack_masks(blob, n, h, w):
    if n == 0:
        return np.zeros((0, h, w), bool)
    bits = np.unpackbits(np.frombuffer(zlib.decompress(blob), np.uint8))[: n * h * w]
    return bits.reshape(n, h, w).astype(bool)


def connect(addr, timeout=None):
    host, port = addr.rsplit(':', 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


class RemoteEngine:
    """Same interface as cgd_engine.CGDEngine.run, computed by cgd_server.py on the GPU server. The image goes as PNG
    (lossless: the server sees exactly the pixels a local CG-D would)."""

    def __init__(self, addr):
        import cv2
        self.addr, self.cv2, self.sock, self.k = addr, cv2, None, 0

    def run(self, image, targets=()):
        cv2 = self.cv2
        ok, png = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.k += 1
        for attempt in (0, 1):                     # one reconnect: the server or the SSH tunnel may have restarted
            try:
                if self.sock is None:
                    self.sock = connect(self.addr, timeout=60)
                send(self.sock, dict(id=self.k, targets=list(targets), encoding='png'), [png.tobytes()])
                h, (masks, clip, ann) = recv(self.sock)
                break
            except (OSError, ConnectionError):
                if self.sock is not None:
                    self.sock.close()
                self.sock = None
                if attempt:
                    raise
        n, (H, Wd) = len(h['objects']), h['shape']
        M = unpack_masks(masks, n, H, Wd); F = np.frombuffer(clip, np.float32).reshape(n, -1) if n else np.zeros((0, 1024), np.float32)
        objs = [dict(o, mask=M[i], clip=F[i].copy()) for i, o in enumerate(h['objects'])]
        annotated = cv2.imdecode(np.frombuffer(ann, np.uint8), cv2.IMREAD_COLOR)
        return dict(objects=objs, annotated=annotated, ram_tags=h['ram_tags'], n_classes=h['n_classes'], seconds=h['seconds'])
