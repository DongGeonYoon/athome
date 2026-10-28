#!/usr/bin/env python3
"""
Tag check only (GPU server): which label does each object in the camera image get? RAM -> GroundingDINO with the same
vocabulary, class cleanup, NMS and dataset filter (confidence >= 0.25, bbox <= 50 % of the image, no wall/floor/ceiling)
as the live CG-D node, so the labels are the ones the full pipeline would give. SAM and CLIP are not loaded (no masks,
no CLIP features: not usable for fusion / Scene Graph), which also leaves room to run RAM on the GPU.
Request / response over the SSH tunnel with tools/robot_live/tag_check_node.py (cgd_wire framing):
  request  {"id"}, [image png]      response {"id", "objects": [{label, confidence, bbox}], "ram_tags", "seconds"}, [annotated jpeg]
  source /data/athome/activate.sh && python tag_check_server.py [--port 47001]
"""
import argparse, json, os, socket, sys, time
os.environ['GSA_RAM_GPU'] = '1'                    # gsa_fast: RAM on the GPU (only RAM + DINO are loaded here)
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, '..', 'replica_rig', 'vocab')
os.environ.setdefault('GSA_PATH', os.path.expanduser('~/conceptgraph_ws/Grounded-Segment-Anything'))
os.environ.setdefault('GSA_EXTRA_CLASSES_FILE', os.path.join(VOCAB, 'household.txt'))
os.environ.setdefault('GSA_BLOCK_TAGS_FILE', os.path.join(VOCAB, 'blocklist.txt'))
os.environ.setdefault('PYTHONHASHSEED', '0')
os.chdir(os.path.expanduser('~/conceptgraph_ws/concept-graphs/conceptgraph'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'replica_rig'))
import torch, torchvision
import gsa_fast
import cgd_wire as W
gsa = gsa_fast.gsa

BG = {'wall', 'floor', 'ceiling'}
REMOVE = ["room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "shade",
          "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern",
          "salon", "doorway", "wall lamp", "wood floor"]
COLORS = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245), (180, 30, 145), (240, 240, 70),
          (230, 50, 240), (60, 245, 210), (212, 190, 250), (128, 128, 0), (40, 110, 170)]


class TagCheck:
    def __init__(self, dev='cuda'):
        self.dino = gsa.Model(model_config_path=gsa.GROUNDING_DINO_CONFIG_PATH,
                              model_checkpoint_path=gsa.GROUNDING_DINO_CHECKPOINT_PATH, device=dev)
        self.tagger = gsa.ram(pretrained=gsa.RAM_CHECKPOINT_PATH, image_size=384, vit='swin_l').eval().to(dev)
        self.tag_tf = gsa.TS.Compose([gsa.TS.Resize((384, 384)), gsa.TS.ToTensor(),
                                      gsa.TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def run(self, image):
        t0 = time.time()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tags = gsa.inference_ram(self.tag_tf(gsa.Image.fromarray(rgb).resize((384, 384))).unsqueeze(0), self.tagger)
        classes = gsa.process_tag_classes(tags[0].replace(' |', ','), add_classes=['other item'] + sorted(BG),
                                          remove_classes=list(REMOVE))
        det = self.dino.predict_with_classes(image=image, classes=classes, box_threshold=0.2, text_threshold=0.2)
        objects = []
        if len(det.class_id) > 0:
            keep = torchvision.ops.nms(torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence), 0.5).numpy().tolist()
            H, Wd = image.shape[:2]
            for i in keep:
                if det.class_id[i] == -1:
                    continue
                lab, conf = classes[int(det.class_id[i])], float(det.confidence[i])
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[i]]
                if conf < 0.25 or lab in BG or (x2 - x1) * (y2 - y1) > 0.5 * Wd * H:     # same filter as the dataset
                    continue
                objects.append(dict(label=lab, confidence=conf, bbox=[x1, y1, x2, y2], cid=int(det.class_id[i])))
        return objects, tags[0], time.time() - t0


def draw(image, objects, seconds):
    out = image.copy()
    for o in sorted(objects, key=lambda o: o['confidence']):
        c = COLORS[o['cid'] % len(COLORS)]
        x1, y1, x2, y2 = (int(v) for v in o['bbox'])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        txt = f"{o['label']} {o['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), c, -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f'{len(objects)} objects  {seconds:.2f} s (tag check: RAM + DINO)', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=47001); a = ap.parse_args()
    t0 = time.time(); tc = TagCheck()
    print(f'[tag_check] RAM + DINO loaded in {time.time() - t0:.0f} s, listening on 127.0.0.1:{a.port}', flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', a.port)); srv.listen(1)
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print('[tag_check] client connected', flush=True)
        while True:
            try:
                req, (img,) = W.recv(conn)
            except (ConnectionError, ValueError):
                break
            image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            objects, tags, sec = tc.run(image)
            ok, ann = cv2.imencode('.jpg', draw(image, objects, sec), [cv2.IMWRITE_JPEG_QUALITY, 85])
            W.send(conn, dict(id=req.get('id'), ram_tags=tags, seconds=sec,
                              objects=[{k: o[k] for k in ('label', 'confidence', 'bbox')} for o in objects]), [ann.tobytes()])
            print(json.dumps(dict(t=time.strftime('%H:%M:%S'), seconds=round(sec, 2),
                                  labels=[f"{o['label']} {o['confidence']:.2f}" for o in objects])), flush=True)
        conn.close(); print('[tag_check] client disconnected', flush=True)


if __name__ == '__main__':
    main()
