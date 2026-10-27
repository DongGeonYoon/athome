"""
Live CG-D on the robot camera (CG-D code unchanged; same GPU-split runner and v2 vocabulary as the dataset).

Takes the newest colour image whenever the previous one is finished (CG-D needs ~4-6 s per frame on an 8 GB GPU),
runs RAM -> GroundingDINO -> SAM -> CLIP exactly like generate_gsa_results.py, applies the same detection filter as
the dataset (confidence >= 0.25, bbox <= 50 % of the image, no wall/floor/ceiling) and publishes
  /cg_d/objects_2d   SemanticObject2DArray (header = image stamp)  -> geometry_fusion_node
  /cg_d/annotated    sensor_msgs/Image (boxes, masks, labels)        -> rqt_image_view
Search targets from /athome/search/targets (athome_interfaces/SearchTargets): every query_names entry of every target
is appended to the DINO class list, so a target detection's label is one of its query_names verbatim (the search server
judges "found" by that label). An empty list returns to the plain vocabulary.

Run with the conceptgraph python after sourcing ROS (see start_perception.sh):
  python cgd_live_node.py [--image-topic /camera/camera/color/image_raw] [--period 0]
"""
import argparse, json, os, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, '..', 'replica_rig', 'vocab')
os.environ.setdefault('GSA_PATH', os.path.expanduser('~/conceptgraph_ws/Grounded-Segment-Anything'))
os.environ.setdefault('GSA_EXTRA_CLASSES_FILE', os.path.join(VOCAB, 'household.txt'))
os.environ.setdefault('GSA_BLOCK_TAGS_FILE', os.path.join(VOCAB, 'blocklist.txt'))
os.environ.setdefault('PYTHONHASHSEED', '0')
os.chdir(os.path.expanduser('~/conceptgraph_ws/concept-graphs/conceptgraph'))   # CG-D resolves some paths relative to it
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
sys.path.insert(0, os.path.join(HERE, '..', 'replica_rig'))
import torch, torchvision
import gsa_fast                                    # patches CG-D: RAM on CPU, fp16 SAM/CLIP, phrase->class fix, vocab v2
gsa = gsa_fast.gsa

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from semantic_geometry_msgs.msg import SemanticObject2D, SemanticObject2DArray
from athome_interfaces.msg import SearchTargets

BG = {'wall', 'floor', 'ceiling'}
REMOVE = ["room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "shade",
          "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern",
          "salon", "doorway", "wall lamp", "wood floor"]


def to_image_msg(img_bgr, header):
    m = Image(); m.header = header; m.height, m.width = img_bgr.shape[:2]
    m.encoding = 'bgr8'; m.step = int(img_bgr.shape[1] * 3); m.data = np.ascontiguousarray(img_bgr).tobytes()
    return m


def mask_msg(mask):
    m = Image(); m.height, m.width = mask.shape; m.encoding = 'mono8'; m.step = int(mask.shape[1])
    m.data = (mask.astype(np.uint8) * 255).tobytes()
    return m


class CGDLive(Node):
    def __init__(self, args):
        super().__init__('cgd_live')
        self.args, self.latest, self.targets, self.n = args, None, [], 0
        dev = 'cuda'
        t0 = time.time()
        self.dino = gsa.Model(model_config_path=gsa.GROUNDING_DINO_CONFIG_PATH,
                              model_checkpoint_path=gsa.GROUNDING_DINO_CHECKPOINT_PATH, device=dev)
        self.sam = gsa.get_sam_predictor('sam', dev)
        self.clip, _, self.clip_pre = gsa.open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k')
        self.clip = self.clip.to(dev); self.clip_tok = gsa.open_clip.get_tokenizer('ViT-H-14')
        self.tagger = gsa.ram(pretrained=gsa.RAM_CHECKPOINT_PATH, image_size=384, vit='swin_l').eval().to(dev)
        self.tag_tf = gsa.TS.Compose([gsa.TS.Resize((384, 384)), gsa.TS.ToTensor(),
                                      gsa.TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.get_logger().info(f'CG-D models loaded in {time.time() - t0:.0f} s')
        compressed = args.image_topic.endswith('/compressed')
        self.create_subscription(CompressedImage if compressed else Image, args.image_topic, self.on_image,
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(SearchTargets, '/athome/search/targets', self.on_targets, latched)
        self.pub = self.create_publisher(SemanticObject2DArray, '/cg_d/objects_2d', 10)
        self.vis = self.create_publisher(Image, '/cg_d/annotated', 2)

    def on_image(self, msg):
        self.latest = msg

    def on_targets(self, msg):
        names = [q.strip() for t in msg.targets for q in t.query_names if q.strip()]
        self.targets = list(dict.fromkeys(names))          # query names in order, duplicates removed
        self.get_logger().info(f'search targets (command {msg.command_id}): '
                               f'{[(t.name, list(t.query_names)) for t in msg.targets]} -> query {self.targets}')

    def decode(self, msg):
        if isinstance(msg, CompressedImage):
            return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding.lower() == 'rgb8' else img.copy()

    @torch.no_grad()
    def step(self):
        msg, self.latest = self.latest, None
        t0 = time.time()
        image = self.decode(msg); rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tags = gsa.inference_ram(self.tag_tf(gsa.Image.fromarray(rgb).resize((384, 384))).unsqueeze(0), self.tagger)
        classes = gsa.process_tag_classes(tags[0].replace(' |', ','), add_classes=['other item'] + sorted(BG),
                                          remove_classes=list(REMOVE))
        classes += [t for t in self.targets if t not in classes]
        det = self.dino.predict_with_classes(image=image, classes=classes, box_threshold=0.2, text_threshold=0.2)
        out = SemanticObject2DArray(); out.header = msg.header
        labels = []
        if len(det.class_id) > 0:
            keep = torchvision.ops.nms(torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence), 0.5).numpy().tolist()
            det.xyxy, det.confidence, det.class_id = det.xyxy[keep], det.confidence[keep], det.class_id[keep]
            valid = det.class_id != -1
            det.xyxy, det.confidence, det.class_id = det.xyxy[valid], det.confidence[valid], det.class_id[valid]
        if len(det.class_id) > 0:
            det.mask = gsa.get_sam_segmentation_from_xyxy(sam_predictor=self.sam, image=rgb, xyxy=det.xyxy)
            _, feats, _ = gsa.compute_clip_features(rgb, det, self.clip, self.clip_pre, self.clip_tok, classes, 'cuda')
            H, W = image.shape[:2]
            for i in range(len(det.xyxy)):
                lab, conf = classes[int(det.class_id[i])], float(det.confidence[i])
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[i]]
                if conf < 0.25 or lab in BG or (x2 - x1) * (y2 - y1) > 0.5 * W * H:     # dataset filter
                    continue
                f = np.asarray(feats[i], np.float32).reshape(-1); f /= max(float(np.linalg.norm(f)), 1e-12)
                o = SemanticObject2D(); o.header = msg.header; o.detection_id = i
                o.bbox_xyxy = [x1, y1, x2, y2]; o.label = lab; o.confidence = conf
                o.mask = mask_msg(det.mask[i]); o.clip_feature = f.tolist()
                out.objects.append(o); labels.append(f'{lab} {conf:.2f}')
            annotated, _ = gsa.vis_result_fast(image, det, classes)
        else:
            annotated = image
        self.pub.publish(out)
        if self.args.save_dir:                     # experiment record: annotated frame + detections per frame
            os.makedirs(os.path.join(self.args.save_dir, 'annotated'), exist_ok=True)
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cv2.imwrite(os.path.join(self.args.save_dir, 'annotated', f'{self.n:05d}_{st:.2f}.jpg'), annotated)
            with open(os.path.join(self.args.save_dir, 'detections.jsonl'), 'a') as f:
                f.write(json.dumps(dict(frame=self.n, stamp=st, seconds=round(time.time() - t0, 2), ram_tags=tags[0],
                                        query=self.targets, n_classes=len(classes),
                                        objects=[dict(label=o.label, confidence=round(float(o.confidence), 3),
                                                      bbox=[round(float(v), 1) for v in o.bbox_xyxy]) for o in out.objects])) + '\n')
        cv2.putText(annotated, f'{len(out.objects)} objects  {time.time() - t0:.1f} s', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        self.vis.publish(to_image_msg(annotated, msg.header))
        self.n += 1
        self.get_logger().info(f'frame {self.n}: {time.time() - t0:.1f} s, {len(out.objects)} objects: {", ".join(labels)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-topic', default='/camera/camera/color/image_raw')
    ap.add_argument('--period', type=float, default=0.0, help='minimum seconds between processed frames (0 = back to back)')
    ap.add_argument('--save-dir', default='', help='write annotated frames and detections.jsonl here')
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = CGDLive(args)
    last = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest is not None and time.time() - last >= args.period:
                last = time.time(); node.step()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
