"""
CG-D inference shared by the live ROS node (cgd_live_node.py) and the GPU-server service (cgd_server.py).
CG-D code unchanged: same GPU-split runner (gsa_fast) and v2 vocabulary as the dataset. One image in ->
RAM (optional) -> GroundingDINO -> SAM -> CLIP, then the dataset filter (confidence >= 0.25, bbox <= 50 % of the image,
no wall/floor/ceiling). Live path only: pre-mapping runs gsa_fast.py directly and is not affected by this file.
  use_ram=False  DINO classes = household list + search targets, RAM is not run (and not loaded until first needed)
  CLIP           same crops and preprocessing as CG-D compute_clip_features, encoded in batches, kept detections only
  drawing        not done here: cgd_wire.draw_detections(image, objects), on the side that shows the image
No ROS imports here.
"""
import os, sys, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(HERE, '..', 'replica_rig', 'vocab')
os.environ.setdefault('GSA_PATH', os.path.expanduser('~/conceptgraph_ws/Grounded-Segment-Anything'))
os.environ.setdefault('GSA_EXTRA_CLASSES_FILE', os.path.join(VOCAB, 'household.txt'))
os.environ.setdefault('GSA_BLOCK_TAGS_FILE', os.path.join(VOCAB, 'blocklist.txt'))
os.environ.setdefault('PYTHONHASHSEED', '0')
os.chdir(os.path.expanduser('~/conceptgraph_ws/concept-graphs/conceptgraph'))   # CG-D resolves some paths relative to it
sys.path.insert(0, os.path.join(HERE, '..', 'replica_rig'))
import torch, torchvision
import gsa_fast                                    # patches CG-D: RAM on CPU (GSA_RAM_GPU=1: GPU), fp16 SAM/CLIP, phrase->class fix, vocab v2
gsa = gsa_fast.gsa

BG = {'wall', 'floor', 'ceiling'}
REMOVE = ["room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "shade",
          "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern",
          "salon", "doorway", "wall lamp", "wood floor"]
CLIP_PADDING, CLIP_BATCH = 20, 16                  # padding of CG-D compute_clip_features; crops per GPU batch


class CGDEngine:
    def __init__(self, dev='cuda', use_ram=True):
        self.dev = dev
        self.dino = gsa.Model(model_config_path=gsa.GROUNDING_DINO_CONFIG_PATH,
                              model_checkpoint_path=gsa.GROUNDING_DINO_CHECKPOINT_PATH, device=dev)
        self.sam = gsa.get_sam_predictor('sam', dev)
        self.clip, _, self.clip_pre = gsa.open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k')
        self.clip = self.clip.to(dev)
        self.tagger = None
        if use_ram:
            self.load_ram()

    def load_ram(self):
        self.tagger = gsa.ram(pretrained=gsa.RAM_CHECKPOINT_PATH, image_size=384, vit='swin_l').eval().to(self.dev)
        self.tag_tf = gsa.TS.Compose([gsa.TS.Resize((384, 384)), gsa.TS.ToTensor(),
                                      gsa.TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def clip_features(self, rgb, xyxy):
        """L2-normalized float32 image features of the boxes (N x 1024). Crops and preprocessing as in CG-D
        compute_clip_features; encoded CLIP_BATCH at a time, without its per-object text features (not used here)."""
        img = gsa.Image.fromarray(rgb); W, H = img.size; p = CLIP_PADDING
        crops = [self.clip_pre(img.crop((x1 - min(p, x1), y1 - min(p, y1), x2 + min(p, W - x2), y2 + min(p, H - y2))))
                 for x1, y1, x2, y2 in xyxy]
        out = []
        with torch.autocast('cuda', dtype=torch.float16, enabled=os.environ.get('GSA_CLIP_FP32', '0') != '1'):
            for i in range(0, len(crops), CLIP_BATCH):
                out.append(self.clip.encode_image(torch.stack(crops[i:i + CLIP_BATCH]).to(self.dev)))
        f = torch.cat(out).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def run(self, image, targets=(), use_ram=True):
        """image: BGR uint8. targets: search query names appended to the DINO classes (label = name verbatim).
        use_ram=False: no RAM tags, DINO classes = household list + targets.
        Returns dict(objects=[dict(detection_id, label, confidence, bbox, mask (H x W bool), clip (1024 float32, L2))],
        ram_tags, n_classes, seconds, timing={ram, dino, sam, clip} [s])."""
        t0 = time.time(); timing = dict(ram=0.0, dino=0.0, sam=0.0, clip=0.0)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tag_text = ''
        if use_ram:
            if self.tagger is None:
                self.load_ram()
            tag_text = gsa.inference_ram(self.tag_tf(gsa.Image.fromarray(rgb).resize((384, 384))).unsqueeze(0), self.tagger)[0]
        timing['ram'] = time.time() - t0
        classes = gsa.process_tag_classes(tag_text.replace(' |', ','), add_classes=['other item'] + sorted(BG),
                                          remove_classes=list(REMOVE))     # adds the household list (gsa_fast)
        classes += [t for t in targets if t not in classes]
        t = time.time()
        det = self.dino.predict_with_classes(image=image, classes=classes, box_threshold=0.2, text_threshold=0.2)
        objects = []
        if len(det.class_id) > 0:
            keep = torchvision.ops.nms(torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence), 0.5).numpy().tolist()
            det.xyxy, det.confidence, det.class_id = det.xyxy[keep], det.confidence[keep], det.class_id[keep]
            valid = det.class_id != -1
            det.xyxy, det.confidence, det.class_id = det.xyxy[valid], det.confidence[valid], det.class_id[valid]
        timing['dino'] = time.time() - t
        if len(det.class_id) > 0:
            t = time.time()
            det.mask = gsa.get_sam_segmentation_from_xyxy(sam_predictor=self.sam, image=rgb, xyxy=det.xyxy)
            timing['sam'] = time.time() - t
            H, W = image.shape[:2]
            kept = []
            for i in range(len(det.xyxy)):
                lab, conf = classes[int(det.class_id[i])], float(det.confidence[i])
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[i]]
                if conf < 0.25 or lab in BG or (x2 - x1) * (y2 - y1) > 0.5 * W * H:     # dataset filter
                    continue
                kept.append(i)
                objects.append(dict(detection_id=i, label=lab, confidence=conf, bbox=[x1, y1, x2, y2],
                                    mask=np.asarray(det.mask[i], bool)))
            if kept:
                t = time.time()
                for o, f in zip(objects, self.clip_features(rgb, det.xyxy[kept])):
                    o['clip'] = f
                timing['clip'] = time.time() - t
        return dict(objects=objects, ram_tags=tag_text, n_classes=len(classes), seconds=time.time() - t0, timing=timing)
