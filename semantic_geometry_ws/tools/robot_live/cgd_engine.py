"""
CG-D inference shared by the live ROS node (cgd_live_node.py) and the GPU-server service (cgd_server.py).
CG-D code unchanged: same GPU-split runner (gsa_fast) and v2 vocabulary as the dataset. One image in ->
RAM -> GroundingDINO -> SAM -> CLIP exactly like generate_gsa_results.py, then the dataset filter
(confidence >= 0.25, bbox <= 50 % of the image, no wall/floor/ceiling). No ROS imports here.
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
import gsa_fast                                    # patches CG-D: RAM on CPU, fp16 SAM/CLIP, phrase->class fix, vocab v2
gsa = gsa_fast.gsa

BG = {'wall', 'floor', 'ceiling'}
REMOVE = ["room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "shade",
          "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern",
          "salon", "doorway", "wall lamp", "wood floor"]


class CGDEngine:
    def __init__(self, dev='cuda'):
        self.dev = dev
        self.dino = gsa.Model(model_config_path=gsa.GROUNDING_DINO_CONFIG_PATH,
                              model_checkpoint_path=gsa.GROUNDING_DINO_CHECKPOINT_PATH, device=dev)
        self.sam = gsa.get_sam_predictor('sam', dev)
        self.clip, _, self.clip_pre = gsa.open_clip.create_model_and_transforms('ViT-H-14', 'laion2b_s32b_b79k')
        self.clip = self.clip.to(dev); self.clip_tok = gsa.open_clip.get_tokenizer('ViT-H-14')
        self.tagger = gsa.ram(pretrained=gsa.RAM_CHECKPOINT_PATH, image_size=384, vit='swin_l').eval().to(dev)
        self.tag_tf = gsa.TS.Compose([gsa.TS.Resize((384, 384)), gsa.TS.ToTensor(),
                                      gsa.TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def run(self, image, targets=()):
        """image: BGR uint8. targets: search query names appended to the DINO classes (label = name verbatim).
        Returns dict(objects=[dict(detection_id, label, confidence, bbox, mask (H x W bool), clip (1024 float32, L2))],
        annotated (BGR), ram_tags, n_classes, seconds)."""
        t0 = time.time()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tags = gsa.inference_ram(self.tag_tf(gsa.Image.fromarray(rgb).resize((384, 384))).unsqueeze(0), self.tagger)
        classes = gsa.process_tag_classes(tags[0].replace(' |', ','), add_classes=['other item'] + sorted(BG),
                                          remove_classes=list(REMOVE))
        classes += [t for t in targets if t not in classes]
        det = self.dino.predict_with_classes(image=image, classes=classes, box_threshold=0.2, text_threshold=0.2)
        objects = []
        if len(det.class_id) > 0:
            keep = torchvision.ops.nms(torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence), 0.5).numpy().tolist()
            det.xyxy, det.confidence, det.class_id = det.xyxy[keep], det.confidence[keep], det.class_id[keep]
            valid = det.class_id != -1
            det.xyxy, det.confidence, det.class_id = det.xyxy[valid], det.confidence[valid], det.class_id[valid]
        if len(det.class_id) > 0:
            det.mask = gsa.get_sam_segmentation_from_xyxy(sam_predictor=self.sam, image=rgb, xyxy=det.xyxy)
            _, feats, _ = gsa.compute_clip_features(rgb, det, self.clip, self.clip_pre, self.clip_tok, classes, self.dev)
            H, W = image.shape[:2]
            for i in range(len(det.xyxy)):
                lab, conf = classes[int(det.class_id[i])], float(det.confidence[i])
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[i]]
                if conf < 0.25 or lab in BG or (x2 - x1) * (y2 - y1) > 0.5 * W * H:     # dataset filter
                    continue
                f = np.asarray(feats[i], np.float32).reshape(-1); f /= max(float(np.linalg.norm(f)), 1e-12)
                objects.append(dict(detection_id=i, label=lab, confidence=conf, bbox=[x1, y1, x2, y2],
                                    mask=np.asarray(det.mask[i], bool), clip=f))
            annotated, _ = gsa.vis_result_fast(image, det, classes)
        else:
            annotated = image.copy()
        return dict(objects=objects, annotated=annotated, ram_tags=tags[0], n_classes=len(classes),
                    seconds=time.time() - t0)
