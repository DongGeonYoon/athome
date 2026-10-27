"""
Run CG-D generate_gsa_results.py unchanged on an 8 GB GPU by splitting devices:
  RAM tagging model (5.6 GB checkpoint)  -> CPU   (runs once per frame on a 384x384 image)
  CLIP ViT-H-14                           -> GPU in fp16 (features returned as float32)
  GroundingDINO, SAM ViT-H                -> GPU
Same CLI as generate_gsa_results.py; pass --device cuda.

GSA_MODE=offload (offline only): the repo setting instead, every model on GPU in fp32, but each model waits on the
CPU and is moved to the GPU only for its own call (one model on the GPU at a time). Slower, same numbers as an
unlimited GPU. Peak memory of each stage is printed at the end.
"""
import importlib.util, os
import numpy as np, torch, open_clip

CG_SCRIPT = os.path.expanduser('~/conceptgraph_ws/concept-graphs/conceptgraph/scripts/generate_gsa_results.py')
spec = importlib.util.spec_from_file_location('generate_gsa_results', CG_SCRIPT)
gsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsa)

MODE = os.environ.get('GSA_MODE', 'split')
_to = torch.nn.Module.to
_peak = {}


def on_gpu(name, module, fn):
    _to(module, 'cuda'); torch.cuda.reset_peak_memory_stats()
    try:
        return fn()
    finally:
        _peak[name] = max(_peak.get(name, 0), torch.cuda.max_memory_allocated())
        _to(module, 'cpu'); torch.cuda.empty_cache()


if MODE == 'offload':
    _ram = gsa.ram
    def ram_offload(*a, **k):
        m = _ram(*a, **k); m.to = lambda *aa, **kk: m
        return m
    gsa.ram = ram_offload
    _inference_ram = gsa.inference_ram
    gsa.inference_ram = lambda img, model: on_gpu('ram', model, lambda: _inference_ram(img.cuda(), model))

    _dino_init, _dino_pred = gsa.Model.__init__, gsa.Model.predict_with_classes
    def dino_init(self, *a, **k):
        _dino_init(self, *a, **k); _to(self.model, 'cpu'); torch.cuda.empty_cache()
    gsa.Model.__init__ = dino_init
    gsa.Model.predict_with_classes = lambda self, *a, **k: on_gpu('dino', self.model, lambda: _dino_pred(self, *a, **k))

    _get_sam = gsa.get_sam_predictor
    def sam_offload(*a, **k):
        p = _get_sam(*a, **k); _to(p.model, 'cpu'); torch.cuda.empty_cache()
        return p
    gsa.get_sam_predictor = sam_offload
    _sam_seg = gsa.get_sam_segmentation_from_xyxy
    def sam_seg_offload(*a, **k):
        p = k.get('sam_predictor', a[0] if a else None)
        return on_gpu('sam', p.model, lambda: _sam_seg(*a, **k))
    gsa.get_sam_segmentation_from_xyxy = sam_seg_offload

    _create = open_clip.create_model_and_transforms
    def clip_offload(*a, **k):
        model, pre_train, pre_val = _create(*a, **k); model.to = lambda *aa, **kk: model
        return model, pre_train, pre_val
    gsa.open_clip.create_model_and_transforms = clip_offload
    _ccf = gsa.compute_clip_features
    gsa.compute_clip_features = lambda *a, **k: on_gpu('clip', a[2], lambda: _ccf(*a, **k))
    print('[gsa_fast] offload mode: fp32, one model on the GPU at a time', flush=True)
else:
    # Per-model switches for testing other resident layouts (default = the setting above):
    #   GSA_RAM_GPU=1 RAM on GPU (fp32)   GSA_SAM_FP32=1 SAM without fp16   GSA_CLIP_FP32=1 CLIP without fp16
    flag = lambda k: os.environ.get(k, '0') == '1'

    def stage(name, fn):
        """Record allocated / reserved GPU peaks of one stage (models stay resident)."""
        torch.cuda.reset_peak_memory_stats()
        try:
            return fn()
        finally:
            a_, r_ = _peak.get(name, (0, 0))
            _peak[name] = (max(a_, torch.cuda.max_memory_allocated()), max(r_, torch.cuda.max_memory_reserved()))

    # RAM stays on CPU: ignore .to(device) on the tagging model, feed it CPU tensors.
    _inference_ram = gsa.inference_ram
    if flag('GSA_RAM_GPU'):
        gsa.inference_ram = lambda img, model: stage('ram', lambda: _inference_ram(img.cuda(), model))
    else:
        _ram = gsa.ram
        def ram_on_cpu(*a, **k):
            m = _ram(*a, **k)
            m.to = lambda *aa, **kk: m
            return m
        gsa.ram = ram_on_cpu
        gsa.inference_ram = lambda img, model: stage('ram', lambda: _inference_ram(img.cpu(), model))

    _dino_pred = gsa.Model.predict_with_classes
    gsa.Model.predict_with_classes = lambda self, *a, **k: stage('dino', lambda: _dino_pred(self, *a, **k))

    # CLIP in fp16 on GPU.
    _create = open_clip.create_model_and_transforms
    def create_half(*a, **k):
        model, pre_train, pre_val = _create(*a, **k)
        return (model if flag('GSA_CLIP_FP32') else model.half()), pre_train, pre_val
    gsa.open_clip.create_model_and_transforms = create_half
    _ccf = gsa.compute_clip_features
    def ccf_half(*a, **k):
        with torch.autocast('cuda', dtype=torch.float16, enabled=not flag('GSA_CLIP_FP32')):
            crops, img_f, txt_f = stage('clip', lambda: _ccf(*a, **k))
        return crops, np.asarray(img_f, np.float32), np.asarray(txt_f, np.float32)
    gsa.compute_clip_features = ccf_half

    # SAM ViT-H: fp16 weights (saves ~1.2 GB) and fp16 autocast for the 1024x1024 encoder activations.
    _get_sam = gsa.get_sam_predictor
    def sam_half(*a, **k):
        p = _get_sam(*a, **k)
        if not flag('GSA_SAM_FP32'):
            p.model.half()
        return p
    gsa.get_sam_predictor = sam_half
    _sam_seg = gsa.get_sam_segmentation_from_xyxy
    def sam_seg_half(*a, **k):
        with torch.autocast('cuda', dtype=torch.float16, enabled=not flag('GSA_SAM_FP32')):
            masks = stage('sam', lambda: _sam_seg(*a, **k))
        torch.cuda.empty_cache()
        return masks
    gsa.get_sam_segmentation_from_xyxy = sam_seg_half

# GroundingDINO phrase -> class mapping. The original Model.find_index matches only the first word
# of the phrase as a substring of the first class that contains it, and on failure returns class 0
# (a random-looking label such as 'chocolate cake'). Replacement: exact phrase, then the longest class
# name contained in the phrase, then a whole-word match; unmatched phrases become 'other item'.
def find_index(string, lst):
    s = string.lower().strip()
    lower = [c.lower() for c in lst]
    if s in lower:
        return lower.index(s)
    contained = [i for i, c in enumerate(lower) if c and c in s]
    if contained:
        return max(contained, key=lambda i: len(lower[i]))
    words = s.split()
    word_hits = [i for i, c in enumerate(lower) if words and any(w in c.split() for w in words)]
    if word_hits:
        return min(word_hits, key=lambda i: len(lower[i]))
    print(f'[label-fallback] "{string}" -> other item', flush=True)
    return lower.index('other item') if 'other item' in lower else 0
gsa.Model.find_index = staticmethod(find_index)


# Optional vocabulary control (applied after CG-D's own tag cleanup):
#   GSA_EXTRA_CLASSES_FILE  phrases appended to every frame's class list (e.g. household objects, search targets)
#   GSA_BLOCK_TAGS_FILE     RAM tags removed (scenery / picture content / abstract words / people)
def _read_list(env):
    path = os.environ.get(env, '')
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')] if path else []


#   GSA_TARGET_CLASSES      search-target query names, comma separated (require3: all query_names of the current
#                           /athome/search/targets), appended last like cgd_live_node.py does
_extra, _block = _read_list('GSA_EXTRA_CLASSES_FILE'), {t.lower() for t in _read_list('GSA_BLOCK_TAGS_FILE')}
_targets = [t.strip() for t in os.environ.get('GSA_TARGET_CLASSES', '').split(',') if t.strip()]
if _extra or _block or _targets:
    _ptc = gsa.process_tag_classes
    def process_tag_classes_vocab(*a, **k):
        classes = [c for c in _ptc(*a, **k) if c.lower() not in _block]
        classes += [c for c in _extra if c not in classes]
        return classes + [c for c in _targets if c not in classes]
    gsa.process_tag_classes = process_tag_classes_vocab
    print(f'[gsa_fast] extra classes: {len(_extra)}, blocked tags: {len(_block)}, targets: {_targets}', flush=True)

if __name__ == '__main__':          # importable by the live node (tools/robot_live/cgd_live_node.py)
    gsa.main(gsa.get_parser().parse_args())
    if _peak:
        gb = lambda v: '/'.join(f'{x / 2**30:.2f}' for x in (v if isinstance(v, tuple) else (v,)))
        print('[gsa_fast] stage peak GB (allocated/reserved): ' + ', '.join(f'{k} {gb(v)}' for k, v in _peak.items()), flush=True)
