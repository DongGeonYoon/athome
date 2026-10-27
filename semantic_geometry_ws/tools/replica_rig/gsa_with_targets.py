"""
Run CG-D generate_gsa_results.py unchanged, but append search-target names to the
GroundingDINO class query of every frame (require.md item 3).

Targets are appended AFTER CG-D's own tag cleanup: CG-D drops any class containing
words such as 'photo', 'shade', 'corner', 'room', so a target like 'photo frame'
would otherwise be removed. Detections from a target query keep the exact target
string as their label.

Usage (same CLI as generate_gsa_results.py):
  SEARCH_TARGETS="vase,book,remote control" python gsa_with_targets.py --dataset_root ... --exp_suffix targets
"""
import importlib.util, os, sys

CG_SCRIPT = os.path.expanduser('~/conceptgraph_ws/concept-graphs/conceptgraph/scripts/generate_gsa_results.py')
targets = [t.strip() for t in os.environ.get('SEARCH_TARGETS', '').split(',') if t.strip()]

spec = importlib.util.spec_from_file_location('generate_gsa_results', CG_SCRIPT)
gsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsa)

_original = gsa.process_tag_classes


def process_tag_classes_with_targets(*args, **kwargs):
    classes = _original(*args, **kwargs)
    return classes + [t for t in targets if t not in classes]


gsa.process_tag_classes = process_tag_classes_with_targets
print(f'[gsa_with_targets] appended targets: {targets}', flush=True)
gsa.main(gsa.get_parser().parse_args())
