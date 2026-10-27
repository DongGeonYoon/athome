#!/usr/bin/env bash
# CG-D (GPU split runner) on every scene dir of a common-intermediate-format source root.
#   bash run_cgd_sources.sh ROOT [SUFFIX=v2]
# Default vocabulary (chosen by the ProcTHOR vocabulary study, see scene_graph_dataset/README): per-frame RAM tags
# minus vocab/blocklist.txt plus vocab/household.txt, no cross-frame tag accumulation (ACCUMU=1 restores it).
# VOCAB=0 runs plain CG-D tags.
# Each ROOT/<scene>/ needs results/frame*.jpg, depth*.png, traj.txt, camera.json. Resumable per scene.
ROOT=$1; SUFFIX=${2:-v2}
VOC=$HOME/semantic_geometry_ws/tools/replica_rig/vocab
if [ "${VOCAB:-1}" = 1 ]; then export GSA_EXTRA_CLASSES_FILE=$VOC/household.txt GSA_BLOCK_TAGS_FILE=$VOC/blocklist.txt; fi
ACC=""; [ "${ACCUMU:-0}" = 1 ] && ACC="--accumu_classes"
export GSA_PATH=$HOME/conceptgraph_ws/Grounded-Segment-Anything
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
PY=$HOME/miniforge3/envs/conceptgraph/bin/python
RUNNER=$HOME/semantic_geometry_ws/tools/replica_rig/gsa_fast.py
for sd in "$ROOT"/*/; do
  s=$(basename "$sd")
  [ -f "$sd/camera.json" ] || continue
  n_img=$(ls "$sd"/results/frame*.jpg 2>/dev/null | wc -l)
  n_done=$(ls "$sd/gsa_detections_ram_$SUFFIX" 2>/dev/null | wc -l)
  if [ "$n_done" -ge "$n_img" ]; then echo "[skip] $s $n_done/$n_img"; continue; fi
  $PY - "$sd" <<'PYEOF'
import json, sys, os
sd = sys.argv[1]; c = json.load(open(os.path.join(sd, 'camera.json'))); K = c['K']
open(os.path.join(sd, 'cgd.yaml'), 'w').write(
    f"dataset_name: 'replica'\ncamera_params:\n  image_height: {c['height']}\n  image_width: {c['width']}\n"
    f"  fx: {K[0][0]}\n  fy: {K[1][1]}\n  cx: {K[0][2]}\n  cy: {K[1][2]}\n  png_depth_scale: 6553.5\n  crop_edge: 0\n")
PYEOF
  echo "[run ] $s start=$n_done frames=$n_img $(date +%T)"
  t0=$(date +%s)
  (cd $HOME/conceptgraph_ws/concept-graphs/conceptgraph && $PY "$RUNNER" --dataset_root "$ROOT" \
     --dataset_config "$sd/cgd.yaml" --scene_id "$s" --start "$n_done" --end -1 --stride 1 --class_set ram \
     --box_threshold 0.2 --text_threshold 0.2 --add_bg_classes $ACC --exp_suffix "$SUFFIX" --device cuda) \
     >> "$ROOT/cgd_$s.log" 2>&1
  echo "[done] $s exit=$? $(ls "$sd/gsa_detections_ram_$SUFFIX" | wc -l)/$n_img sec=$(( $(date +%s) - t0 ))"
done
echo "ALL DONE $(date +%T)"
