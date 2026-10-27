#!/usr/bin/env bash
# Run CG-D (CPU) on every active-observation scene dir; skips dirs that are already complete,
# so the script can simply be re-run after an interruption.
set -u
OUT=${1:-$HOME/exp1}
ROOT=$OUT/active_obs
PY=$HOME/miniforge3/envs/conceptgraph/bin/python
export GSA_PATH=$HOME/conceptgraph_ws/Grounded-Segment-Anything
CG=$HOME/conceptgraph_ws/concept-graphs/conceptgraph
TOOLS=$(cd "$(dirname "$0")" && pwd)
TARGET_SCENE=loc2_A6
export SEARCH_TARGETS="vase,book,remote control,throw blanket,picture frame"

run_scene () {  # $1 scene, $2 suffix, $3 runner
  local scene=$1 suffix=$2 runner=$3 cfg=${1##*_}
  local n_img=$(ls "$ROOT/$scene/results"/frame*.jpg | wc -l)
  local n_det=$(ls "$ROOT/$scene/gsa_detections_ram_$suffix" 2>/dev/null | wc -l)
  if [ "$n_det" -ge "$n_img" ]; then echo "[skip] $scene ($suffix) done"; return; fi
  echo "[run ] $scene ($suffix) $(date +%T)"
  local t0=$(date +%s)
  (cd "$CG" && $PY "$runner" --dataset_root "$ROOT" --dataset_config "$ROOT/$cfg.yaml" --scene_id "$scene" \
     --start 0 --end -1 --stride 1 --class_set ram --box_threshold 0.2 --text_threshold 0.2 \
     --add_bg_classes --accumu_classes --exp_suffix "$suffix" --device cpu) >> "$ROOT/cgd_${scene}_${suffix}.log" 2>&1
  echo "[done] $scene ($suffix) exit=$? sec=$(( $(date +%s) - t0 )) frames=$n_img" | tee -a "$ROOT/cgd_timing.txt"
}

for scene in loc0_A6 loc0_B8 loc1_A6 loc1_B8 loc2_A6 loc2_B8; do
  run_scene "$scene" base "$CG/scripts/generate_gsa_results.py"
done
run_scene "$TARGET_SCENE" targets "$TOOLS/gsa_with_targets.py"
echo "ALL DONE $(date +%T)"
