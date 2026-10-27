#!/usr/bin/env bash
# CG-D precision comparison on the run4 offline keyframes (one variant at a time, never two CG-D processes at once):
#   fp32   repo setting: every model on GPU in fp32 (GSA_MODE=offload, one model on the GPU at a time)
#   split  current robot setting: RAM on CPU, SAM ViT-H and CLIP fp16
#   mobile proposal setting: MobileSAM instead of SAM ViT-H, split otherwise
# Each variant -> CG-D detections, GPU peak (nvidia-smi), sec/frame, then the dataset builder -> sg_<variant>/
#   bash precision_compare.sh            -> ~/real/results/run4/precision_cmp/
E=$HOME/real/results/run4/precision_cmp; S=$E/robot/sources/run4
VOC=$HOME/semantic_geometry_ws/tools/replica_rig/vocab
export GSA_EXTRA_CLASSES_FILE=$VOC/household.txt GSA_BLOCK_TAGS_FILE=$VOC/blocklist.txt
export GSA_PATH=$HOME/conceptgraph_ws/Grounded-Segment-Anything PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
PY=$HOME/miniforge3/envs/conceptgraph/bin/python; RUNNER=$HOME/semantic_geometry_ws/tools/replica_rig/gsa_fast.py
N=$(ls $S/results/frame*.jpg | wc -l)
for v in fp32 split mobile; do
  case $v in
    fp32)   mode=offload; sam="";        det=v2fp32 ;;
    split)  mode=split;   sam="";        det=v2split ;;
    mobile) mode=split;   sam=mobilesam; det=mobilesam_v2mobile ;;
  esac
  suf=${det#mobilesam_}
  if [ "$(ls $S/gsa_detections_ram_$det 2>/dev/null | wc -l)" -lt "$N" ]; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $E/gpu_$v.txt & SMI=$!
    t0=$(date +%s.%N)
    (cd $HOME/conceptgraph_ws/concept-graphs/conceptgraph && GSA_MODE=$mode $PY $RUNNER --dataset_root $E/robot/sources \
       --dataset_config $S/cgd.yaml --scene_id run4 --start 0 --end -1 --stride 1 --class_set ram ${sam:+--sam_variant $sam} \
       --box_threshold 0.2 --text_threshold 0.2 --add_bg_classes --exp_suffix $suf --device cuda) > $E/cgd_$v.log 2>&1
    rc=$?; t1=$(date +%s.%N); kill $SMI
    echo "$v exit=$rc frames=$(ls $S/gsa_detections_ram_$det | wc -l) sec=$(awk "BEGIN{printf \"%.0f\", $t1 - $t0}") gpu_peak_MiB=$(sort -n $E/gpu_$v.txt | tail -1)" | tee -a $E/runs.txt
  fi
  $PY $HOME/semantic_geometry_ws/tools/scene_graph_dataset/build_dataset.py --exp $E --sources robot --det-suffix $det \
     --out-name sg_$v > $E/build_$v.log 2>&1 && echo "[built] sg_$v"
done
