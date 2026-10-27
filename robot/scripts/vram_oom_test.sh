#!/usr/bin/env bash
# Every CG-D GPU layout that can be chosen, one at a time on the run4 keyframes (28 x 1280x720): OOM or not, GPU peak.
#   bash vram_oom_test.sh [config ...]   -> ~/real/results/run4/vram_test/{results.txt, log_<config>.txt}
# config = name:GSA_MODE:sam_variant:env switches (gsa_fast.py: GSA_RAM_GPU / GSA_SAM_FP32 / GSA_CLIP_FP32)
T=$HOME/real/results/run4/vram_test; S=$T/sources/run4
VOC=$HOME/semantic_geometry_ws/tools/replica_rig/vocab
export GSA_EXTRA_CLASSES_FILE=$VOC/household.txt GSA_BLOCK_TAGS_FILE=$VOC/blocklist.txt
export GSA_PATH=$HOME/conceptgraph_ws/Grounded-Segment-Anything PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
PY=$HOME/miniforge3/envs/conceptgraph/bin/python; RUNNER=$HOME/semantic_geometry_ws/tools/replica_rig/gsa_fast.py
ALL="split:split:: ramgpu:split::GSA_RAM_GPU=1 samfp32:split::GSA_SAM_FP32=1 clipfp32:split::GSA_CLIP_FP32=1
ramgpu_samfp32:split::GSA_RAM_GPU=1,GSA_SAM_FP32=1 samfp32_clipfp32:split::GSA_SAM_FP32=1,GSA_CLIP_FP32=1
repo_resident:split::GSA_RAM_GPU=1,GSA_SAM_FP32=1,GSA_CLIP_FP32=1 offload_fp32:offload::
mobile:split:mobilesam: mobile_repo_resident:split:mobilesam:GSA_RAM_GPU=1,GSA_SAM_FP32=1,GSA_CLIP_FP32=1"
for c in ${@:-$ALL}; do
  IFS=: read name mode sam envs <<< "$c"
  rm -rf $S/gsa_*_$name $S/gsa_*_${name}.json
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $T/gpu_$name.txt & SMI=$!
  t0=$(date +%s)
  (cd $HOME/conceptgraph_ws/concept-graphs/conceptgraph && env GSA_MODE=$mode ${envs//,/ } $PY $RUNNER --dataset_root $T/sources \
     --dataset_config $S/cgd.yaml --scene_id run4 --start 0 --end -1 --stride 1 --class_set ram ${sam:+--sam_variant $sam} \
     --box_threshold 0.2 --text_threshold 0.2 --add_bg_classes --exp_suffix $name --device cuda) > $T/log_$name.txt 2>&1
  rc=$?; kill $SMI; sec=$(( $(date +%s) - t0 ))
  n=$(ls $S/gsa_detections_ram_*$name 2>/dev/null | wc -l)
  oom=$(grep -c "out of memory\|OutOfMemoryError" $T/log_$name.txt)
  echo "$name exit=$rc OOM=$([ $oom -gt 0 ] && echo YES || echo no) frames=$n/28 sec=$sec gpu_peak_MiB=$(sort -n $T/gpu_$name.txt | tail -1) $(grep -o 'stage peak.*' $T/log_$name.txt)" | tee -a $T/results.txt
  sleep 3
done
