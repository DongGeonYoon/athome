#!/usr/bin/env bash
# Online GPU test: real cgd_live_node + vram_live_test.py feeder on ROS_DOMAIN_ID 77 (isolated from the robot).
#   bash vram_live_test.sh NAME [targets ...]   -> ~/real/results/run4/vram_test/live_<NAME>.{log,gpu}
T=$HOME/real/results/run4/vram_test; NAME=$1; shift
source $HOME/real/scripts/env.sh; export ROS_DOMAIN_ID=77
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $T/live_$NAME.gpu & SMI=$!
# node started directly (run_cgd_live.sh re-sources env.sh, which would put it back on the robot domain 25)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONHASHSEED=0 $HOME/miniforge3/envs/conceptgraph/bin/python \
  $HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py --image-topic /vram_test/image > $T/live_${NAME}_node.log 2>&1 & NODE=$!
python3 $HOME/real/scripts/vram_live_test.py $T/sources/run4/results ${1:+--targets "$@"} > $T/live_$NAME.log 2>&1
kill -INT $NODE; sleep 5; kill $NODE 2>/dev/null; kill $SMI
oom=$(grep -c "out of memory\|OutOfMemoryError" $T/live_${NAME}_node.log)
echo "live_$NAME OOM=$([ $oom -gt 0 ] && echo YES || echo no) $(tail -1 $T/live_$NAME.log) gpu_peak_MiB=$(sort -n $T/live_$NAME.gpu | tail -1)" | tee -a $T/results.txt
