#!/usr/bin/env bash
# Live CG-D timing as the search server sees it, on the run4 bag replayed at 1x (ROS_DOMAIN_ID 77):
#   [RATE=1] [RAW=0] bash search_latency_check.sh OUT_DIR [NODE_PY=cgd_live_node.py] [node args...]
#   RAW=1: the node gets raw bgr8 /camera/camera/color/image_raw (as from the live V4L2 camera; raw_republisher.py),
#   RATE=2: bag at 2x (run4 images 4.5 -> 9 fps, like the live USB 2.0 camera). ATHOME_RMW picks the middleware.
# -> OUT_DIR/monitor.json (search_latency_monitor.py), node.log. Local GPU unless --server is given.
OUT=${1:?OUT}; NODE=${2:-$HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py}; shift 2; mkdir -p $OUT
source $HOME/real/scripts/env.sh; export ROS_DOMAIN_ID=77
IMG=/camera/camera/color/image_raw/compressed; NIMG=$IMG
if [ "${RAW:-0}" = 1 ]; then NIMG=/camera/camera/color/image_raw; python3 $HOME/real/scripts/raw_republisher.py > $OUT/republisher.log 2>&1 & RP=$!; fi
PY=$HOME/miniforge3/envs/conceptgraph/bin/python; [[ " $* " == *" --server "* ]] && PY=python3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONHASHSEED=0 setsid $PY $NODE --image-topic $NIMG "$@" > $OUT/node.log 2>&1 < /dev/null & N=$!
until grep -qE "models loaded|GPU server via" $OUT/node.log; do sleep 2; kill -0 $N 2>/dev/null || { echo node died; exit 1; }; done; sleep 2
python3 $HOME/real/scripts/search_latency_monitor.py $OUT/monitor.json --image-topic $NIMG & M=$!; sleep 2
ros2 bag play $HOME/real/results/run4/bag -r ${RATE:-1} --topics /tf /tf_static $IMG /camera/camera/color/camera_info > $OUT/play.log 2>&1
sleep 12; kill -INT $M; sleep 2; kill -INT -$N; sleep 3; kill -9 -$N 2>/dev/null; [ -n "$RP" ] && kill $RP; echo "done $OUT"
