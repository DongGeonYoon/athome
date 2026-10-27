#!/usr/bin/env bash
# run4-like search recording without the robot (require3 item 5): the run4 bag (camera, LiDAR body cloud, TF) is
# replayed at 1x through the real online pipeline - cgd_live_node (latest image whenever free, like on the robot)
# + geometry_fusion_node mode:=online - while search_server_mock.py publishes /athome/search/targets the way the
# search server does. Everything on ROS_DOMAIN_ID 77 (isolated). Nothing commands a robot.
#   bash search_replay.sh PRIOR_OBJECTS_JSON NAME [--target "remote control:remote control" --target "cups:cup" ...]
#   -> ~/real/results/<NAME>/ : bag/ (same topics as run_search.sh + /cg_d/annotated), annotated/, detections.jsonl,
#      targets_events.jsonl (mock server events), logs/
set -e
PRIOR=$(realpath "$1"); NAME=$2; shift 2
R=$HOME/real/results/$NAME; mkdir -p $R/logs; BAG=$HOME/real/results/run4/bag
source $HOME/real/scripts/env.sh; export ROS_DOMAIN_ID=77
S=$HOME/real/scripts; PY=$HOME/miniforge3/envs/conceptgraph/bin/python
DT=$(python3 -c "import json;print(json.load(open('$HOME/real/results/run4/sync.json')).get('recommended_max_sync_dt_sec', 0.1))")
cp $PRIOR $R/prior_objects.json
setsid ros2 launch semantic_geometry_fusion perception.launch.py mode:=online use_sim_time:=true max_sync_dt_sec:=$DT \
  prior_objects_json:=$PRIOR > $R/logs/fusion.log 2>&1 < /dev/null & FUS=$!
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONHASHSEED=0 setsid $PY $HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py \
  --image-topic /camera/camera/color/image_raw/compressed --save-dir $R > $R/logs/cgd_live.log 2>&1 < /dev/null & CGD=$!
until grep -q "models loaded" $R/logs/cgd_live.log; do sleep 2; kill -0 $CGD || { echo "cgd_live died"; exit 1; }; done
setsid ros2 bag record -o $R/bag /clock /tf /tf_static /Odometry /camera/camera/color/camera_info /cg_d/objects_2d /cg_d/annotated \
  /athome/perception/objects /athome/search/targets > $R/logs/bag.log 2>&1 < /dev/null & REC=$!
python3 $S/search_server_mock.py --command-id 1 "$@" > $R/targets_events.jsonl 2> $R/logs/mock.log & MOCK=$!
sleep 3
echo "[replay] playing run4 bag at 1x ($(date +%T))"
ros2 bag play $BAG --clock 100 --topics /tf /tf_static /Odometry /cloud_registered_body /camera/camera/color/camera_info \
  /camera/camera/color/image_raw/compressed /map > $R/logs/play.log 2>&1
sleep 15                                  # last CG-D frame + fusion
kill -INT $MOCK; sleep 3                  # command end -> empty target list
kill -INT $REC; sleep 3
kill -INT -$CGD -$FUS 2>/dev/null; sleep 5; kill -9 -$CGD -$FUS 2>/dev/null || true
echo "[replay] done: $(wc -l < $R/detections.jsonl) CG-D frames, events: $(wc -l < $R/targets_events.jsonl)"
