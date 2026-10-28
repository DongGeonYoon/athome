#!/usr/bin/env bash
# Same-input check of geometry_fusion_node (ROS) on any machine: run4 bag (LiDAR body cloud, TF, camera_info) at 1x
# + the offline CG-D detections of its keyframes (semantic_replay.py) -> mapping mode -> Scene Graph JSON.
# ROS_DOMAIN_ID 79 (isolated). Local: env.sh; GPU server: SERVER=1 (ros_env.sh, conceptgraph python for the replay).
#   [FUSION_ARGS='-p name:=value ...'] bash fusion_replay_check.sh OUT_DIR   (extra node parameters, e.g. to compare settings)
OUT=${1:?OUT_DIR}; mkdir -p $OUT
if [ "${SERVER:-0}" = 1 ]; then source /data/athome/server_sync/ros_env.sh; PYR="$CGD_PY"; CFG=$WS/install/semantic_geometry_fusion/share/semantic_geometry_fusion/config
else source $HOME/real/scripts/env.sh; PYR=python3; CFG=$HOME/semantic_geometry_ws/install/semantic_geometry_fusion/share/semantic_geometry_fusion/config; fi
export ROS_DOMAIN_ID=79
R4=$HOME/real/results/run4; SRC=$R4/offline5/robot/sources/run4
DT=$(python3 -c "import json;print(json.load(open('$R4/sync.json')).get('recommended_max_sync_dt_sec', 0.1))")
# own process groups (setsid): `ros2 run` does not pass SIGINT on, so the node itself must be signalled, or it keeps
# running on this domain and mixes the next run's data into its JSON
setsid ros2 run semantic_geometry_fusion geometry_fusion_node --ros-args --params-file $CFG/fusion_params.yaml --params-file $CFG/robot.yaml \
  -p use_sim_time:=true -p max_sync_dt_sec:=$DT -p sensor_buffer_sec:=120.0 -p scene_graph_json_path:=$OUT/objects.json $FUSION_ARGS \
  > $OUT/node.log 2>&1 & NODE=$!
setsid $PYR $HOME/semantic_geometry_ws/tools/sim_sources/semantic_replay.py $SRC/gsa_detections_ram_v2 $SRC/keyframe_stamps.csv \
  > $OUT/replay.log 2>&1 & REP=$!
sleep 6
ros2 bag play $R4/bag --clock 100 --topics /tf /tf_static /cloud_registered_body /camera/camera/color/camera_info > $OUT/play.log 2>&1
sleep 8; kill -INT -$REP -$NODE; for i in $(seq 30); do pgrep -g $NODE >/dev/null || break; sleep 1; done; kill -9 -$NODE -$REP 2>/dev/null
python3 -c "import json;print('objects', len(json.load(open('$OUT/objects.json'))['objects']))"
