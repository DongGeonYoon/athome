#!/usr/bin/env bash
# End-to-end check of the real ROS node on a recorded Gazebo episode:
#   rosbag (sim time) -> FAST-LIO /cloud_registered_body + TF + camera_info (+ aligned depth)
#   semantic_replay.py -> /cg_d/objects_2d (CG-D detections of the extracted keyframes, at their bag stamps)
#   geometry_fusion_node -> scene_graph_json_path (same schema as the offline dataset)
#   bash ros_replay.sh BAG_DIR SOURCE_DIR OUT_JSON [RATE=3] [SUFFIX=v2]
BAG=$1; SRC=$2; OUT=$3; RATE=${4:-3}; SUFFIX=${5:-v2}
source $HOME/sim_ws/env.sh
source $HOME/semantic_geometry_ws/install/setup.bash
CFG=$HOME/semantic_geometry_ws/install/semantic_geometry_fusion/share/semantic_geometry_fusion/config
# 1280x720 SAM masks make ~20 MB detection messages: same-host shared memory, otherwise they arrive late over UDP
export FASTRTPS_DEFAULT_PROFILES_FILE=$CFG/fastdds_shm.xml RMW_FASTRTPS_USE_QOS_FROM_XML=0
# Same parameters as the robot (fusion_params.yaml + robot.yaml: body cloud, T_cam_body). Check that the episode's
# camera<-body transform is the one in robot.yaml.
python3 - "$SRC" "$CFG/robot.yaml" <<'PYEOF' || exit 1
import sys, yaml, numpy as np
T_src = np.load(sys.argv[1] + '/lidar/frame000000.npz')['T_cam_sensor']
T_cfg = np.array(yaml.safe_load(open(sys.argv[2]))['geometry_fusion_node']['ros__parameters']['T_cam_lidar']).reshape(4, 4)
err = np.abs(T_src - T_cfg).max(); print(f'[replay] T_cam_body episode vs robot.yaml: max abs diff {err:.2e}')
sys.exit(0 if err < 1e-3 else 1)   # 1 mm / 0.06 deg (URDF stores the calib rounded)
PYEOF
DEPTH=/camera/camera/aligned_depth_to_color/depth/image_raw
ros2 bag info "$BAG" 2>/dev/null | grep -q "$DEPTH" || DEPTH=""
LOG=$(dirname "$OUT"); mkdir -p "$LOG"; rm -f "$OUT"
ros2 run semantic_geometry_fusion geometry_fusion_node --ros-args --params-file $CFG/fusion_params.yaml \
  --params-file $CFG/robot.yaml -p use_sim_time:=true -p max_sync_dt_sec:=0.3 -p sensor_buffer_sec:=${BUFFER_SEC:-60.0} \
  -p depth_topic:="${DEPTH:-''}" -p scene_graph_json_path:="$OUT" > "$LOG/node.log" 2>&1 & NODE=$!
python3 $HOME/semantic_geometry_ws/tools/sim_sources/semantic_replay.py "$SRC/gsa_detections_ram_$SUFFIX" \
  "$SRC/keyframe_stamps.csv" > "$LOG/replay.log" 2>&1 & REPLAY=$!
sleep 5
TOPICS="/tf /tf_static /cloud_registered_body /camera/camera/color/camera_info $DEPTH"
ros2 bag play "$BAG" --clock 100 -r "$RATE" --topics $TOPICS > "$LOG/play.log" 2>&1
sleep 5
# SIGINT -> the node writes the final Scene Graph JSON on shutdown; give it time before forcing
kill -INT $REPLAY; pkill -INT -f "lib/semantic_geometry_fusion/geometry_fusion_node"
for i in $(seq 30); do pgrep -f "lib/semantic_geometry_fusion/geometry_fusion_node" >/dev/null || break; sleep 1; done
pkill -9 -f "lib/semantic_geometry_fusion/geometry_fusion_node"; kill -9 $NODE $REPLAY 2>/dev/null
echo "[replay] done depth_topic='${DEPTH}' json=$OUT objects=$(python3 -c "import json;print(len(json.load(open('$OUT'))['objects']))" 2>/dev/null)"
