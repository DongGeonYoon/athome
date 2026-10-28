#!/usr/bin/env bash
# Perception side of a search run with the navigation laptop (high planner + Nav2). Nothing here commands the robot.
#   bash run_search.sh PRIOR_OBJECTS_JSON [NAME]      -> ~/real/results/<NAME or search_YYYYmmdd_HHMMSS>/
# PRIOR_OBJECTS_JSON: the pre-mapped Scene Graph objects.json (same map as /map), e.g.
#   ~/real_run4_update_20261023/scene_graph_dataset/scenes/robot_run4/objects.json
# Roles (require2.md): this PC publishes TF map -> base_link, /map, /athome/perception/objects
# (athome_interfaces/ObservedObjectArray); the laptop runs Nav2 (costmap, map_server off) and the search server, and
# publishes /athome/search/targets. Same ROS_DOMAIN_ID / RMW on both sides (env.sh).
# 1 localization + camera, unless already running   2 wait for a pose (RViz "2D Pose Estimate")
# 3 timing check -> max_sync_dt_sec   4 live CG-D (GPU server by default, CGD=local) + geometry_fusion_node (online)
# 5 wait for the first observation
# Stop with: bash ~/real/scripts/stop_experiment.sh
set -e
PRIOR=$(realpath "${1:?usage: run_search.sh PRIOR_OBJECTS_JSON [NAME]}")
[ -f "$PRIOR" ] || { echo "no such file: $PRIOR"; exit 1; }
NAME=${2:-search_$(date +%Y%m%d_%H%M%S)}; R=$HOME/real/results/$NAME; mkdir -p $R/logs
source $HOME/real/scripts/env.sh
echo $R > $HOME/real/results/CURRENT
S=$HOME/real/scripts
if pgrep -f "geometry_fusion_nod[e]" >/dev/null; then
  echo "[search] geometry_fusion_node already running (mapping run?): stop it first (stop_experiment.sh)"; exit 1
fi
pgrep -f "slam_costmap_bringup.launch.p[y]" >/dev/null || setsid bash $S/start_localization.sh > $R/logs/localization.log 2>&1 < /dev/null &
pgrep -f "v4l2_rgbd_nod[e]" >/dev/null || setsid bash $S/start_camera_v4l2.sh 1280 720 10 --no-depth > $R/logs/camera.log 2>&1 < /dev/null &
echo "[search] waiting for localization (RViz: 2D Pose Estimate) ..."
until timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q Translation; do sleep 2; done
timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -A1 Translation | tee $R/start_pose.txt
python3 $S/check_sync.py 10 $R/sync.json | tee $R/logs/sync.txt
DT=$(python3 -c "import json;print(json.load(open('$R/sync.json')).get('recommended_max_sync_dt_sec', 0.1))")
echo "[search] max_sync_dt_sec = $DT, prior map = $PRIOR"
cp $HOME/real/config/calibration.yaml $PRIOR $R/
# CG-D: on the GPU server by default (CGD=server: tunnel kept open by cgd_server_link.sh, this PC only relays images),
# CGD=local runs it on this PC's GPU
if [ "${CGD:-server}" = server ]; then
  setsid bash $S/cgd_server_link.sh > $R/logs/cgd_server_link.log 2>&1 < /dev/null &
  until grep -q "cgd_server ready" $R/logs/cgd_server_link.log 2>/dev/null; do sleep 2; done
  setsid python3 $HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py --server 127.0.0.1:47000 \
    --image-topic /camera/camera/color/image_raw --save-dir $R > $R/logs/cgd_live.log 2>&1 < /dev/null &
else
  setsid bash $S/run_cgd_live.sh --image-topic /camera/camera/color/image_raw --save-dir $R > $R/logs/cgd_live.log 2>&1 < /dev/null &
fi
setsid ros2 launch semantic_geometry_fusion perception.launch.py mode:=online max_sync_dt_sec:=$DT \
  prior_objects_json:=$PRIOR > $R/logs/fusion.log 2>&1 < /dev/null &
setsid ros2 bag record -o $R/bag /tf /tf_static /Odometry /camera/camera/color/camera_info \
  /cg_d/objects_2d /athome/perception/objects /athome/search/targets > $R/logs/bag.log 2>&1 < /dev/null &
echo "[search] waiting for the first /athome/perception/objects (CG-D model load + first frame) ..."
until timeout 15 ros2 topic echo --once /athome/perception/objects --field clip_model >/dev/null 2>&1; do sleep 2; done
echo "[search] perception ready -> start the search server on the laptop   (logs: $R/logs)"
