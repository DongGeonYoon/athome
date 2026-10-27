#!/usr/bin/env bash
# One real-robot perception run (sensors + perception only; nothing here commands the robot).
#   bash run_experiment.sh [NAME]      -> ~/real/results/<NAME or run_YYYYmmdd_HHMMSS>/
# Setup agreed in require2.md:
#   - detection on the RGB image, 1280x720 (fx 903.47 -> horizontal FOV 70.6 deg), D435 via V4L2 (USB 2.0)
#   - one map frame for /map, objects.json and /athome/perception/objects: FAST_LIO_LOCALIZATION2 on the lab map
#     (floor z = -0.41 m in that frame, see config/calibration.yaml)
# 1 localization (Livox + FAST_LIO_LOCALIZATION2 + /map + body->base_link) and camera, unless already running
# 2 wait until localization has a pose (set it once in RViz with "2D Pose Estimate")
# 3 timing check -> max_sync_dt_sec
# 4 live CG-D (annotated frames, detections.jsonl) + geometry_fusion_node (mapping mode -> objects.json)
# 5 rosbag (offline rebuild with tools/sim_sources/offline_scene_graph.sh)
# Stop with: bash ~/real/scripts/stop_experiment.sh   (SIGINT: final objects.json, bag closed)
NAME=${1:-run_$(date +%Y%m%d_%H%M%S)}; R=$HOME/real/results/$NAME; mkdir -p $R/logs
source $HOME/real/scripts/env.sh
echo $R > $HOME/real/results/CURRENT
S=$HOME/real/scripts
pgrep -f "slam_costmap_bringup.launch.p[y]" >/dev/null || setsid bash $S/start_localization.sh > $R/logs/localization.log 2>&1 < /dev/null &
pgrep -f "v4l2_rgbd_nod[e]" >/dev/null || setsid bash $S/start_camera_v4l2.sh 1280 720 10 --no-depth > $R/logs/camera.log 2>&1 < /dev/null &
echo "[experiment] waiting for localization (RViz: 2D Pose Estimate) ..."
until timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q Translation; do sleep 2; done
timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -A1 Translation | tee $R/start_pose.txt
python3 $S/check_sync.py 10 $R/sync.json | tee $R/logs/sync.txt
DT=$(python3 -c "import json;print(json.load(open('$R/sync.json')).get('recommended_max_sync_dt_sec', 0.1))")
echo "[experiment] max_sync_dt_sec = $DT"
cp $HOME/real/config/calibration.yaml $R/
setsid bash $S/run_cgd_live.sh --image-topic /camera/camera/color/image_raw --save-dir $R > $R/logs/cgd_live.log 2>&1 < /dev/null &
setsid ros2 launch semantic_geometry_fusion perception.launch.py mode:=mapping max_sync_dt_sec:=$DT \
  scene_graph_json_path:=$R/objects.json > $R/logs/fusion.log 2>&1 < /dev/null &
setsid ros2 bag record -o $R/bag /tf /tf_static /map /livox/lidar /livox/imu /cloud_registered_body /Odometry \
  /camera/camera/color/camera_info /camera/camera/color/image_raw/compressed \
  /cg_d/objects_2d /cg_d/annotated /athome/perception/objects /athome/search/targets > $R/logs/bag.log 2>&1 < /dev/null &
setsid python3 $S/monitor_run.py $R > $R/logs/monitor.log 2>&1 < /dev/null &   # association / accumulation / SG report
echo "[experiment] running -> $R   (view: rqt_image_view /cg_d/annotated)"
