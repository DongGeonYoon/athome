#!/usr/bin/env bash
# Pre-mapping recording on the robot PC (sensors only; the Scene Graph is built offline from the bag, prebuild.sh).
# Nothing here commands the robot: drive it by hand, slowly, and stop often so every place is seen from standstill.
#   bash record_premap.sh [NAME]        -> ~/real/results/<NAME or premap_YYYYmmdd_HHMMSS>/bag
#   bash record_premap.sh stop          close the bag (SIGINT), stop what this script started
#   MAP_PCD=<pcd> MAP_YAML=<nav2 yaml> bash record_premap.sh NAME   localize on another map (default lab map);
#                                       written to NAME/map.env, prebuild.sh builds with the same map
#   CAMERA=realsense bash record_premap.sh NAME   USB 3.0: official driver, colour 1280x720 @ 30 fps + depth aligned to colour
#                                       @ 6 fps, recorded with the bag. Check first that colour camera_info K[0] is 903.47 (HFOV 70.6 deg)
# 1 localization (Livox + FAST_LIO_LOCALIZATION2 on the lab map or MAP_PCD / MAP_YAML), unless running
# 2 camera: D435 colour 1280x720 @ 10 fps via the V4L2 node (works on USB 2.0, no depth), unless a camera topic exists
# 3 wait until the start pose is set in RViz ("2D Pose Estimate"): recording starts only after that, because the map frame is
#   wrong before it (a TF exists from the start, so its presence alone does not tell)
# 4 camera/LiDAR timing check, rosbag of the sensor topics the offline pipeline needs
S=$HOME/real/scripts; PIDS=$HOME/real/results/premap.pids
if [ "$1" = stop ]; then
  [ -f $PIDS ] && for p in $(tac $PIDS); do kill -INT $p 2>/dev/null; done; sleep 5
  [ -f $PIDS ] && for p in $(cat $PIDS); do kill $p 2>/dev/null; done; rm -f $PIDS; echo "[premap] stopped"; exit 0
fi
NAME=${1:-premap_$(date +%Y%m%d_%H%M%S)}; R=$HOME/real/results/$NAME; mkdir -p $R/logs
source $S/env.sh; export DISPLAY=${DISPLAY:-:0}; : > $PIDS
LOC_LOG=""
pgrep -f "slam_costmap_bringup.launch.p[y]" >/dev/null || { LOC_LOG=$R/logs/localization.log; setsid bash $S/start_localization.sh > $LOC_LOG 2>&1 < /dev/null & echo $! >> $PIDS; }
DEPTH_TOPICS=""
if [ "${CAMERA:-v4l2}" = realsense ]; then
  DEPTH_TOPICS="/camera/camera/aligned_depth_to_color/image_raw /camera/camera/aligned_depth_to_color/camera_info"
fi
if ! timeout 5 ros2 topic list 2>/dev/null | grep -q "^/camera/camera/color/image_raw$"; then
  if [ "${CAMERA:-v4l2}" = realsense ]; then
    setsid ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=1280x720x30 enable_color:=true \
      enable_depth:=true depth_module.depth_profile:=1280x720x6 align_depth.enable:=true > $R/logs/camera.log 2>&1 < /dev/null & echo $! >> $PIDS
  else
    setsid bash $S/start_camera_v4l2.sh 1280 720 10 --no-depth > $R/logs/camera.log 2>&1 < /dev/null & echo $! >> $PIDS
  fi
fi
echo "[premap] set the start pose in RViz (2D Pose Estimate); recording starts after it ..."
if [ -n "$LOC_LOG" ]; then
  until grep -q "ICP relocalization" $LOC_LOG 2>/dev/null; do sleep 1; done        # pose received and matched to the PCD map
else
  read -r -p "[premap] localization was already running: set the pose in RViz if not done, then press Enter " _
fi
until timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q Translation; do sleep 2; done
timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -A1 Translation | tee $R/start_pose.txt
python3 $S/check_sync.py 10 $R/sync.json | tee $R/logs/sync.txt
cp $HOME/real/config/calibration.yaml $R/
M=$HOME/fast_lio_ws/maps; echo -e "MAP_PCD=${MAP_PCD:-$M/lab_map_20261005_174210.pcd}\nMAP_YAML=${MAP_YAML:-$M/lab_map_20261005_174210_structural.yaml}" > $R/map.env
setsid ros2 bag record -o $R/bag /tf /tf_static /map /livox/lidar /livox/imu /cloud_registered_body /Odometry \
  /camera/camera/color/camera_info /camera/camera/color/image_raw/compressed $DEPTH_TOPICS > $R/logs/bag.log 2>&1 < /dev/null & echo $! >> $PIDS
echo "[premap] recording -> $R/bag   (stop: bash $S/record_premap.sh stop)"
