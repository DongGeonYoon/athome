#!/usr/bin/env bash
# Pre-mapping recording on the robot PC (sensors only, no processing here: the GPU server builds the Scene Graph,
# premap_to_server.sh). Nothing here commands the robot; drive / rotate it by hand (heading guide below).
#   bash record_premap.sh [NAME]        -> ~/real/results/<NAME or premap_YYYYmmdd_HHMMSS>/bag
#   bash record_premap.sh stop          close the bag (SIGINT), stop what this script started
#   CAMERA=realsense bash record_premap.sh NAME   USB 3.0 cable (require4 1): official driver, colour 1280x720 @ 30 fps
#                                       + depth aligned to colour @ 6 fps (low rate for size), recorded with the bag.
#                                       Check first that colour camera_info K[0] is still 903.47 (horizontal FOV 70.6 deg)
# 1 localization (Livox + FAST_LIO_LOCALIZATION2 on the lab map), unless running
# 2 camera: D435 on the USB 2.0 hub port via the V4L2 node, colour 1280x720 @ 10 fps, no depth (require2 option A;
#   librealsense gives no frames over USB 2.0), unless a camera topic already exists
# 3 wait for a pose (RViz "2D Pose Estimate"), camera/LiDAR timing check
# 4 rosbag of the sensor topics the offline pipeline needs
# Camera = the calibrated D435 (calibration.yaml / robot.yaml T_cam_body valid; decision 2026-10-28).
S=$HOME/real/scripts; PIDS=$HOME/real/results/premap.pids
if [ "$1" = stop ]; then
  [ -f $PIDS ] && for p in $(tac $PIDS); do kill -INT $p 2>/dev/null; done; sleep 5
  [ -f $PIDS ] && for p in $(cat $PIDS); do kill $p 2>/dev/null; done; rm -f $PIDS; echo "[premap] stopped"; exit 0
fi
NAME=${1:-premap_$(date +%Y%m%d_%H%M%S)}; R=$HOME/real/results/$NAME; mkdir -p $R/logs
source $S/env.sh; export DISPLAY=${DISPLAY:-:0}; : > $PIDS
pgrep -f "slam_costmap_bringup.launch.p[y]" >/dev/null || { setsid bash $S/start_localization.sh > $R/logs/localization.log 2>&1 < /dev/null & echo $! >> $PIDS; }
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
echo "[premap] waiting for localization (RViz: 2D Pose Estimate) ..."
until timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q Translation; do sleep 2; done
timeout 3 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -A1 Translation | tee $R/start_pose.txt
python3 $S/check_sync.py 10 $R/sync.json | tee $R/logs/sync.txt
cp $HOME/real/config/calibration.yaml $R/
setsid ros2 bag record -o $R/bag /tf /tf_static /map /livox/lidar /livox/imu /cloud_registered_body /Odometry \
  /camera/camera/color/camera_info /camera/camera/color/image_raw/compressed $DEPTH_TOPICS > $R/logs/bag.log 2>&1 < /dev/null & echo $! >> $PIDS
echo "[premap] recording -> $R/bag   (rotation guide: python3 $S/heading_guide.py; stop: bash $S/record_premap.sh stop)"
