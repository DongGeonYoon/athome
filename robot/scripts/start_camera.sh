#!/usr/bin/env bash
# librealsense node (USB 3 only: on USB 2.0 this camera delivered no frames, see README). USB 2.0: start_camera_v4l2.sh
# 15 fps because the camera is on USB 2.0; use 30 on USB 3.
FPS=${1:-15}
source $HOME/real/scripts/env.sh
exec ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640,480,$FPS depth_module.depth_profile:=640,480,$FPS \
  align_depth.enable:=true enable_sync:=true pointcloud.enable:=false enable_gyro:=false enable_accel:=false
