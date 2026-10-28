#!/usr/bin/env bash
# Team bringup: Livox + FAST_LIO_LOCALIZATION2 on the lab map (map -> camera_init -> body), body -> base_link,
# /map (2D structural map). The map frame is the same for /map, the Scene Graph objects.json and
# /athome/perception/objects.
# Costmaps: global costmap = Nav2 on the navigation laptop (require2 4). The MPPI local costmap (/local_costmap/costmap)
# is not started here unless LOCAL_COSTMAP=1 (team check 2026-10-28: which PC runs it is still open).
# Middleware: rmw_zenoh_cpp; the router is started by env.sh (ATHOME_ZENOH_ENDPOINT connects it to another PC), so the
# bringup's own router stays off (one router per PC, port 7447).
# RViz opens: set the robot pose once with "2D Pose Estimate" (global_localization waits for /initialpose).
source $HOME/real/scripts/env.sh
export DISPLAY=${DISPLAY:-:0}
# global_localization.py imports open3d, which drags in user-site sklearn/pandas built for a newer numpy: stub them
export PYTHONPATH=$HOME/real/config/py_shim:$PYTHONPATH
LC=false; [ "${LOCAL_COSTMAP:-0}" = 1 ] && LC=true
exec ros2 launch mppi_nav slam_costmap_bringup.launch.py run_livox:=true run_localization:=true run_map_server:=true \
  run_static_costmap:=false run_local_costmap:=$LC run_local_cloud_filter:=$LC run_zenoh:=false run_rviz:=true
