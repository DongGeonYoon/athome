#!/usr/bin/env bash
# Team bringup without costmaps (costmap = Nav2 on the navigation laptop, require2 4): Livox + FAST_LIO_LOCALIZATION2
# on the lab map (map -> camera_init -> body), body -> base_link, /map (2D structural map). The map frame is the same
# for /map, the Scene Graph objects.json and /athome/perception/objects.
# RViz opens: set the robot pose once with "2D Pose Estimate" (global_localization waits for /initialpose).
source $HOME/real/scripts/env.sh
export DISPLAY=${DISPLAY:-:0}
# global_localization.py imports open3d, which drags in user-site sklearn/pandas built for a newer numpy: stub them
export PYTHONPATH=$HOME/real/config/py_shim:$PYTHONPATH
exec ros2 launch mppi_nav slam_costmap_bringup.launch.py run_livox:=true run_localization:=true run_map_server:=true \
  run_static_costmap:=false run_local_costmap:=false run_local_cloud_filter:=false run_zenoh:=false run_rviz:=true
