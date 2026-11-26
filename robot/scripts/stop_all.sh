#!/usr/bin/env bash
# Stop everything started from robot/scripts: pre-mapping recording, camera, localization, RViz.
bash $(dirname $(readlink -f $0))/record_premap.sh stop
pkill -INT -f "v4l2_rgbd_nod[e]"; pkill -INT -f "slam_costmap_bringup.launch.p[y]"
sleep 5
pkill -9 -f "fastlio_mappin[g]|livox_ros_driver2_nod[e]|global_localizatio[n]|transform_fusio[n]|map_serve[r]|costmap_rvi[z]|body_to_base_lin[k]" 2>/dev/null
echo "[stop_all] done"
