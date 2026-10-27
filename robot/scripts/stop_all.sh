#!/usr/bin/env bash
# Stop everything started from ~/real/scripts (perception, camera, localization, RViz, rqt).
bash $HOME/real/scripts/stop_experiment.sh
pkill -INT -f "v4l2_rgbd_nod[e]"; pkill -INT -f "slam_costmap_bringup.launch.p[y]"; pkill -INT -f "rqt_image_vie[w]"
sleep 5
pkill -9 -f "fastlio_mappin[g]|livox_ros_driver2_nod[e]|global_localizatio[n]|transform_fusio[n]|map_serve[r]|costmap_rvi[z]|body_to_base_lin[k]" 2>/dev/null
echo "[stop_all] done"
