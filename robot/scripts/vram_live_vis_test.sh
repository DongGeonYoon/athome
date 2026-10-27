#!/usr/bin/env bash
# Online GPU test with visualisation running: RViz (image of /cg_d/annotated) + rqt_image_view next to the live
# CG-D node (vram_live_test.sh), all on ROS_DOMAIN_ID 77.
#   bash vram_live_vis_test.sh NAME [nvidia]   nvidia = render the GUIs on the NVIDIA GPU (PRIME offload, worst case)
T=$HOME/real/results/run4/vram_test; NAME=$1
source $HOME/real/scripts/env.sh; export ROS_DOMAIN_ID=77 DISPLAY=${DISPLAY:-:0}
[ "$2" = nvidia ] && export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
rviz2 -d $HOME/real/config/vram_vis_test.rviz > $T/vis_${NAME}_rviz.log 2>&1 & RV=$!
ros2 run rqt_image_view rqt_image_view /cg_d/annotated > $T/vis_${NAME}_rqt.log 2>&1 & RQ=$!
sleep 8
unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME          # CG-D itself as usual
( sleep 60; nvidia-smi > $T/vis_${NAME}_nvidia_smi.txt ) &
bash $HOME/real/scripts/vram_live_test.sh vis_$NAME
kill $RV $RQ 2>/dev/null; pkill -9 -f "^python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view /cg_d/annotated$"   # ros2 run leaves the child, which ignores SIGTERM
