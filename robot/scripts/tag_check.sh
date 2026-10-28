#!/usr/bin/env bash
# Tag check: are objects labelled right? Camera -> GPU server (RAM + DINO only, ~1 s per image) -> rqt.
#   bash tag_check.sh          start (camera if none is publishing, server, tunnel, node, rqt) on ROS_DOMAIN_ID 25
#   bash tag_check.sh stop     stop what this script started (a camera it did not start keeps running) + the server process
# Camera started here (when no camera topic exists): D435 on USB 2.0 via the V4L2 node, colour 1280x720 @ 10 fps.
P=$HOME/real/results/tag_check; PIDS=$P/pids
source $HOME/real/config/gpu_server.env       # ATHOME_GPU_HOST / _PORT / _KEY
SSH="ssh -o BatchMode=yes -i $ATHOME_GPU_KEY -p $ATHOME_GPU_PORT $ATHOME_GPU_HOST"
source $HOME/real/scripts/env.sh; export DISPLAY=${DISPLAY:-:0}
if [ "$1" = stop ]; then
  [ -f $PIDS ] && for p in $(cat $PIDS); do kill -INT $p 2>/dev/null; done; sleep 4
  [ -f $PIDS ] && for p in $(cat $PIDS); do kill $p 2>/dev/null; done
  pkill -9 -f "^python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view /tag_check/annotated$"
  $SSH "bash /data/athome/server_sync/start_tag_check_server.sh stop" 2>/dev/null; rm -f $PIDS; echo "tag check stopped"; exit 0
fi
mkdir -p $P/logs; : > $PIDS
$SSH "bash /data/athome/server_sync/start_tag_check_server.sh" 2>/dev/null
if ! timeout 5 ros2 topic list 2>/dev/null | grep -q "^/camera/camera/color/image_raw$"; then
  setsid bash $HOME/real/scripts/start_camera_v4l2.sh 1280 720 10 --no-depth > $P/logs/camera.log 2>&1 < /dev/null & echo $! >> $PIDS
fi
ss -ltn | grep -q "127.0.0.1:47001" || { setsid ssh -N -o BatchMode=yes -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
  -i $ATHOME_GPU_KEY -p $ATHOME_GPU_PORT -L 47001:127.0.0.1:47001 $ATHOME_GPU_HOST > $P/logs/tunnel.log 2>&1 < /dev/null & echo $! >> $PIDS; }
until $SSH "grep -q listening /data/athome/live_search/server_logs/tag_check_server.log" 2>/dev/null; do sleep 3; done
setsid python3 $HOME/semantic_geometry_ws/tools/robot_live/tag_check_node.py --save-dir $P > $P/logs/node.log 2>&1 < /dev/null & echo $! >> $PIDS
setsid ros2 run rqt_image_view rqt_image_view /tag_check/annotated > $P/logs/rqt.log 2>&1 < /dev/null & echo $! >> $PIDS
echo "[tag_check] running: rqt shows /tag_check/annotated, log $P/logs/node.log"
