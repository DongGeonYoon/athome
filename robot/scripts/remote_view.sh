#!/usr/bin/env bash
# See what the GPU server recognises: camera images (live camera or a bag) -> cgd_live_node --server (this PC, relays
# to cgd_server.py on the GPU server through the SSH tunnel) -> /cg_d/annotated -> rqt_image_view. ROS_DOMAIN_ID 77.
#   bash remote_view.sh [BAG=~/real/results/run4/bag]   ("live" = camera topic instead of a bag)
#   stop: bash remote_view.sh stop          DOMAIN=25 bash remote_view.sh live  -> use a camera already running on domain 25
# Needs on the server: cgd_server.py running (README "live search"), here: tunnel 127.0.0.1:47000 -> server.
P=$HOME/real/results/remote_view; PIDS=$P/pids
if [ "$1" = stop ]; then [ -f $PIDS ] && kill $(cat $PIDS) 2>/dev/null; pkill -9 -f "^python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view /cg_d/annotated$"; rm -f $PIDS; echo stopped; exit 0; fi
SRC=${1:-$HOME/real/results/run4/bag}
source $HOME/real/config/gpu_server.env       # ATHOME_GPU_HOST / _PORT / _KEY
source $HOME/real/scripts/env.sh; export ROS_DOMAIN_ID=${DOMAIN:-77} DISPLAY=${DISPLAY:-:0}
mkdir -p $P/logs
ss -ltn | grep -q "127.0.0.1:47000" || setsid ssh -N -o BatchMode=yes -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
  -i $ATHOME_GPU_KEY -p $ATHOME_GPU_PORT -L 47000:127.0.0.1:47000 $ATHOME_GPU_HOST > $P/logs/tunnel.log 2>&1 < /dev/null &
sleep 2
TOPIC=/camera/camera/color/image_raw/compressed; [ "$SRC" = live ] && TOPIC=/camera/camera/color/image_raw
setsid python3 $HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py --server 127.0.0.1:47000 --image-topic $TOPIC \
  --save-dir $P > $P/logs/relay.log 2>&1 < /dev/null & echo $! > $PIDS
setsid ros2 run rqt_image_view rqt_image_view /cg_d/annotated > $P/logs/rqt.log 2>&1 < /dev/null & echo $! >> $PIDS
if [ "$SRC" != live ]; then
  setsid ros2 bag play "$SRC" -l --topics $TOPIC /camera/camera/color/camera_info > $P/logs/play.log 2>&1 < /dev/null & echo $! >> $PIDS
fi
echo "[remote_view] running: rqt shows /cg_d/annotated (one frame every ~5 s), records in $P"
