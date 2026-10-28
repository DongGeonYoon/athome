#!/usr/bin/env bash
# On the GPU server: (re)start tag_check_server.py (labels only: RAM + DINO) in the background.
#   bash /data/athome/server_sync/start_tag_check_server.sh [PORT=47001]      stop: ... stop
A=/data/athome; L=$A/live_search/server_logs; mkdir -p $L
for p in $(pgrep -f "tools/robot_live/[t]ag_check_server.py"); do kill $p; done
[ "$1" = stop ] && { echo "tag_check_server stopped"; exit 0; }
sleep 2; source $A/activate.sh
setsid nohup python $A/code/semantic_geometry_ws/tools/robot_live/tag_check_server.py --port ${1:-47001} > $L/tag_check_server.log 2>&1 < /dev/null &
echo "tag_check_server starting (pid $!), log $L/tag_check_server.log"
