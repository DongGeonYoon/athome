#!/usr/bin/env bash
# On the GPU server: (re)start cgd_server.py (CG-D service for cgd_live_node --server) in the background.
#   bash /data/athome/server_sync/start_cgd_server.sh [PORT=47000]   log: /data/athome/live_search/server_logs/cgd_server.log
#   ... ensure   start only if it is not running (live search: keep a running server and its loaded models)
A=/data/athome; L=$A/live_search/server_logs; mkdir -p $L
if [ "$1" = ensure ]; then
  pgrep -f "tools/robot_live/[c]gd_server.py" >/dev/null && { echo "cgd_server already running"; exit 0; }; shift
fi
for p in $(pgrep -f "tools/robot_live/[c]gd_server.py"); do kill $p; done; sleep 2
source $A/activate.sh
setsid nohup python $A/code/semantic_geometry_ws/tools/robot_live/cgd_server.py --port ${1:-47000} > $L/cgd_server.log 2>&1 < /dev/null &
echo "cgd_server starting (pid $!), log $L/cgd_server.log"
