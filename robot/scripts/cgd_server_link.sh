#!/usr/bin/env bash
# Link to the GPU server's CG-D service for live use (run_search.sh CGD=server, remote_view.sh):
#   bash cgd_server_link.sh            make sure cgd_server.py runs on the server (keeps a running one), then keep an
#                                      SSH tunnel 127.0.0.1:47000 -> server open, reopening it whenever it drops
#                                      (internet / DNS hiccup, server restart). Runs until killed.
source $HOME/real/config/gpu_server.env       # ATHOME_GPU_HOST / _PORT / _KEY
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -i $ATHOME_GPU_KEY -p $ATHOME_GPU_PORT"
H=$ATHOME_GPU_HOST
until $SSH $H "bash /data/athome/server_sync/start_cgd_server.sh ensure" 2>/dev/null; do echo "[link] server not reachable, retrying"; sleep 5; done
until $SSH $H "grep -q listening /data/athome/live_search/server_logs/cgd_server.log" 2>/dev/null; do sleep 3; done
echo "[link] cgd_server ready"
while true; do
  ss -ltn | grep -q "127.0.0.1:47000 " || $SSH -N -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L 47000:127.0.0.1:47000 $H 2>&1 | grep -v setlocale
  echo "[link] tunnel closed $(date +%T), reopening"; sleep 2
done
