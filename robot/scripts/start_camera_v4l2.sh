#!/usr/bin/env bash
# D435 colour via V4L2 (works on USB 2.0), same topics as realsense2_camera. Default 1280x720 @ 10 fps.
#   bash start_camera_v4l2.sh [WIDTH=1280] [HEIGHT=720] [FPS=10] [--no-depth]   (aligned depth only at 640x480)
source $HOME/real/scripts/env.sh
exec python3 $(dirname $(readlink -f $0))/v4l2_rgbd_node.py --width ${1:-1280} --height ${2:-720} --fps ${3:-10} "${@:4}"
