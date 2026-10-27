#!/usr/bin/env bash
# Offline Scene Graph features from a real-robot pre-mapping rosbag (the Scene Graph is built before navigation,
# not while driving). Same tools and parameters as the simulation dataset:
#   1. robot_bag_extract.py   bag -> keyframes (image, aligned depth, 0.5 s LiDAR body cloud, pose) + map points
#   2. run_cgd_sources.sh     CG-D with the v2 vocabulary on every keyframe (GPU, ~4-6 s per frame, resumable)
#   3. build_dataset.py       fusion -> ~/exp1/scene_graph_dataset/scenes/robot_<NAME>/objects.json (+ points, occupancy)
#
#   bash offline_scene_graph.sh BAG_DIR NAME [extra robot_bag_extract.py options, e.g. --image-topic ...]
# Optional check with the real node on the same bag: ros_replay.sh BAG_DIR ~/exp1/robot/sources/NAME OUT.json
set -eo pipefail
BAG=$1; NAME=$2; shift 2
EXP=$HOME/exp1; WS=$HOME/semantic_geometry_ws; PY=$HOME/miniforge3/envs/conceptgraph/bin/python
SRC=$EXP/robot/sources/$NAME
source /opt/ros/humble/setup.bash
[ -f "$HOME/fast_lio_ws/install/setup.bash" ] && source "$HOME/fast_lio_ws/install/setup.bash"   # livox msg types
source "$WS/install/setup.bash"
if [ ! -f "$SRC/source.json" ]; then
  (cd "$WS/tools/sim_sources" && python3 robot_bag_extract.py "$BAG" "$SRC" --name "$NAME" "$@")
fi
bash "$WS/tools/sim_sources/run_cgd_sources.sh" "$EXP/robot/sources" v2
"$PY" "$WS/tools/scene_graph_dataset/build_dataset.py" --sources robot --scenes "robot_$NAME" --exp "$EXP"
echo "[offline_scene_graph] $EXP/scene_graph_dataset/scenes/robot_$NAME/objects.json"
