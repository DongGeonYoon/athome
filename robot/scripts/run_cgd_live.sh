#!/usr/bin/env bash
# Live CG-D node (conceptgraph python + ROS). Extra args go to cgd_live_node.py, e.g. --image-topic ...
source $HOME/real/scripts/env.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 PYTHONHASHSEED=0
exec $HOME/miniforge3/envs/conceptgraph/bin/python $HOME/semantic_geometry_ws/tools/robot_live/cgd_live_node.py "$@"
