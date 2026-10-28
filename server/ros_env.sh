# ROS 2 Humble on the GPU server (conda env ros_humble, RoboStack; Ubuntu 20.04 has no apt Humble), same packages and
# settings as the robot PC's ~/real/scripts/env.sh:  source /data/athome/server_sync/ros_env.sh
#   ros nodes (geometry_fusion_node, bag extraction):  python of ros_humble
#   CG-D node with ROS (cgd_live_node):  CGD_PY  (conceptgraph python 3.10 + this ROS on its path)
A=/data/athome
export HOME=$A XDG_CACHE_HOME=$A/cache TMPDIR=$A/tmp ROS_LOG_DIR=$A/logs/ros
source $A/miniforge3/etc/profile.d/conda.sh && conda activate ros_humble
WS=$A/code/semantic_geometry_ws
[ -f $WS/install/setup.bash ] && source $WS/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-25} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export FASTRTPS_DEFAULT_PROFILES_FILE=$A/code/config/fastdds_shm.xml RMW_FASTRTPS_USE_QOS_FROM_XML=0
export GSA_PATH=$A/conceptgraph_ws/Grounded-Segment-Anything
# OpenCV in this env needs libGL.so.1 (no GUI libs in the container; conda libgl conflicts with the env pins):
# only the three GL libraries, linked from the conceptgraph env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$A/lib_gl
export CGD_PY="env PYTHONPATH=$PYTHONPATH:$CONDA_PREFIX/lib/python3.10/site-packages LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH $A/miniforge3/envs/conceptgraph/bin/python"
mkdir -p $TMPDIR $ROS_LOG_DIR
