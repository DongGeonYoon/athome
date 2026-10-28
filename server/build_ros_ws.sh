#!/usr/bin/env bash
# Build the perception ROS packages on the GPU server (conda ros_humble env). Two RoboStack quirks handled here:
#   empy must be 3.3.4 (4.x breaks rosidl: "module 'em' has no attribute 'BUFFERED_OPT'")
#   conda gcc 15 vs the Humble headers: '<cstdint>' is not pulled in -> -include cstdint
#   bash /data/athome/server_sync/build_ros_ws.sh
source /data/athome/server_sync/ros_env.sh
python -c "import em, sys; sys.exit(0 if em.__version__.startswith('3.') else 1)" || \
  CONDA_PKGS_DIRS=/data/athome/cache/conda_pkgs conda install -y -q -c conda-forge "empy=3.3.4"
cd $WS && colcon build --packages-select athome_interfaces semantic_geometry_msgs semantic_geometry_fusion \
  --cmake-args -DCMAKE_CXX_FLAGS="-include cstdint" -DPython3_EXECUTABLE=$(which python)
