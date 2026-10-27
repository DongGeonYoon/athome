# Perception live test environment (sensors only: no controller, no /cmd_vel)
source /opt/ros/humble/setup.bash
source $HOME/fast_lio_ws/install/setup.bash
source $HOME/realsense_ros2_ws/install/setup.bash
source $HOME/semantic_geometry_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=25
# large same-host messages over shared memory (see ~/real/config/fastdds_shm.xml)
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/real/config/fastdds_shm.xml
export RMW_FASTRTPS_USE_QOS_FROM_XML=0
