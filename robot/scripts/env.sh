# Perception live test environment (sensors only: no controller, no /cmd_vel)
source /opt/ros/humble/setup.bash
source $HOME/fast_lio_ws/install/setup.bash
source $HOME/realsense_ros2_ws/install/setup.bash
source $HOME/semantic_geometry_ws/install/setup.bash
export ROS_DOMAIN_ID=25
# Middleware: rmw_zenoh_cpp for every node (team decision, fast_lio_ws/src/mppi_nav/README_COSTMAP_INTERFACE.md 0;
# require4 6). Each PC needs one zenoh router (tcp/[::]:7447): started here when none is listening, connected to
# ATHOME_ZENOH_ENDPOINT (e.g. tcp/<navigation laptop IP>:7447) when set. Log: ~/real/results/zenohd.log
# ATHOME_RMW=fastdds restores FastDDS with the shared-memory profile (before 2026-10-28).
if [ "${ATHOME_RMW:-zenoh}" = zenoh ]; then
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  if ! ss -ltn 2>/dev/null | grep -q ':7447 '; then
    ( [ -n "$ATHOME_ZENOH_ENDPOINT" ] && export ZENOH_CONFIG_OVERRIDE="connect/endpoints=[\"$ATHOME_ZENOH_ENDPOINT\"]"
      setsid ros2 run rmw_zenoh_cpp rmw_zenohd > $HOME/real/results/zenohd.log 2>&1 < /dev/null & )
    for _ in $(seq 20); do ss -ltn 2>/dev/null | grep -q ':7447 ' && break; sleep 0.25; done
  fi
else
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  # large same-host messages over shared memory (see ~/real/config/fastdds_shm.xml)
  export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/real/config/fastdds_shm.xml
  export RMW_FASTRTPS_USE_QOS_FROM_XML=0
fi
