cmake_minimum_required(VERSION 3.8)
project(semantic_geometry_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SemanticObject2D.msg"
  "msg/SemanticObject2DArray.msg"
  "msg/ObjectObservation3D.msg"
  "msg/ObjectObservation3DArray.msg"
  "msg/SearchTargets.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
