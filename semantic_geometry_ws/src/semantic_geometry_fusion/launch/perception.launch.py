"""
Perception fusion node on the real robot.

  mode:=mapping  pre-mapping run (or its rosbag replay): builds the static object map and writes the Scene Graph
                 JSON (periodically, on ~/save_scene_graph and on shutdown). The Scene Graph is built offline from it.
  mode:=online   navigation: publishes per-frame observations on /athome/perception/objects
                 (athome_interfaces/ObservedObjectArray) for Target Matching, no JSON. prior_objects_json:=<the
                 pre-mapped objects.json> gives observed objects their persistent object_id.

  ros2 launch semantic_geometry_fusion perception.launch.py mode:=mapping max_sync_dt_sec:=<measured> \
      scene_graph_json_path:=$HOME/scene_graph/objects.json [use_sim_time:=true for bag replay]
  ros2 launch semantic_geometry_fusion perception.launch.py mode:=online max_sync_dt_sec:=<measured> \
      prior_objects_json:=$HOME/scene_graph/objects.json
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(context):
    share = get_package_share_directory('semantic_geometry_fusion')
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in ('mapping', 'online'):
        raise RuntimeError(f"mode must be 'mapping' or 'online', got '{mode}'")
    json_path = LaunchConfiguration('scene_graph_json_path').perform(context) if mode == 'mapping' else ''
    if mode == 'mapping' and not json_path:
        raise RuntimeError('mode:=mapping needs scene_graph_json_path:=<file>')
    prior_path = LaunchConfiguration('prior_objects_json').perform(context) if mode == 'online' else ''
    if mode == 'online' and not prior_path:
        raise RuntimeError('mode:=online needs prior_objects_json:=<pre-mapped objects.json> (object_id)')
    return [Node(
        package='semantic_geometry_fusion', executable='geometry_fusion_node', name='geometry_fusion_node',
        output='screen',
        parameters=[os.path.join(share, 'config', 'fusion_params.yaml'), os.path.join(share, 'config', 'robot.yaml'),
                    {'max_sync_dt_sec': float(LaunchConfiguration('max_sync_dt_sec').perform(context)),
                     'scene_graph_json_path': json_path,
                     'prior_objects_json': prior_path,
                     'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='mapping', description='mapping | online'),
        DeclareLaunchArgument('max_sync_dt_sec', description='camera-LiDAR stamp tolerance measured on the robot'),
        DeclareLaunchArgument('scene_graph_json_path', default_value=''),
        DeclareLaunchArgument('prior_objects_json', default_value='', description='online: pre-mapped objects.json'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_node),
    ])
