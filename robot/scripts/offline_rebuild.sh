#!/usr/bin/env bash
# Offline Scene Graph features from a recorded run (the Scene Graph is built offline, so use every keyframe of the bag):
#   bash offline_rebuild.sh RUN_NAME [extractor options]   -> ~/real/results/<RUN>/offline/
# 1 bag -> keyframes while still or slowly moving (|v| <= 0.5 m/s, |w| <= 0.6 rad/s, every 0.3 m / 15 deg / 0.5 s)
# 2 CG-D v2 on every keyframe (~4.5 s each)  3 fusion with the dataset builder (fragment merge 0.9)
# 4 real geometry_fusion_node on the same bag (ros_replay.sh) and comparison  5 checks + BEV against the live result
set -e
R=$HOME/real/results/$1; O=$R/offline; shift
source $HOME/real/scripts/env.sh
WS=$HOME/semantic_geometry_ws; PY=$HOME/miniforge3/envs/conceptgraph/bin/python; RUN=$(basename $R)
mkdir -p $O/robot/sources
if [ ! -f $O/robot/sources/$RUN/source.json ]; then
  (cd $WS/tools/sim_sources && python3 robot_bag_extract.py $R/bag $O/robot/sources/$RUN --name $RUN --max-speed 0.5 --max-turn 0.6 \
     --kf-dist 0.3 --kf-yaw 15 --kf-gap 0.5 "$@")
  python3 -c "import json;p='$O/robot/sources/$RUN/source.json';s=json.load(open(p));s['floor_z_base_link_median']=s['floor_z'];s['floor_z']=-0.41;json.dump(s,open(p,'w'),indent=1)"
fi
bash $WS/tools/sim_sources/run_cgd_sources.sh $O/robot/sources v2
$PY $WS/tools/scene_graph_dataset/build_dataset.py --exp $O --sources robot --out-name sg_final
mkdir -p $O/ros_replay
BUFFER_SEC=120.0 bash $WS/tools/sim_sources/ros_replay.sh $R/bag $O/robot/sources/$RUN $O/ros_replay/objects_ros.json 1 v2 > $O/ros_replay/run.log 2>&1
$PY $WS/tools/sim_sources/compare_ros_offline.py $O/ros_replay/objects_ros.json $O/sg_final/scenes/robot_$RUN $O/robot/sources/$RUN | head -8
python3 $HOME/real/scripts/compare_sg.py live=$R/objects.json offline=$O/sg_final/scenes/robot_$RUN/objects.json --bev $O/bev_offline.png
