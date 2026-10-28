#!/usr/bin/env bash
# Pre-mapping on the GPU server: recorded bag -> keyframes -> CG-D features -> Scene Graph dataset (objects.json).
#   bash /data/athome/server_sync/prebuild_from_bag.sh RUN [robot_bag_extract options]
# Input : /data/athome/prebuild/robot_mapping/RUN/bag   (sent by the robot PC's premap_to_server.sh)
# Output: .../RUN/offline/robot/sources/RUN/  keyframes (results/frame*.jpg, traj.txt, lidar/, depth if recorded) + CG-D
#         .../RUN/offline/sg_VERSION/scenes/robot_RUN/objects.json (+ points/, occupancy/, metadata.json)
#         delivery (search server picks it up, require4 6): /data/athome/scene_graphs/RUN/VERSION/ (+ manifest.json),
#         /data/athome/scene_graphs/RUN/latest -> newest VERSION. VERSION = build time YYYYmmdd_HHMMSS; metadata.json
#         occupancy.map_version.map_version = "sha256:<16 hex>" of the Nav2 map (yaml + image), same as the server's check.
#         log: .../RUN/offline/prebuild.log
# Steps (same code and settings as the robot PC's offline rebuild, run4 "offline5"):
#   1 robot_bag_extract.py (ROS env): keyframes while still / slow (|v| <= 0.5 m/s, |w| <= 0.6 rad/s,
#     every 0.3 m / 5 deg / 0.5 s), FAST-LIO body cloud 0.5 s, localization TF, floor z -0.41 (lab map)
#   2 run_cgd_sources.sh (conceptgraph env, GPU): CG-D v2 vocabulary on every keyframe
#   3 build_dataset.py: fusion (association 1.1, stray points removed, fragment merge 0.9), Nav2 map as occupancy
# MAP=<nav2 map yaml> overrides the lab map (/data/athome/prebuild/maps).
set -e
RUN=${1:?usage: prebuild_from_bag.sh RUN [extract options]}; shift
A=/data/athome; R=$A/prebuild/robot_mapping/$RUN; O=$R/offline; S=$O/robot/sources/$RUN
MAP=${MAP:-$A/prebuild/maps/lab_map_20261005_174210_structural.yaml}
[ -d $R/bag ] || { echo "no bag: $R/bag"; exit 1; }
# the GPU slice (9.5 GB) holds one CG-D at a time: live services must be stopped first
if pgrep -f "tools/robot_live/[c]gd_server.py|tools/robot_live/[t]ag_check_server.py" >/dev/null; then
  echo "GPU in use by cgd_server / tag_check_server: stop them first (start_cgd_server.sh: kill; start_tag_check_server.sh stop)"; exit 1
fi
mkdir -p $O; exec > >(tee -a $O/prebuild.log) 2>&1
echo "[prebuild] $RUN $(date '+%F %T')"
WST=$A/code/semantic_geometry_ws/tools
# step 1 in the ROS env, steps 2-3 in the conceptgraph env: separate shells (the ROS env puts its own OpenCV on
# PYTHONPATH, which breaks the conceptgraph python)
if [ ! -f $S/source.json ]; then
  bash -c "source $A/server_sync/ros_env.sh && cd $WST/sim_sources && python robot_bag_extract.py $R/bag $S --name $RUN \
    --max-speed 0.5 --max-turn 0.6 --kf-dist 0.3 --kf-yaw 5 --kf-gap 0.5 $* && python -c \"import json;p='$S/source.json';\
s=json.load(open(p));s['floor_z_base_link_median']=s['floor_z'];s['floor_z']=-0.41;json.dump(s,open(p,'w'),indent=1)\""
fi
echo "[prebuild] keyframes: $(ls $S/results/frame*.jpg | wc -l)"
VERSION=$(date +%Y%m%d_%H%M%S)
env -i HOME=$A PATH=/usr/bin:/bin bash -c "source $A/activate.sh >/dev/null && bash $WST/sim_sources/run_cgd_sources.sh $O/robot/sources v2 \
  && python $WST/scene_graph_dataset/build_dataset.py --exp $O --sources robot --out-name sg_$VERSION --occupancy-map $MAP"
SC=$O/sg_$VERSION/scenes/robot_$RUN; D=$A/scene_graphs/$RUN/$VERSION
mkdir -p $D && cp -r $SC/. $D/ && ln -sfn $VERSION $A/scene_graphs/$RUN/latest
python3 - "$D" "$RUN" "$VERSION" <<'PY'
import json, os, sys
d, run, ver = sys.argv[1:]; m = json.load(open(os.path.join(d, 'metadata.json')))
mv = (m.get('occupancy') or {}).get('map_version') or {}
json.dump(dict(run=run, version=ver, objects=len(json.load(open(os.path.join(d, 'objects.json')))['objects']),
               frame_id='map', map_version=mv.get('map_version'), map_files=mv.get('sha256'),
               files=sorted(os.listdir(d))), open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
PY
echo "[prebuild] done $(date '+%F %T'): $(python3 -c "import json;print(json.load(open('$D/manifest.json'))['objects'])") objects, version $VERSION -> $D"
