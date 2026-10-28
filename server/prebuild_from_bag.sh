#!/usr/bin/env bash
# Pre-mapping on the GPU server: recorded bag -> keyframes -> CG-D features -> Scene Graph dataset (objects.json).
#   bash /data/athome/server_sync/prebuild_from_bag.sh RUN [robot_bag_extract options]
# Input : /data/athome/prebuild/robot_mapping/RUN/bag   (sent by the robot PC's premap_to_server.sh)
# Output: .../RUN/offline/robot/sources/RUN/  keyframes (results/frame*.jpg, traj.txt, lidar/, depth if recorded) + CG-D
#         .../RUN/offline/sg_VERSION/scenes/robot_RUN/objects.json (+ points/, occupancy/, metadata.json)
#         delivery, same layout as the packages sent so far (search server picks it up, require4 6):
#           /data/athome/scene_graphs/RUN/VERSION/
#             scene_graph_dataset/  index.json, schema/, scenes/robot_RUN/ (objects.json, metadata.json, points/, ...)
#             nav2_map/             occupancy grid of the map used (yaml + image) + SHA256SUMS
#             pcd_map/              localization PCD map of the same map + SHA256SUMS
#             manifest.json, README.md
#           /data/athome/scene_graphs/RUN/RUN_VERSION.zip (+ .sha256) to send; RUN/latest -> newest VERSION
#         VERSION = build time YYYYmmdd_HHMMSS; metadata.json occupancy.map_version.map_version and manifest map_version =
#         "sha256:<16 hex>" of the Nav2 map (yaml + image), the search server's check.
#         log: .../RUN/offline/prebuild.log
# Steps (same code and settings as the robot PC's offline rebuild, run4 "offline5"):
#   1 robot_bag_extract.py (ROS env): keyframes while still / slow (|v| <= 0.5 m/s, |w| <= 0.6 rad/s,
#     every 0.3 m / 5 deg / 0.5 s), FAST-LIO body cloud 0.5 s, localization TF, floor z -0.41 (lab map)
#   2 run_cgd_sources.sh (conceptgraph env, GPU): CG-D v2 vocabulary on every keyframe
#   3 build_dataset.py: fusion (association 1.1, stray points removed, fragment merge 0.9), Nav2 map as occupancy
# MAP=<nav2 map yaml> overrides the lab map (/data/athome/prebuild/maps); PCD=<pcd> overrides its PCD map
# (default: same name without "_structural", .pcd, next to the yaml).
set -e
RUN=${1:?usage: prebuild_from_bag.sh RUN [extract options]}; shift
A=/data/athome; R=$A/prebuild/robot_mapping/$RUN; O=$R/offline; S=$O/robot/sources/$RUN
MAP=${MAP:-$A/prebuild/maps/lab_map_20261005_174210_structural.yaml}
PCD=${PCD:-$(echo $MAP | sed 's/_structural\.yaml$/.pcd/')}
[ -f $PCD ] || { echo "no PCD map: $PCD (set PCD=...)"; exit 1; }
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
SG=$O/sg_$VERSION; D=$A/scene_graphs/$RUN/$VERSION
mkdir -p $D/scene_graph_dataset $D/nav2_map $D/pcd_map
cp -r $SG/index.json $SG/schema $SG/scenes $D/scene_graph_dataset/
IMG=$(dirname $MAP)/$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$MAP'))['image'])" 2>/dev/null || grep -oP '^image:\s*\K\S+' $MAP)
cp $MAP $IMG $D/nav2_map/ && (cd $D/nav2_map && sha256sum * > SHA256SUMS)
cp $PCD $D/pcd_map/ && (cd $D/pcd_map && sha256sum *.pcd > SHA256SUMS)
python3 - "$D" "$RUN" "$VERSION" <<'PY'
import hashlib, json, os, sys, zipfile
d, run, ver = sys.argv[1:]
sc = os.path.join(d, 'scene_graph_dataset', 'scenes', f'robot_{run}')
m = json.load(open(os.path.join(sc, 'metadata.json')))
mv = (m.get('occupancy') or {}).get('map_version') or {}
n = len(json.load(open(os.path.join(sc, 'objects.json')))['objects'])
sums = {}
for sub in ('nav2_map', 'pcd_map'):
    for l in open(os.path.join(d, sub, 'SHA256SUMS')):
        h, f = l.split(); sums[f'{sub}/{f}'] = h
json.dump(dict(run=run, version=ver, objects=n, frame_id='map', map_version=mv.get('map_version'), files_sha256=sums,
               scene=f'scene_graph_dataset/scenes/robot_{run}'), open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
open(os.path.join(d, 'README.md'), 'w').write(f"""# 사전 구축 Scene Graph 데이터셋 — {run} ({ver})

| 폴더 / 파일 | 내용 |
|---|---|
| `scene_graph_dataset/` | 지금까지와 같은 데이터셋 구조: `index.json`, `schema/objects.schema.json`, `scenes/robot_{run}/` (`objects.json` {n}개, `objects_all.json`, `metadata.json`, `points/`, `occupancy/`) |
| `nav2_map/` | 이 데이터셋의 map 좌표계 2D occupancy grid (map_server yaml + 이미지), `SHA256SUMS` |
| `pcd_map/` | 같은 지도의 위치추정용 PCD 지도, `SHA256SUMS` |
| `manifest.json` | 녹화 이름, 버전, 물체 수, 지도 버전 `{mv.get('map_version')}`, 파일 해시 |

- 모든 좌표는 `map` 좌표계입니다. 지도 버전은 `"sha256:" + sha256(지도 yaml + 이미지)` 앞 16자리입니다 (`metadata.json`의 `occupancy.map_version.map_version`과 같음).
- `scenes/robot_{run}/occupancy/pcd_map.pcd`는 이 녹화의 LiDAR로 만든 점 지도이고, `pcd_map/`은 위치추정에 쓰는 원래 지도입니다.
""")
z = os.path.join(os.path.dirname(d), f'{run}_{ver}.zip')
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
    for root, _, files in os.walk(d):
        for x in files:
            p = os.path.join(root, x); f.write(p, os.path.join(f'{run}_{ver}', os.path.relpath(p, d)))
open(z + '.sha256', 'w').write(hashlib.sha256(open(z, 'rb').read()).hexdigest() + f'  {os.path.basename(z)}\n')
PY
ln -sfn $VERSION $A/scene_graphs/$RUN/latest
echo "[prebuild] done $(date '+%F %T'): $(python3 -c "import json;print(json.load(open('$D/manifest.json'))['objects'])") objects, version $VERSION -> $D (+ $A/scene_graphs/$RUN/${RUN}_$VERSION.zip)"
