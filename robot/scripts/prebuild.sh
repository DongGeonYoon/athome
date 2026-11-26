#!/usr/bin/env bash
# Pre-mapping on a GPU PC: recorded bag -> keyframes -> CG-D -> Scene Graph dataset -> delivery package.
#   bash prebuild.sh NAME [robot_bag_extract options]          NAME = folder under ~/real/results (record_premap.sh)
# Map: NAME/map.env (MAP_PCD, MAP_YAML written by record_premap.sh), or MAP_PCD= / MAP_YAML= in the environment.
# FLOOR_Z=<z in map>: floor height; default the extractor's estimate (median base_link z). Lab map: -0.41.
# Output: ~/real/results/NAME/
#   offline/robot/sources/NAME/   keyframes (results/, traj.txt, lidar/, map_points.npz) + CG-D detections
#   offline/sg_VERSION/           dataset as built
#   scene_graph/VERSION/          package: scene_graph_dataset/, nav2_map/, pcd_map/ (+ SHA256SUMS), manifest.json, README.md
#   scene_graph/NAME_VERSION.zip (+ .sha256)
# Step 1 runs in the ROS environment, steps 2-3 in the conceptgraph environment: separate shells (the ROS environment puts its
# own OpenCV on PYTHONPATH, which breaks the conceptgraph python).
set -e
NAME=${1:?usage: prebuild.sh NAME [extract options]}; shift
R=$HOME/real/results/$NAME; O=$R/offline; S=$O/robot/sources/$NAME
T=$(dirname $(readlink -f $0))/../../semantic_geometry_ws/tools
PY=${CGD_PYTHON:-$HOME/miniforge3/envs/conceptgraph/bin/python}
[ -f $R/map.env ] && [ -z "$MAP_YAML" ] && source $R/map.env
: ${MAP_YAML:?no map: NAME/map.env or MAP_YAML=}; : ${MAP_PCD:?no map: NAME/map.env or MAP_PCD=}
[ -d $R/bag ] || { echo "no bag: $R/bag"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/prebuild.log) 2>&1
echo "[prebuild] $NAME $(date '+%F %T'), map $(basename $MAP_YAML) + $(basename $MAP_PCD)"
if [ ! -f $S/source.json ]; then
  env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "source /opt/ros/humble/setup.bash && cd $T/sim_sources && python3 robot_bag_extract.py $R/bag $S \
    --name $NAME --max-speed 0.5 --max-turn 0.6 --kf-dist 0.3 --kf-yaw 5 --kf-gap 0.5 $*"
  [ -n "$FLOOR_Z" ] && python3 -c "import json;p='$S/source.json';s=json.load(open(p));s['floor_z_base_link_median']=s['floor_z'];\
s['floor_z']=float('$FLOOR_Z');json.dump(s,open(p,'w'),indent=1)"
fi
echo "[prebuild] keyframes: $(ls $S/results/frame*.jpg | wc -l)"
VERSION=$(date +%Y%m%d_%H%M%S)
env -i HOME=$HOME PATH=/usr/bin:/bin CGD_PYTHON=$PY bash -c "bash $T/sim_sources/run_cgd_sources.sh $O/robot/sources v2 \
  && cd $T/scene_graph_dataset && $PY build_dataset.py --exp $O --sources robot --out-name sg_$VERSION --occupancy-map $MAP_YAML"
SG=$O/sg_$VERSION; D=$R/scene_graph/$VERSION
mkdir -p $D/scene_graph_dataset $D/nav2_map $D/pcd_map
cp -r $SG/index.json $SG/schema $SG/scenes $D/scene_graph_dataset/
IMG=$(dirname $MAP_YAML)/$(python3 -c "import yaml;print(yaml.safe_load(open('$MAP_YAML'))['image'])")
cp $MAP_YAML $IMG $D/nav2_map/ && (cd $D/nav2_map && sha256sum * > SHA256SUMS)
cp $MAP_PCD $D/pcd_map/ && (cd $D/pcd_map && sha256sum *.pcd > SHA256SUMS)
python3 - "$D" "$NAME" "$VERSION" <<'PY'
import hashlib, json, os, sys, zipfile
d, run, ver = sys.argv[1:]
sc = os.path.join(d, 'scene_graph_dataset', 'scenes', f'robot_{run}')
mv = (json.load(open(os.path.join(sc, 'metadata.json'))).get('occupancy') or {}).get('map_version') or {}
n = len(json.load(open(os.path.join(sc, 'objects.json')))['objects'])
sums = {f'{sub}/{f}': h for sub in ('nav2_map', 'pcd_map') for h, f in (l.split() for l in open(os.path.join(d, sub, 'SHA256SUMS')))}
json.dump(dict(run=run, version=ver, objects=n, frame_id='map', map_version=mv.get('map_version'), files_sha256=sums,
               scene=f'scene_graph_dataset/scenes/robot_{run}'), open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
open(os.path.join(d, 'README.md'), 'w').write(f"""# 사전 구축 Scene Graph 데이터셋 — {run} ({ver})

| 폴더 / 파일 | 내용 |
|---|---|
| `scene_graph_dataset/` | `index.json`, `schema/objects.schema.json`, `scenes/robot_{run}/` (`objects.json` {n}개, `objects_all.json`, `metadata.json`, `points/`, `occupancy/`) |
| `nav2_map/` | 이 데이터셋의 map 좌표계 2D occupancy grid (map_server yaml + 이미지), `SHA256SUMS` |
| `pcd_map/` | 같은 지도의 위치추정용 PCD 지도, `SHA256SUMS` |
| `manifest.json` | 녹화 이름, 버전, 물체 수, 지도 버전 `{mv.get('map_version')}`, 파일 해시 |

- 모든 좌표는 `map` 좌표계입니다. 지도 버전은 `"sha256:" + sha256(지도 yaml + 이미지)` 앞 16자리입니다.
- `scenes/robot_{run}/occupancy/pcd_map.pcd`는 이 녹화의 LiDAR로 만든 점 지도이고, `pcd_map/`은 위치추정에 쓰는 원래 지도입니다.
""")
z = os.path.join(os.path.dirname(d), f'{run}_{ver}.zip')
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
    for root, _, files in os.walk(d):
        for x in files:
            p = os.path.join(root, x); f.write(p, os.path.join(f'{run}_{ver}', os.path.relpath(p, d)))
open(z + '.sha256', 'w').write(hashlib.sha256(open(z, 'rb').read()).hexdigest() + f'  {os.path.basename(z)}\n')
print(f'[prebuild] done: {n} objects, version {ver} -> {d} (+ {os.path.basename(z)})')
PY
