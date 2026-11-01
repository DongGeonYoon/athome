# athome — 실내 물체 탐색 로봇의 인지(Perception) 모듈: 사전 구축

AtHOME 프로젝트에서 인지를 맡은 코드입니다. 카메라 사진의 의미 정보(ConceptGraphs CG-D: RAM → GroundingDINO → SAM → CLIP)와 LiDAR(Livox Mid-360)의 3D 위치를 합쳐 물체 feature를 만듭니다.

- **사전 구축 (offline, 이 저장소의 중심):** 로봇으로 실내를 녹화한 rosbag을 GPU 서버에서 처리해 Scene Graph용 `objects.json`과 지도 파일 묶음을 만듭니다.
- **live search (online):** 탐색 중 사진마다 관측된 물체를 `/athome/perception/objects`로 발행합니다(아래 "사용: live search").

물체 feature: `F_j = {ID, label, CLIP(1024), 점구름, 3D 상자(center, size, yaw), 관측 수}`

> 2026-11-01 정리: 확인·시험·평가용 코드(tag 확인, 지연·VRAM·정밀도 시험, 탐색 평가, 시뮬레이션 수집기)는 뺐습니다. 정리 전 전체판은 태그 `full-20261028`에 있습니다.

## 처리 과정

```
[로봇 PC] record_premap.sh 또는 run_experiment.sh
   위치추정(FAST_LIO_LOCALIZATION2, PCD 지도) + D435 RGB 1280×720 + Mid-360 → rosbag
        │ premap_to_server.sh (bag + 지도 파일 전송, 처리 시작, 결과 받기)
        ▼
[GPU 서버] prebuild_from_bag.sh
   1 robot_bag_extract.py   keyframe 추출 (정지·저속 구간, 0.3 m / 5° / 0.5 s), LiDAR body 점군, map 자세
   2 run_cgd_sources.sh     CG-D (gsa_fast.py wrapper, 어휘 v2)
   3 build_dataset.py       LiDAR fusion → objects.json (+ Nav2 지도를 occupancy로)
   4 전달 묶음               scene_graph_dataset/ + nav2_map/ + pcd_map/ + manifest.json + zip/sha256
```

## 구성

```
semantic_geometry_ws/
  src/semantic_geometry_fusion/   fusion 라이브러리 + ROS 2 노드
      geometry.py                 DBSCAN, 튀는 점 제거, 3D 상자
      association.py              관측–물체 연결 점수 (φ_sem=(cos+1)/2 + φ_geo)
      feature_fusion.py           연결·융합·재병합 (사전 구축과 live 공통)
      scene_graph_export.py       objects.json 쓰기
      geometry_fusion_node.py     ROS 노드 (live / mapping), prior_map.py: 사전 지도 ID 연결
      config/                     fusion_params.yaml, robot.yaml(T_cam_body, 마스크 경계 3 px); launch/
  src/semantic_geometry_msgs/     2D/3D 관측 메시지
  src/athome_interfaces/          (포함 안 됨) 탐색 서버 팀 메시지 패키지 — 아래 설치 4
  tools/
    sim_sources/
      robot_bag_extract.py        ★ 사전 구축 1단계: bag → keyframe (gazebo_extract.py의 bag 읽기 함수 사용)
      run_cgd_sources.sh          ★ 2단계: keyframe 폴더에 CG-D 실행
      gazebo_extract.py           bag 읽기·자세 보간 (Gazebo 시뮬레이션 추출도 포함)
    replica_rig/
      gsa_fast.py                 ★ CG-D 실행 wrapper (CG-D 코드는 그대로, 8 GB GPU용 장치·정밀도 분할)
      vocab/                      어휘 v2: blocklist.txt(제외 태그), household.txt(추가 46개)
      rig_exp.py, render_rig.py, render_active_obs.py   build_dataset.py가 import (센서 상수, 시뮬레이션 장면)
    replica_feasibility/plyload.py   rig_exp.py가 import
    scene_graph_dataset/
      build_dataset.py            ★ 3단계: 데이터셋 빌더 (실제 로봇 + 시뮬레이션 소스), objects.schema.json
    robot_live/                   live search: cgd_live_node, cgd_engine, cgd_server / cgd_wire (GPU 서버 CG-D 서비스)
                                  v4l2_rgbd_node.py: USB 2.0 카메라 노드 (녹화에도 사용)
robot/
  scripts/                        로봇 PC 스크립트 (아래 표)
  config/                         calibration.yaml, fastdds_shm.xml, py_shim(위치추정 open3d import 우회),
                                  gpu_server.env.example (서버 접속 설정 틀)
server/                           GPU 서버: 환경 설치, 가중치, prebuild_from_bag.sh, live용 start_cgd_server.sh
third_party/                      CG-D에 적용한 patch 1건
```

### 로봇 PC 스크립트 (`robot/scripts/`)

| 스크립트 | 역할 |
|---|---|
| `env.sh` | ROS 환경, `ROS_DOMAIN_ID=25`, `rmw_zenoh_cpp`(라우터 자동 실행, `ATHOME_RMW=fastdds`로 되돌림) |
| `start_localization.sh` | Livox + FAST_LIO_LOCALIZATION2 + map_server + RViz. `MAP_PCD=`, `MAP_YAML=`로 지도 지정(없으면 launch 기본 lab 지도) |
| `start_camera_v4l2.sh` | D435 컬러를 V4L2로 직접 읽음 (USB 2.0, 1280×720 10 fps, depth 없음). `start_camera.sh`는 USB 3.0용 공식 드라이버 |
| `record_premap.sh` | **사전 구축 녹화**: 위치추정 + 카메라 + 시각 차이 확인 + rosbag. 쓴 지도를 `<이름>/map.env`에 기록 |
| `run_experiment.sh` | 녹화 + 실시간 CG-D·fusion을 함께 실행, rqt 검출 화면 자동 실행 |
| `recog_wait_monitor.py` | 녹화 중 확인 창: 멈춘 뒤 찍힌 사진이 CG-D로 몇 장 처리됐는지 → 다음 위치로 이동해도 되는지 표시 |
| `heading_guide.py` | (선택) 제자리 회전 방향 안내 |
| `check_sync.py` | 카메라–LiDAR 시각 차이 측정 (`max_sync_dt_sec` 권장값) |
| `premap_to_server.sh` | bag·지도 파일을 GPU 서버로 보내 `prebuild_from_bag.sh` 실행, `--wait`이면 결과 묶음 받기 |
| `stop_experiment.sh`, `stop_all.sh` | 녹화·인지 종료 / 위치추정·카메라까지 종료 |
| `monitor_run.py` | `run_experiment.sh`가 띄우는 누적 상황 기록 |
| `run_search.sh`, `run_cgd_live.sh`, `cgd_server_link.sh` | live search (아래) |

## 설치

1. 외부 코드 (아래 버전 clone, `~/conceptgraph_ws/` 아래):

| 저장소 | commit |
|---|---|
| [concept-graphs](https://github.com/concept-graphs/concept-graphs) | `93277a02bd` + `third_party/concept-graphs_model_utils_device.patch` |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | `a4d76a2b55` |
| [recognize-anything](https://github.com/xinyu1205/recognize-anything) | `7cb804a860` |
| [gradslam](https://github.com/gradslam/gradslam) | `59ca872e3d` |
| [chamferdist](https://github.com/krrish94/chamferdist) | `ee753899b5` |

   CG-D 수정은 patch 1건뿐입니다(`model_utils.compute_clip_features`의 `.to("cuda")` → `.to(device)`). GPU 결과는 같습니다.

2. Python 환경 `conceptgraph` (python 3.10, torch 2.0.1 + cu118): `server/requirements_server.txt`, 설치 순서 `server/setup_server_env.sh`, 환경 설정 `server/activate.sh`
3. 가중치: `server/download_weights.sh` (GroundingDINO SwinT-OGC, SAM ViT-H, RAM Swin-L, MobileSAM, CLIP ViT-H-14 laion2b_s32b_b79k; `server/weights.sha256`)
4. **`athome_interfaces` 패키지 필요 (이 저장소에 없음):** 탐색 서버 팀 저장소
   [robotjdals/AtHOME-object-search](https://github.com/robotjdals/AtHOME-object-search) (`dataset-v5-training` 브랜치)의
   `ros2/athome_interfaces`를 `semantic_geometry_ws/src/athome_interfaces`로 링크합니다.
   ```bash
   git clone --depth 1 -b dataset-v5-training --filter=blob:none --sparse https://github.com/robotjdals/AtHOME-object-search.git ~/team_repos/AtHOME-object-search
   git -C ~/team_repos/AtHOME-object-search sparse-checkout set ros2/athome_interfaces
   ln -s ../../team_repos/AtHOME-object-search/ros2/athome_interfaces semantic_geometry_ws/src/athome_interfaces
   ```
5. ROS 2 Humble: `colcon build --packages-select athome_interfaces semantic_geometry_msgs semantic_geometry_fusion`
   - `robot_bag_extract.py`는 빌드된 `install/semantic_geometry_fusion/.../config/robot.yaml`에서 카메라 extrinsic을 읽습니다(`--robot-yaml`로 바꿀 수 있음).
   - GPU 서버처럼 Ubuntu 20.04이면 conda RoboStack을 씁니다: `server/ros_env.sh`, `server/build_ros_ws.sh`.
   - ROS 단계(bag 추출)와 CG-D 단계는 **다른 셸**에서 실행합니다(두 환경의 OpenCV가 섞이면 오류).
6. GPU 서버 접속: `robot/config/gpu_server.env.example`을 `gpu_server.env`로 복사해 채웁니다(`.gitignore`로 저장소에 올라가지 않음).

## 사용: 사전 구축

```bash
# 1) 로봇 PC: 녹화 (로봇은 손으로 천천히 움직이고 자주 멈춤)
MAP_PCD=<maps>/<지도>.pcd MAP_YAML=<maps>/<지도>.yaml bash robot/scripts/record_premap.sh premap_A
#    RViz에서 "2D Pose Estimate"로 시작 위치 지정 → 녹화 시작
#    실시간 인식도 같이 보려면 record_premap.sh 대신 run_experiment.sh, 확인 창은 python3 robot/scripts/recog_wait_monitor.py
bash robot/scripts/record_premap.sh stop

# 2) 서버로 보내 처리하고 결과 받기
bash robot/scripts/premap_to_server.sh premap_A --wait
#    → 로봇 PC ~/real/results/premap_A/scene_graph/<버전>/ + premap_A_<버전>.zip(.sha256)
```

서버에서 직접 실행: `bash server/prebuild_from_bag.sh <이름>` (bag 위치 `/data/athome/prebuild/robot_mapping/<이름>/bag`)

| 옵션 (환경 변수) | 뜻 |
|---|---|
| `MAP=<nav2 yaml>` | occupancy로 쓸 Nav2 지도 (기본 lab 지도). `premap_to_server.sh`가 `map.env`에서 자동 지정 |
| `PCD=<pcd>` | 함께 전달할 위치추정 PCD (기본: yaml 이름에서 `_structural`을 뺀 `.pcd`) |
| `FLOOR_Z=<z>` | map 좌표의 바닥 높이. lab 지도는 −0.41, 다른 지도는 녹화 중 base_link 높이 중앙값 |

- GPU 조각 하나에 CG-D 하나만 올라갑니다. live용 `cgd_server.py`가 켜져 있으면 처리를 시작하지 않으므로 먼저 끕니다(`pkill -f tools/robot_live/cgd_server.py`).
- keyframe 폴더에서 직접: `tools/sim_sources/run_cgd_sources.sh <SOURCE_ROOT> v2` → `tools/scene_graph_dataset/build_dataset.py --exp <EXP> --sources robot --occupancy-map <nav2 yaml>`

### 결과 묶음 (`scene_graphs/<이름>/<버전>/`)

| 폴더 / 파일 | 내용 |
|---|---|
| `scene_graph_dataset/` | `index.json`, `schema/objects.schema.json`, `scenes/robot_<이름>/` (`objects.json`, `objects_all.json`, `points/obj_<id>.npy`, `occupancy/`, `metadata.json`) — 형식 v0.8 |
| `nav2_map/` | 사용한 지도의 2D occupancy grid (map_server yaml + 이미지) + `SHA256SUMS` |
| `pcd_map/` | 같은 지도의 위치추정용 PCD + `SHA256SUMS` |
| `manifest.json`, `README.md` | 이름, 버전, 물체 수, 지도 버전, 파일 해시 |
| `../<이름>_<버전>.zip` + `.sha256` | 전달용 묶음 |

지도 버전: `"sha256:" + sha256(지도 yaml 바이트 + 이미지 바이트)` 앞 16자리. 주행 쪽이 불러온 지도와 같아야 합니다.

## 사용: live search

```bash
bash robot/scripts/run_search.sh <사전 지도 objects.json>   # 위치추정 + 카메라 + CG-D(GPU 서버, CGD=local이면 로봇 PC) + fusion(online)
bash robot/scripts/stop_experiment.sh
```

- 서버 쪽 CG-D 서비스: `server/start_cgd_server.sh` (`ensure`: 이미 켜져 있으면 유지). SSH 터널은 `cgd_server_link.sh`가 유지합니다.
- 인식 노드는 처리할 때마다 최신 사진만 쓰고, 회전 중(> 0.1 rad/s) 사진은 건너뜁니다. 서버·터널이 끊기면 다시 연결합니다.

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/athome/perception/objects` | `athome_interfaces/ObservedObjectArray` | 사진마다 발행(빈 배열 포함), stamp = 촬영 시각, map 좌표, object_id = 사전 지도와 연결되면 그 ID, 아니면 −1 |
| `/athome/search/targets` | `athome_interfaces/SearchTargets` | 모든 query_names를 검출 질의에 추가, label은 query_name 그대로, 목표 검출은 신뢰도 0.4 이상 |
| `/cg_d/objects_2d`, `/cg_d/annotated` | `semantic_geometry_msgs/SemanticObject2DArray`, `sensor_msgs/Image` | CG-D 2D 검출, 검출 화면 |

## 주요 설정

- CG-D: GroundingDINO GPU fp32, SAM ViT-H·CLIP fp16, RAM CPU (repo 설정 fp32와 검출·물체 결과 동일, GPU 최대 5.4 GB). `GSA_MODE=offload`는 fp32로 모델을 하나씩 GPU에 올림
- 어휘 v2: 사진마다 RAM 태그 − `vocab/blocklist.txt` + `vocab/household.txt`, box/text threshold 0.2, 검출 신뢰도 0.25
- fusion: 연결 점수 1.1, voxel 5 cm, 관측별 DBSCAN 0.15 m, 상자 전 튀는 점 제거, 같은 물체 재병합 cos 0.9
- 실제 로봇 데이터만 2D 마스크 경계 3 px 축소 (`robot.yaml`, 빌더의 robot source)

## License

MIT (`LICENSE`). 외부 코드와 모델 가중치는 각 저장소의 라이선스를 따릅니다.
