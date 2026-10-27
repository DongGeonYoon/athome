# athome — 실내 물체 탐색 로봇의 인지(Perception) 모듈

AtHOME 프로젝트에서 인지 부분을 맡은 코드입니다. 로봇이 실내 물체를 찾을 때 필요한 물체 정보를 만듭니다. 카메라 사진의 의미 정보(ConceptGraphs CG-D: RAM → GroundingDINO → SAM → CLIP)와 LiDAR(Livox Mid-360)의 3D 위치를 합칩니다.

- **사전 구축 (offline):** 미리 녹화한 실내 탐색 데이터에서 물체별 feature를 만들어 Scene Graph용 `objects.json`을 출력합니다.
- **live search (online):** 탐색 중 사진마다 관측된 물체(label, CLIP feature, 3D 상자)를 `/athome/perception/objects`로 보냅니다. 찾는 목표 이름은 `/athome/search/targets`로 받아 검출 질의에 추가합니다.

물체 feature: `F_j = {ID, label, CLIP(1024), 점구름, 3D 상자(center, size, yaw), 관측 수}`

## 구성

```
semantic_geometry_ws/
  src/semantic_geometry_fusion/   ROS 2 노드 geometry_fusion_node (mask → LiDAR 점 → 3D 관측 → 물체 지도 / Scene Graph),
                                  geometry(DBSCAN, 튀는 점 제거), feature_fusion(연결·융합·재병합), config, launch
  src/semantic_geometry_msgs/     2D/3D 관측 메시지
  src/athome_interfaces/          (포함 안 됨) 탐색 서버 팀의 메시지 패키지를 여기에 둠 — 아래 설치 4
  tools/
    replica_rig/gsa_fast.py       CG-D 실행 wrapper (CG-D 코드는 그대로, 8 GB GPU용 장치·정밀도 분할)
    scene_graph_dataset/          데이터셋 빌더 build_dataset.py (keyframe + LiDAR + CG-D → objects.json)
    sim_sources/                  Replica / ProcTHOR / Gazebo / 로봇 bag → 공통 keyframe 형식, ROS 재생 시험
    robot_live/                   cgd_live_node (실시간 CG-D), V4L2 카메라 노드
    search_eval/                  검색 목표 평가 세트 생성·평가, 상자 평가
robot/
  scripts/                        로봇 실험·search·시험 스크립트 (localization, 카메라, 녹화, search, 재생, GPU 시험)
  config/                         카메라·LiDAR calibration, FastDDS 공유메모리, RViz
server/                           GPU 서버(conda) 환경 설치, 가중치 받기, 패키지 버전
third_party/                      CG-D에 적용한 patch
```

## 설치

1. 외부 코드 (아래 버전 clone, `~/conceptgraph_ws/` 아래):

| 저장소 | commit |
|---|---|
| [concept-graphs](https://github.com/concept-graphs/concept-graphs) | `93277a02bd` + `third_party/concept-graphs_model_utils_device.patch` |
| [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | `a4d76a2b55` |
| [recognize-anything](https://github.com/xinyu1205/recognize-anything) | `7cb804a860` |
| [gradslam](https://github.com/gradslam/gradslam) | `59ca872e3d` |
| [chamferdist](https://github.com/krrish94/chamferdist) | `ee753899b5` |

   CG-D 수정은 patch 1건뿐입니다: `model_utils.compute_clip_features`의 `.to("cuda")`를 `.to(device)`로 바꿈. GPU 실행 결과는 같습니다.

2. Python 환경 (python 3.10, torch 2.0.1 + cu118): `server/requirements_server.txt`, 설치 순서는 `server/setup_server_env.sh`
3. 가중치: `server/download_weights.sh` (GroundingDINO SwinT-OGC, SAM ViT-H, RAM Swin-L, MobileSAM, CLIP ViT-H-14 laion2b_s32b_b79k; `server/weights.sha256`)
4. **`athome_interfaces` 패키지 필요 (이 저장소에 없음):** 탐색 서버 팀 저장소의 `ros2/athome_interfaces`
   (`ObservedObject`, `ObservedObjectArray`, `SearchTarget`, `SearchTargets`)를 `semantic_geometry_ws/src/athome_interfaces`에 넣어 주세요.
   `geometry_fusion_node`, `cgd_live_node`, search 스크립트가 이 메시지를 씁니다.
5. ROS 2 Humble: `colcon build --packages-select athome_interfaces semantic_geometry_msgs semantic_geometry_fusion`

## 사용

사전 구축 (keyframe 폴더: `results/frame*.jpg`, `traj.txt`, `camera.json`, `lidar/`):

```bash
bash semantic_geometry_ws/tools/sim_sources/run_cgd_sources.sh <SOURCE_ROOT> v2       # CG-D (GPU 분할)
python semantic_geometry_ws/tools/scene_graph_dataset/build_dataset.py --exp <EXP> --sources robot --out-name sg
```

로봇 bag → keyframe: `semantic_geometry_ws/tools/sim_sources/robot_bag_extract.py`, 한 번에: `robot/scripts/offline_rebuild.sh`

live search (로봇):

```bash
bash robot/scripts/run_search.sh <PRE_MAPPED_objects.json>     # localization + 카메라 + live CG-D + fusion(online)
ros2 launch semantic_geometry_fusion perception.launch.py mode:=online max_sync_dt_sec:=<측정값> prior_objects_json:=<objects.json>
```

녹화로 시험: `robot/scripts/search_replay.sh` (bag 재생 + `search_server_mock.py`가 목표 토픽을 탐색 서버처럼 발행)

## 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/athome/perception/objects` | `athome_interfaces/ObservedObjectArray` | 사진마다 발행(빈 배열 포함), stamp = 촬영 시각, map 좌표, object_id = 사전 지도와 연결되면 그 ID, 아니면 −1 |
| `/athome/search/targets` | `athome_interfaces/SearchTargets` | `command_id`, `targets[{name, query_names[]}]`, reliable·transient local. 모든 query_names를 검출 질의에 추가, label은 query_names 그대로, 목표 신뢰도 0.4, 빈 목록이면 원래 어휘 |
| `/cg_d/objects_2d` | `semantic_geometry_msgs/SemanticObject2DArray` | CG-D 2D 검출(mask, CLIP) |

## 주요 설정

- CG-D: GroundingDINO GPU fp32, SAM ViT-H·CLIP fp16, RAM CPU (8 GB GPU에서 repo 설정 fp32와 결과 동일 확인). `GSA_MODE=offload`는 repo 설정 그대로 모델을 하나씩 GPU에 올림
- 어휘: RAM 태그 − `vocab/blocklist.txt` + `vocab/household.txt`, box/text threshold 0.2
- fusion: 연결 점수 φ_sem=(cos+1)/2 + φ_geo ≥ 1.1, voxel 5 cm, 관측별 DBSCAN 0.15 m, 상자 전 튀는 점 제거(통계적 이상점 ∩ 8 cm 이상 외톨이), 같은 물체 재병합 cos 0.9

## License

MIT (`LICENSE`). 외부 코드와 모델 가중치는 각 저장소의 라이선스를 따릅니다.
