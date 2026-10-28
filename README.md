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
  src/athome_interfaces/          (포함 안 됨) 탐색 서버 팀 저장소의 메시지 패키지를 링크 — 아래 설치 4
  tools/
    replica_rig/gsa_fast.py       CG-D 실행 wrapper (CG-D 코드는 그대로, 8 GB GPU용 장치·정밀도 분할)
    scene_graph_dataset/          데이터셋 빌더 build_dataset.py (keyframe + LiDAR + CG-D → objects.json)
    sim_sources/                  Replica / ProcTHOR / Gazebo / 로봇 bag → 공통 keyframe 형식, ROS 재생 시험
    robot_live/                   cgd_live_node (실시간 CG-D: 최신 사진만, 회전 중 사진 제외, --server로 GPU 서버 사용),
                                  cgd_engine (CG-D 한 장 추론), cgd_server / cgd_wire (GPU 서버 CG-D 서비스, SSH 터널),
                                  tag_check_node / tag_check_server (label만 빠르게 확인), V4L2 카메라 노드
    search_eval/                  검색 목표 평가 세트, "위에 놓임" 평가(회전 상자 / 축 정렬 상자), CLIP 특징 내보내기
robot/
  scripts/                        로봇 PC 스크립트: 위치추정, 카메라, 사전 구축 녹화(record_premap) → 서버 처리(premap_to_server),
                                  live search(run_search), 재생·지연·재연결 시험, tag 확인
  config/                         카메라·LiDAR calibration, FastDDS 공유메모리, RViz, gpu_server.env.example(서버 접속 설정 틀)
server/                           GPU 서버: conda 환경(CG-D, ROS 2 Humble RoboStack) 설치·빌드, 가중치 받기,
                                  사전 구축 처리(prebuild_from_bag), CG-D·tag 확인 서비스 시작
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
4. **`athome_interfaces` 패키지 필요 (이 저장소에 없음):** 탐색 서버 팀 저장소
   [robotjdals/AtHOME-object-search](https://github.com/robotjdals/AtHOME-object-search) (`dataset-v5-training` 브랜치)의
   `ros2/athome_interfaces`를 `semantic_geometry_ws/src/athome_interfaces`로 링크합니다.
   ```bash
   git clone --depth 1 -b dataset-v5-training --filter=blob:none --sparse https://github.com/robotjdals/AtHOME-object-search.git ~/team_repos/AtHOME-object-search
   git -C ~/team_repos/AtHOME-object-search sparse-checkout set ros2/athome_interfaces
   ln -s ../../team_repos/AtHOME-object-search/ros2/athome_interfaces semantic_geometry_ws/src/athome_interfaces
   ```
5. ROS 2 Humble: `colcon build --packages-select athome_interfaces semantic_geometry_msgs semantic_geometry_fusion`
   (GPU 서버처럼 Ubuntu 20.04면 conda RoboStack: `server/ros_env.sh`, `server/build_ros_ws.sh`)
6. 통신: 팀 전체 `rmw_zenoh_cpp`, `ROS_DOMAIN_ID=25` (`robot/scripts/env.sh`가 PC마다 zenoh 라우터를 띄우고,
   `ATHOME_ZENOH_ENDPOINT=tcp/<IP>:7447`이면 다른 PC 라우터에 연결). GPU 서버 접속은 `robot/config/gpu_server.env`
   (`gpu_server.env.example`을 복사해 채움)

## 사용

사전 구축 (녹화는 로봇 PC, 처리는 GPU 서버):

```bash
bash robot/scripts/record_premap.sh premap_A         # 위치추정 + 카메라 + LiDAR → rosbag (CAMERA=realsense: USB 3.0, 깊이 포함)
bash robot/scripts/record_premap.sh stop
bash robot/scripts/premap_to_server.sh premap_A --wait   # 서버로 보내 prebuild_from_bag.sh 실행 → 결과 받기
#   서버: bag → keyframe → CG-D → objects.json, /data/.../scene_graphs/premap_A/<버전>/ (+ latest, <이름>_<버전>.zip + .sha256)
```

전달 묶음 (`<버전>/`, 결과는 로봇 PC `~/real/results/<이름>/scene_graph/`로 받음):

| 폴더 / 파일 | 내용 |
|---|---|
| `scene_graph_dataset/` | `index.json`, `schema/objects.schema.json`, `scenes/robot_<이름>/` (`objects.json`, `metadata.json`, `points/`, `occupancy/`) |
| `nav2_map/` | 사용한 지도의 2D occupancy grid (yaml + 이미지) + `SHA256SUMS` |
| `pcd_map/` | 같은 지도의 위치추정용 PCD + `SHA256SUMS` (기본: Nav2 지도 이름에서 `_structural`을 뺀 `.pcd`, `PCD=`로 지정 가능) |
| `manifest.json`, `README.md` | 이름, 버전, 물체 수, 지도 버전, 파일 해시 / 묶음 설명 |

keyframe 폴더에서 직접: `tools/sim_sources/run_cgd_sources.sh <SOURCE_ROOT> v2` → `tools/scene_graph_dataset/build_dataset.py --exp <EXP> --sources robot`

live search (로봇 PC):

```bash
bash robot/scripts/run_search.sh <PRE_MAPPED_objects.json>     # 위치추정 + 카메라 + CG-D(GPU 서버, CGD=local이면 로봇 PC) + fusion(online)
bash robot/scripts/stop_experiment.sh
```

- 인식 노드는 처리할 때마다 **최신 사진**만 쓰고, **회전 중(> 0.1 rad/s) 사진은 건너뜁니다**. 서버·터널이 끊기면 다시 연결합니다.
- 확인·시험: `tag_check.sh`(label만, 장당 약 1.5 s), `remote_view.sh`(서버 인식 결과 보기), `search_replay.sh`(bag 재생 + 목표 토픽 흉내),
  `search_latency_check.sh`(촬영→결과 지연 측정), `fusion_replay_check.sh`(fusion 노드 재생)

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
- 실제 로봇 데이터만 2D 마스크 경계 3 px 축소 (`robot.yaml`, 빌더의 robot source). 카메라–LiDAR 정렬 오차로 마스크 경계에 섞이는 배경 점 제거
- 장면 그래프 `metadata.json`의 지도 버전: `"sha256:" + sha256(지도 yaml + 이미지)` 앞 16자리

## License

MIT (`LICENSE`). 외부 코드와 모델 가중치는 각 저장소의 라이선스를 따릅니다.
