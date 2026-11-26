# athome — 실내 물체 탐색 로봇의 사전 구축(Pre-mapping) 인지 모듈

로봇으로 실내를 한 번 녹화한 rosbag에서 **Scene Graph 입력용 물체 데이터셋**(`objects.json`)을 만드는 코드입니다. 카메라 사진의 의미 정보는 foundation model 조합(ConceptGraphs의 CG-D: RAM → Grounding DINO → SAM → CLIP)으로 얻습니다. 3D 위치와 형상은 LiDAR(Livox Mid-360, FAST-LIO)로 얻습니다.

```
물체 j:  F_j = { object_id, semantic_tag, confidence, centroid, bbox_3d(center, size, yaw), clip_feature(1024), num_observations }
          + points/obj_<id>.npy (map 좌표 점구름)
```

- 이 저장소는 **사전 구축(offline)만** 담습니다. 탐색 중 실시간 인식 코드까지 포함한 이전 판은 태그 `live-search-20261105`, 확인·시험 도구까지 포함한 판은 태그 `full-20261028`에 있습니다.
- 실물 로봇(D435 + Mid-360, PCD 지도 위치추정)이 기준입니다. 시뮬레이션(Replica, ProcTHOR, Gazebo) 데이터셋도 같은 빌더로 만듭니다.

---

## 1. 처리 과정

```
[로봇 PC] robot/scripts/record_premap.sh
   위치추정(FAST_LIO_LOCALIZATION2, PCD 지도) + D435 RGB 1280×720 + Mid-360 → rosbag
        │ premap_to_server.sh: bag + 지도 전송, 처리 시작, 결과 받기 (GPU PC에서 직접 처리해도 됨, 4.3절)
        ▼
[GPU] server/prebuild_from_bag.sh
   1 robot_bag_extract.py   keyframe 선택(저속·정지 구간, 0.3 m / 5° / 0.5 s), keyframe별 LiDAR(0.5 s 누적), map 자세, LiDAR 지도
   2 run_cgd_sources.sh     CG-D (gsa_fast.py: CG-D 원본 코드를 그대로 실행, 어휘 v2)
   3 build_dataset.py       2D 검출 → LiDAR 3D 관측 → 물체 연결·융합 → objects.json (+ Nav2 지도를 occupancy로)
   4 전달 묶음               scene_graph_dataset/ + nav2_map/ + pcd_map/ + manifest.json + zip/sha256
        │ (선택) 사람의 라벨 확인이 필요한 장소
        ▼
[GPU/PC] tools/label_correction/
   prepare.py → build.py → finalize.py   검출별 태그 수정, 같은 태그끼리 연결, 단일 물품 여러 뷰 배치, 가구 맞춤
```

---

## 2. 참고한 foundation model과 논문

| 단계 | 모델 / 방법 | 논문 | 이 저장소에서 쓰는 형태 |
|---|---|---|---|
| 전체 틀 | ConceptGraphs (CG-D 검출 경로) | Gu et al., *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning*, ICRA 2024 (arXiv:2309.16650) | 사진마다 태그 → 검출 → 마스크 → CLIP을 만들고, 관측을 3D 물체로 연결·융합하는 구조 |
| 태그 | RAM (Swin-L) | Zhang et al., *Recognize Anything: A Strong Image Tagging Model*, arXiv:2306.03514 | 사진의 물체 이름 후보 |
| 검출 | Grounding DINO (SwinT-OGC) | Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, ECCV 2024 (arXiv:2303.05499) | 이름 목록으로 상자 검출 (box/text threshold 0.2) |
| 분할 | SAM (ViT-H) | Kirillov et al., *Segment Anything*, ICCV 2023 (arXiv:2304.02643) | 상자 → 마스크 |
| 검출+분할 묶음 | Grounded-SAM | Ren et al., *Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks*, arXiv:2401.14159 | Grounding DINO와 SAM을 잇는 공개 구현 |
| 의미 feature | CLIP / OpenCLIP ViT-H-14 (laion2b_s32b_b79k) | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021; Cherti et al., *Reproducible Scaling Laws for Contrastive Language-Image Learning*, CVPR 2023 | 마스크 영역 crop → 1024차원 feature |
| 위치·LiDAR | FAST-LIO2 (+ PCD 지도 위치추정) | Xu et al., *FAST-LIO2: Fast Direct LiDAR-Inertial Odometry*, IEEE T-RO 2022 | map 좌표 자세, body 좌표 점군 |
| 시뮬레이션 장면 | Replica, ProcTHOR | Straub et al., *The Replica Dataset*, arXiv:1906.05797; Deitke et al., *ProcTHOR*, NeurIPS 2022 | 정답(GT)이 있는 평가 장면 |

### 연결 점수 (ConceptGraphs 방식)
새 관측 o와 지도 물체 j의 점수는 `φ_sem + φ_geo`이고, 기준값 이상이면 같은 물체로 봅니다.
- `φ_sem = (cos(CLIP_o, CLIP_j) + 1) / 2`
- `φ_geo`: o의 점 중 j의 점과 δ = 5 cm 안에 있는 비율

융합하면 CLIP은 관측 수로 가중 평균한 뒤 정규화하고, 점은 voxel 5 cm로 합치며, 이름은 관측 이름의 다수결로 정합니다.

---

## 3. 원본 방법에서 바꾸거나 더한 것

foundation model과 CG-D 원본 코드는 고치지 않았습니다. 장치 설정 patch 1건만 있습니다. 아래 변경은 모두 바깥 wrapper(`gsa_fast.py`)와 빌더에서 처리했습니다.

| 원본 방식의 문제 | 이 저장소의 처리 | 근거 결과 |
|---|---|---|
| ConceptGraphs는 RGB-D depth로 3D를 만드는데, 로봇 D435는 USB 2.0이라 depth를 쓸 수 없음 | **LiDAR 기하:** 마스크 안에 투영되는 FAST-LIO body 점군(0.5 s 누적)을 사용, depth gate(MAD 2.5), 관측별 DBSCAN 0.15 m. depth는 LiDAR 점이 3개 미만일 때만 보조 | Gazebo 실제 ROS 경로와 오프라인 결과 비교: 물체 90쌍 일치, 중심 거리 중앙값 0.000 m |
| 카메라–LiDAR 시각 차이·정렬 오차로 배경 점이 마스크 가장자리에 섞임 | **실물만 마스크 경계 3 px 축소** | run4 책상 위 작은 물체 6개 중 책상 "위"로 잡힌 수 1 → 5 |
| 튀는 점 하나가 상자를 키움 | **상자 계산 전 외톨이 점 제거** (k = 8, 2σ, 8 cm 이상 떨어진 점) | run4 책상 윗면 높이 0.44 → 0.33 m |
| RAM 태그가 프레임마다 누적되어 엉뚱한 이름이 늘고, 작은 생활용품 이름은 잘 안 나옴 | **어휘 v2:** 프레임별 RAM 태그 − `vocab/blocklist.txt`(풍경·그림 내용·사람·추상어) + `vocab/household.txt`(생활용품 46개), 누적 끔 | ProcTHOR 2개 집 의미 정밀도 0.33 → 0.52, 0.42 → 0.56 |
| Grounding DINO 문구 → 클래스 대응이 첫 단어만 보고 실패하면 0번 클래스를 붙임 | **대응 규칙 교체:** 완전 일치 → 문구에 포함된 가장 긴 클래스 → 단어 일치 → 없으면 `other item` | 무작위 이름(예: `chocolate cake`)이 붙던 문제 제거 |
| 연결 기준값 1.2는 CG-D(raw cosine) 기준이라 위 점수식에서는 검증되지 않음 | **GT 30개 장면에서 1.0–1.4를 비교해 1.1 채택** | 6.1절 표 |
| 한 물체가 두 물체로 남음 (다른 면, 얇은 조각, 동의어 이름) | **내보내기 직전 재병합:** CLIP cos ≥ 0.9이고 공간을 공유하며, 합친 상자가 큰 쪽의 1.25배 이내일 때 | 실물 run2 쪼개짐 22 → 13쌍, GT 의미 재현율 유지(0.376 → 0.372) |
| 문(2 m)에 `remote control` 같은 작은 물체 이름이 붙음 | **크기 사전 검사:** 작은 물체 이름의 상자가 상한 × 1.5를 넘으면 크기에 맞는 다른 투표 이름으로, 없으면 제외 | 비현실적 이름 제거 |
| 8 GB GPU에서 원본 설정(전부 GPU fp32)이 메모리 부족 | **장치·정밀도 분할:** RAM은 CPU, SAM·CLIP은 fp16, Grounding DINO는 GPU fp32 | 원본 설정과 검출·물체 결과 동일, GPU 최대 5.4 GB |
| 로봇 주행 데이터에는 keyframe, 지도 좌표, 바닥 높이가 없음 | **로봇 입력 변환:** 저속·정지 구간 keyframe, 위치추정 TF로 map 좌표, base_link 높이로 바닥 추정, Nav2 지도를 occupancy로 붙이고 지도 버전(해시)을 기록 | 2D 지도와 같은 좌표의 물체 지도 |
| 자동 결과에 목록 밖 이름, 오검출, 같은 물품 중복이 남음 | **(선택) 사람의 라벨 확인 단계 `tools/label_correction`** (5절) | room_baram: 물체 41 → 34개, 목록에 있는 이름 25 → 34개, 중복 13 → 0 |

---

## 4. 설치와 사용

### 4.1 설치
0. **저장소 배치:** 스크립트가 아래 경로를 기준으로 동작합니다.
   ```bash
   git clone https://github.com/DongGeonYoon/athome.git ~/athome
   ln -s ~/athome/semantic_geometry_ws ~/semantic_geometry_ws
   mkdir -p ~/real/results
   ln -s ~/athome/robot/scripts ~/real/scripts
   ln -s ~/athome/robot/config ~/real/config
   ```
   외부 코드는 `~/conceptgraph_ws`, conda는 `~/miniforge3`, 팀 위치추정 워크스페이스는 `~/fast_lio_ws`에 둡니다.
1. 외부 코드 (`~/conceptgraph_ws/` 아래, 아래 commit):

   | 저장소 | commit |
   |---|---|
   | [concept-graphs](https://github.com/concept-graphs/concept-graphs) | `93277a02bd` + `third_party/concept-graphs_model_utils_device.patch` |
   | [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | `a4d76a2b55` |
   | [recognize-anything](https://github.com/xinyu1205/recognize-anything) | `7cb804a860` |
   | [gradslam](https://github.com/gradslam/gradslam) | `59ca872e3d` |
   | [chamferdist](https://github.com/krrish94/chamferdist) | `ee753899b5` |

   CG-D 수정은 patch 1건뿐입니다(`compute_clip_features`의 `.to("cuda")` → `.to(device)`).
2. Python 환경 `conceptgraph` (python 3.10, torch 2.0.1 + cu118): `server/requirements_server.txt`, 설치 순서 `server/setup_server_env.sh`, 환경 설정 `server/activate.sh`
3. 가중치: `server/download_weights.sh` (해시 `server/weights.sha256`)
4. ROS 2 Humble: bag 추출(`robot_bag_extract.py`)에만 필요합니다. 이 저장소에는 빌드할 ROS 패키지가 없습니다. Ubuntu 20.04 서버에서는 conda RoboStack 환경을 씁니다(`server/ros_env.sh`).
   - ROS 단계(bag 추출)와 CG-D 단계는 **다른 셸**에서 실행합니다. 두 환경의 OpenCV가 섞이면 오류가 납니다.
5. **팀 위치추정 워크스페이스 (이 저장소에 없음):** `start_localization.sh`는 `~/fast_lio_ws`의 `mppi_nav/slam_costmap_bringup.launch.py`를 실행합니다. 위치추정 설정(FAST-LIO, calibration, TF)은 그 워크스페이스의 것을 그대로 씁니다.
6. 카메라 extrinsic: `semantic_geometry_ws/config/robot_calibration.yaml` (`T_cam_body`)
7. GPU 서버를 쓸 때: `robot/config/gpu_server.env.example`을 `gpu_server.env`로 복사해 채웁니다(저장소에 올라가지 않음).

### 4.2 녹화 (로봇 PC)
```bash
MAP_PCD=<pcd> MAP_YAML=<nav2 yaml> bash robot/scripts/record_premap.sh premap_A
#  RViz에서 "2D Pose Estimate"로 시작 위치를 지정하면 그때부터 녹화가 시작됩니다.
#  로봇은 천천히 움직이고 자주 멈춥니다 (keyframe은 멈췄거나 느린 구간에서 뽑힘).
bash robot/scripts/record_premap.sh stop
```

| 스크립트 | 역할 |
|---|---|
| `env.sh` | ROS 환경, `ROS_DOMAIN_ID=25`, `rmw_zenoh_cpp` (`ATHOME_RMW=fastdds`로 되돌림) |
| `start_localization.sh` | Livox + FAST_LIO_LOCALIZATION2 + map_server + RViz (`MAP_PCD`, `MAP_YAML`) |
| `start_camera_v4l2.sh`, `v4l2_rgbd_node.py` | D435 컬러를 V4L2로 직접 읽음 (USB 2.0, 1280×720). `start_camera.sh`는 USB 3.0용 공식 드라이버 |
| `check_sync.py` | 카메라–LiDAR 시각 차이 측정 |
| `record_premap.sh` | 위치추정 + 카메라 + 위치 지정 대기 + 시각 차이 확인 + rosbag. 쓴 지도를 `map.env`에 기록 |
| `premap_to_server.sh` | bag·지도를 GPU 서버로 보내 처리, `--wait`이면 결과 묶음 받기 |
| `stop_all.sh` | 녹화·카메라·위치추정 종료 |

### 4.3 데이터셋 만들기
GPU 서버:
```bash
bash robot/scripts/premap_to_server.sh premap_A --wait
#  → ~/real/results/premap_A/scene_graph/<버전>/ + premap_A_<버전>.zip(.sha256)
```
GPU PC에서 직접:
```bash
T=~/semantic_geometry_ws/tools; EXP=~/exp_premap
# 1) bag → keyframe (ROS 셸)
source /opt/ros/humble/setup.bash
python3 $T/sim_sources/robot_bag_extract.py ~/real/results/premap_A/bag $EXP/robot/sources/premap_A --name premap_A \
  --max-speed 0.5 --max-turn 0.6 --kf-dist 0.3 --kf-yaw 5 --kf-gap 0.5
# 2) CG-D, 3) 데이터셋 (새 셸, conceptgraph 환경)
bash $T/sim_sources/run_cgd_sources.sh $EXP/robot/sources v2
(cd $T/scene_graph_dataset && ~/miniforge3/envs/conceptgraph/bin/python build_dataset.py --exp $EXP --sources robot --out-name sg \
  --occupancy-map <nav2 yaml>)
#  → $EXP/sg/scenes/robot_premap_A/objects.json  (8 GB GPU에서 keyframe 47장 약 4분)
```

| 결과 | 내용 |
|---|---|
| `scene_graph_dataset/` | `index.json`, `schema/objects.schema.json`, `scenes/robot_<이름>/` (`objects.json`, `objects_all.json`, `points/`, `occupancy/`, `metadata.json`), 형식 v0.8 |
| `nav2_map/`, `pcd_map/` | 사용한 2D 지도(yaml + 이미지), 위치추정 PCD, 각 `SHA256SUMS` |
| `manifest.json` | 이름, 버전, 물체 수, 지도 버전 `"sha256:" + sha256(yaml 바이트 + 이미지 바이트)` 앞 16자리 |

- `objects.json`은 관측 2회 이상이고 일반 태그(`other item` 등)가 아닌 물체만 담습니다. `objects_all.json`은 필터 전 전체입니다.

### 4.4 시뮬레이션 데이터셋
같은 빌더로 `--sources replica|replica_full|procthor|gazebo`를 지정합니다.
- **Replica:** 렌더링부터 할 수 있습니다(`replica_rig/render_rig.py`).
- **Gazebo:** bag 추출(`sim_sources/gazebo_extract.py`, Livox 메시지가 있는 `fast_lio_ws` 필요)부터 할 수 있습니다.
- **ProcTHOR, Replica 전체판:** 수집해 둔 원천 데이터가 필요합니다(수집 도구는 태그 `full-20261028`).
```bash
PY=~/miniforge3/envs/conceptgraph/bin/python; T=~/semantic_geometry_ws/tools
(cd $T/replica_rig && $PY render_rig.py --scene room0 --out ~/exp_sim)
bash $T/sim_sources/run_cgd_sources.sh ~/exp_sim/replica_rig v2
(cd $T/scene_graph_dataset && $PY build_dataset.py --exp ~/exp_sim --sources replica --out-name sg)
```

---

## 5. (선택) 라벨 확인 단계: `tools/label_correction`

자동 결과에는 목록 밖 이름(`instrument`, `dish washer`), 오검출(칸막이, 그림), 같은 물품의 중복이 남습니다. 방에 놓인 물품 목록을 아는 경우 사람이 검출 단위로 태그를 정하고, 나머지를 다시 만듭니다. foundation model과 빌더 코드는 그대로이며, 결정과 설정 파일만 방마다 다릅니다. 예시: `configs/room_baram_labels.txt`, `configs/room_baram.yaml`

```bash
L=~/semantic_geometry_ws/tools/label_correction; PY=~/miniforge3/envs/conceptgraph/bin/python; SRC=$EXP/robot/sources/<이름>
$PY $L/prepare.py $SRC labels.txt --exclude-keyframes 0
$PY $L/build.py --exp $EXP --occupancy-map <nav2 yaml> --single-view-tags cup,headphone,plate --trace trace.json
$PY $L/finalize.py room.yaml $SRC labels.txt $EXP/sg_labeled out_dataset --trace trace.json --keyframe-json kf/ --figures fig/
```

| 단계 | 내용 |
|---|---|
| `labels.txt` | keyframe마다 `<keyframe>: <검출 id>=<태그> ...`. 적지 않은 검출은 제거 |
| `prepare.py` | 태그를 바꾼 검출 사본(`gsa_detections_ram_v2labeled`)을 만듭니다. 제외 keyframe(예: 위치 지정 전 사진)은 검출과 LiDAR 지도 점을 모두 뺍니다 |
| `build.py` | 같은 태그끼리만 연결하고, 지정 태그는 관측 1회도 내보냅니다. `--trace`: 물체마다 어느 keyframe의 어느 검출로 만들어졌는지 기록 |
| `finalize.py` | **단일 물품:** 여러 뷰의 마스크 아래쪽 시선을 받침면(책상 면 또는 바닥)과 교차해 위치를 정하고, 그 근처에서 여러 뷰가 동의한 LiDAR 점을 씁니다. LiDAR 점이 없으면 위치만 정하고 크기는 미측정(`points_source: views_only`)으로 표시합니다. **개수를 아는 가구(책상):** 책상 높이 LiDAR 단면에 사각형을 맞춥니다 |

같은 입력으로 다시 실행하면 room_baram 최종 데이터셋의 45개 물체가 모두 일치합니다(위치·크기 1 mm, CLIP cos 0.9999 이내).

---

## 6. 실제로 만든 데이터셋과 결과

### 6.1 시뮬레이션 데이터셋 v0.8 (39개 장면)

| 소스 | 장면 | keyframe | 물체 | 특징 |
|---|---|---|---|---|
| Replica (NICE-SLAM 배포판) | 8 | 2,712 | 329 | 방 1개, mesh 렌더링 |
| Replica 전체판 | 10 | 525 | 562 | 여러 방, semantic mesh GT |
| ProcTHOR-10k | 20 집 | 1,194 | 1,118 | 여러 방, 작은 물체 많음 |
| Gazebo (AWS small house) | 1 | 108 | 106 | 실제 ROS 경로 (Mid-360 plugin → FAST-LIO → TF) |
| **합계** | **39** | **4,539** | **2,115** | |

로봇 rig 조건(카메라 높이 0.77 m, Mid-360 0.445 m, 실제 calibration 값)으로 재현했습니다.

**GT 평가** (GT 중심과 물체 상자 중심의 XY 거리 0.6 m 이내를 매칭으로 봄, 동의어 허용)

| 소스 | 기하 정밀도 | 의미 정밀도* | 기하 재현율 | 의미 재현율 | 작은 물체(< 0.3 m) 의미 재현율 |
|---|---|---|---|---|---|
| Replica 전체판 | 0.94 | 0.48 | 0.81 | 0.30 | 0.31 |
| ProcTHOR | 0.83 | 0.60 | 0.77 | 0.37 | 0.12 |
| Gazebo | 0.68 | 0.43 | 0.86 | 0.55 | – |

\* 문·창문·커튼 태그를 뺀 값입니다. 기하는 안정적이고, 의미 재현율을 낮추는 주원인은 CG-D 검출 단계입니다(ProcTHOR: 큰 물체 79개 보임 → 67 검출 → 65 기하 확보).

**연결 기준값 비교** (ProcTHOR 20 / Replica 전체판 10)

| 기준값 | 의미 재현율 | 같은 태그 중복 (0.3 m 안) | 판단 |
|---|---|---|---|
| 1.0 | 0.387 / 0.300 | 7 / 13 | 과병합 증가 |
| **1.1** | **0.376 / 0.305** | **9 / 15** | 채택 |
| 1.2 | 0.364 / 0.300 | 11 / 16 | |
| 1.4 | 0.338 / 0.271 | 23 / 29 | 쪼개짐 증가 |

**어휘 비교** (ProcTHOR 2개 집)

| 설정 | 의미 정밀도 | 의미 재현율 | 작은 물체 재현율 |
|---|---|---|---|
| 원본 (RAM 태그 누적) | 0.33 / 0.42 | 0.17 / 0.31 | 0.01 / 0.03 |
| v1: 누적 끔 | 0.51 / 0.49 | 0.14 / 0.18 | 0.00 / 0.03 |
| **v2: 누적 끔 + blocklist + household** | **0.52 / 0.56** | **0.19 / 0.31** | **0.01 / 0.10** |
| v3: v2 + 누적 | 0.42 / 0.50 | 0.21 / 0.34 | 0.04 / 0.08 |

**실제 ROS 경로 검증 (Gazebo):** 같은 bag에서 ROS 노드와 오프라인 빌더의 결과를 비교했습니다. 물체 97 / 106개 중 90쌍이 0.3 m 안에서 짝지어졌고(89쌍은 이름도 같음), 중심 거리 중앙값 0.000 m, CLIP cos 중앙값 1.000이었습니다. FAST-LIO 자세 오차는 중앙값 0.13 m입니다.

### 6.2 실물 로봇 데이터셋

| 녹화 | 장소 | 녹화 | keyframe | 처리 단계 | 최종 물체 |
|---|---|---|---|---|---|
| run5 (2026-10-28) | 412호 | 주행 | 47 | 관측 444 → 물체 66 | **38** |
| run6 (11-03) | 406호 | 주행, 343 s | 51 | 검출 335 → 관측 273 → 물체 42 | **18** |
| room_baram (11-05) | 4개 구역 방 | 주행, 452 s | 76 | 검출 686 → 라벨 확인 505 (keyframe 0 제외 495) → 관측 374 | **34** (태그 27종) |

- 카메라 D435 컬러 1280×720(fx 903.47, 수평 시야 70.6°), depth 없음. 카메라–LiDAR 시각 차이는 최대 0.05~0.07 s입니다.
- 바닥 높이: lab 지도 z −0.41 m, 그 밖의 지도는 base_link 높이 중앙값(406호 −0.36, room_baram −0.39)입니다.
- **자동 결과의 이름 정확도 (run6, 사진으로 확인한 14개):** 맞음 9개(64 %). 틀린 경우는 도마 → `book`, 책상 일부 → `drawer`·`table`·`chair`, 화이트보드 → `computer monitor`였습니다.
- **room_baram 라벨 확인 전후** (물품 27종 기준)

  | | 자동 | 라벨 확인 후 |
  |---|---|---|
  | 물체 수 | 41 | 34 (책상 8 + 26종 각 1개) |
  | 목록에 있는 이름 | 25 | 34 |
  | 같은 물품 중복 (책상 제외) | 13 | 0 |
  | 태그 종류 (27종 중) | 12 | 27 |

  - 위치 출처: LiDAR 15개, LiDAR 단면 책상 8개, 사진 기반 추정 11개(크기 미측정). 사진 기반 위치는 뷰 간 차이가 0.14~0.23 m입니다.
  - 그릇, 수저, 칫솔은 LiDAR 점이 0개였습니다. 0.445 m 높이의 LiDAR가 책상 위 작은 물체를 거의 맞히지 못하기 때문입니다.
- 이 저장소 코드로 room_baram bag을 다시 추출하면 서버 결과와 같습니다(keyframe 76장 자세·LiDAR 동일, LiDAR 지도 682,986점).

### 6.3 처리 시간과 메모리 (CG-D, 사진 1280×720 한 장, RTX 4060 Laptop 8 GB)

| RAM (CPU) | Grounding DINO | SAM | CLIP | 합계 |
|---|---|---|---|---|
| 1.52 s | 0.33 s | 0.77 s | 0.66 s | 약 3.3 s |

- GPU 메모리 최대 5.4 GB(RAM CPU, SAM·CLIP fp16). 원본 설정(전부 GPU fp32)은 8 GB에서 메모리가 부족했습니다.
- 공유 GPU 서버(A100 MIG 조각)는 혼잡할 때 사진당 약 30 s까지 느려졌습니다(keyframe 76장 약 40분).

### 6.4 (부록) 사전 지도 사진으로 측정한 검출 성능
로봇이 멈춘 위치의 사진에서, 기준 물체가 보일 때 같은 이름으로 검출되는 비율입니다(신뢰도 0.25, 목표 이름 0.4).

| 장소 | 목표 이름 없음 | 목표 이름 추가 | 비고 |
|---|---|---|---|
| 412호 | 0.55 | 0.52 | 1.5 m 안 0.84 |
| 406호 | 0.75 | 0.67 | 1.5 m 안 0.88, 1.5~3 m 0.31 |
| room_baram | 0.47 | 0.49 | 목표 27종. 사진 기반 위치 물품은 판정 오차가 섞임 |

거리가 멀어지면 크게 떨어집니다. 놓친 물체는 같은 자리에서 다시 볼 때보다 다른 자리에서 볼 때 더 잘 검출되었습니다(재검출률: 같은 자리 0.07~0.13, 다른 자리 0.32~0.50).

---

## 7. 한계
1. **의미 정확도:** 자동 결과의 이름 오류는 CG-D 검출 단계에서 생깁니다. 목록이 정해진 장소는 5절의 라벨 확인으로 보완했지만, 사람의 판단이 들어간 결과입니다.
2. **작은 물체의 3D:** LiDAR 높이(0.445 m) 때문에 책상 위 작은 물체에는 점이 거의 없습니다. 카메라 depth를 함께 녹화하면(USB 3.0, `CAMERA=realsense`) 나아질 수 있습니다.
3. **상자 크기:** LiDAR가 앞면 위주로 보기 때문에 짧은 변이 작게 나오고, 주변 점이 섞인 큰 상자도 남습니다(room_baram 의자 1.72 m).
4. **위치 실측:** 실물 데이터셋의 물체 위치는 현장 실측과 비교하지 않았습니다(사진 투영으로만 확인).

## 8. 구성

```
semantic_geometry_ws/
  config/robot_calibration.yaml        카메라 extrinsic (T_cam_body)
  src/semantic_geometry_fusion/        fusion 라이브러리: geometry.py, association.py, feature_fusion.py, scene_graph_export.py
  tools/
    sim_sources/                       robot_bag_extract.py (로봇 bag → keyframe), gazebo_extract.py, run_cgd_sources.sh
    replica_rig/                       gsa_fast.py (CG-D 실행 wrapper), vocab/ (어휘 v2), Replica 렌더링·공용 상수
    replica_feasibility/plyload.py
    scene_graph_dataset/               build_dataset.py, objects.schema.json
    label_correction/                  prepare.py, build.py, finalize.py, common.py, configs/ (room_baram 예시)
robot/scripts, robot/config            녹화 스크립트, 카메라 노드, calibration, py_shim (위치추정 import 우회)
server/                                GPU 서버 환경 설치, 가중치, prebuild_from_bag.sh
third_party/                           CG-D patch 1건
```

## License
MIT (`LICENSE`). 외부 코드와 모델 가중치는 각 저장소의 라이선스를 따릅니다.
