# `multi_legged_robot_env_cfg.py` 해설과 2026-11-25 수정 내역

기반 환경 설정 파일(`source/.../multi_legged_robot/multi_legged_robot_env_cfg.py`)을 쉽게 풀어 쓴 문서다.
CPG v1(`multi_legged_robot_cpg_env_cfg.py`)과 CPG2(`multi_legged_robot_cpg2_env_cfg.py`)가 모두 이 파일을 상속한다.

- **→ CPG2** 표시는 CPG2 에서 값이 바뀌거나 안 쓰는 부분이다.
- 9월 논문 결과를 똑같이 재현하려면 이 수정 전 커밋(`main` 의 `2c5319b`)을 쓴다.

---

## 1. 이번 수정 요약 (2026-11-25)

| # | 내용 | 기존 학습 결과와 비교 |
|---|---|---|
| 1 | 몸통 수평 보상 정리. `BASE_LEVEL_FLAT/ROUGH_GAIN`(둘 다 0.8)과 `terrain_adaptive_base_leveling_l2` 를 지우고 `base_leveling`(가중치 −0.8) 하나로 합침 | **수치상 동일**. 텐서보드 이름만 `terrain_adaptive_base_leveling` → `base_leveling` |
| 2 | `imu_projected_gravity_xy_l2` 보상 항 삭제. 1번과 같은 계산이고 가중치도 0이었음 (함수는 1번이 재사용) | 동일 |
| 3 | `support_contact_count` 항과 `feet_support_count` 함수 삭제. 가중치 0이었음 | 동일 |
| 4 | 발 힘 벌점 기준 700 N → 정적 하중의 1.5배 (`FEET_FORCE_LIMIT` ≈ 1168 N) | 벌점이 조금 줄어듦(원래 크기가 작았음) |
| 5 | armature 무작위화 이벤트 추가 (`randomize_armature`, 학습 시작 때 ×0.5~1.5). `ARMATURE_RANDOM_SCALE = None` 이면 꺼짐 | 학습 분포가 바뀜 |
| 6 | 순수 RL 의 명령 재샘플링 8초 고정 → 5~10초 무작위 (CPG v1, CPG2 는 각자 값을 따로 씀) | 순수 RL 만 바뀜 |
| 7 | GUI 카메라가 로봇 하나를 따라감. `train.py`, `play.py` 에서 env 수가 확정된 뒤 무작위로 고름. `HUGO_VIEW_ENV=번호` 로 고정 가능 | 학습과 무관 |

관측 차원은 바뀌지 않았다. 9월 체크포인트(순수 RL, CPG v1)도 그대로 불러올 수 있다.

CPG v1 cfg 는 1·3번 이름 변경에 맞춰 두 줄만 고쳤다. 값은 그대로다(`base_leveling -0.8` = 예전 −1.0 × 0.8).

---

## 2. 파일 해설

### 0) 이 파일의 정체

Isaac Lab "manager-based" 환경의 **설계도**다. 환경은 다음 매니저들로 이루어진다.

- 장면: 로봇, 지형, 센서
- 관측: 정책이 보는 것
- 행동: 정책 출력을 관절 목표로 바꾸는 것
- 보상, 종료, 이벤트(무작위화), 명령(목표 속도)

각 매니저에 "무엇을, 어떤 함수로, 얼마나" 할지 채워 넣는다. 중간에 있는 함수들은 매니저가 매 스텝 불러 쓰는 계산식이다.

### 1) 경로와 기본 상수

| 이름 | 값 | 뜻 |
|---|---|---|
| `ROBOT_USD_PATH` | 연구실 PC 절대경로 | 로봇 모델 파일. PC가 바뀌면 여기부터 고친다 |
| `ROLL_SPLAY` | 0.2 rad | 다리를 바깥으로 벌린 각(약 11°). 지지 면적을 넓히려는 목적 |
| `PRISMATIC_DEFAULT` | 0.03 m | 직동관절 기본 위치. 바닥(0)에 두면 위아래 양쪽으로 못 써서 띄워둠 |
| `NOMINAL_LEG_DROP` / `NOMINAL_BODY_HEIGHT` | 0.74 / 0.725 m | 위 자세로 섰을 때 hip→발바닥 거리 / 몸통 높이 |
| `INITIAL_BODY_HEIGHT` | 0.80 m | 처음 놓는 높이. 바닥에 박히지 않게 띄움 |
| `SIM_DT`, `DECIMATION` | 1/200, 4 | 물리 200 Hz, 정책 50 Hz |
| `ENV_SPACING` | 4 m | 로봇 간격. 지형 패치를 쓰면 무시되지만 필수 인자이고, 평지(plane)로 바꾸면 다시 쓰인다 |

→ CPG2: 기본 자세와 높이는 IK 기준 자세(몸통 0.74 m, 무릎 약 50° 굽힘)로 덮어쓴다.

### 2) 높이 스캐너 상수

몸통에 붙어 아래로 쏘는 **레이저 격자**다.

- `OFFSET_POS (0.10, 0, 0.40)`: 몸통 중심에서 앞으로 10 cm, 위로 40 cm 지점에서 쏜다.
- `SIZE (1.6, 1.0)`, `RESOLUTION 0.1`: 17 × 11 = **187개 점**
- `NOMINAL_HEIGHT`: 평지에서 정상 높이로 서 있을 때 스캔 값이 0이 되도록 빼주는 기준값

### 3) 지형 적응 보상 상수 (9월 논문)

- `FLAT_ / ROUGH_ROUGHNESS_THRESHOLD` (0.025 / 0.060 m): 스캔 높이 표준편차로 평지/험지를 판정한다. 두 값 사이는 섞는다.
- `SWING_CLEARANCE_MIN/MAX` (0.08 / 0.20 m): 떠 있는 발이 지면 위 8~20 cm 사이에 있어야 한다.
- `PRISMATIC_*_GAIN`: 직동관절 사용 벌점의 세기. 평지 1.0(쓰지 마), 험지 0.12(써도 돼). **지금 남아 있는 지형 적응형 항은 이것뿐이다.**
- (삭제) `BASE_LEVEL_*_GAIN`: 둘 다 0.8이라 지형 적응 효과가 없었다(1장 1번 참고).

→ CPG2: 직동 사용 벌점과 스윙 높이 벌점은 쓰지 않는다.

### 4) 액추에이터 (하드웨어 사양 도출용)

| 이름 | 뜻 |
|---|---|
| `EFFORT_LIMIT_*` (500 N·m, 1500 N) | **일부러 실제보다 크게** 잡음. 잘리지 않은 진짜 필요 토크를 기록하려는 목적 |
| `VELOCITY_LIMIT_*` | 회전 5 rad/s, 직동 **0.25 m/s**(볼스크류) |
| `ARMATURE_*` | 모터 회전자 관성 × 감속비². 감속기 때문에 관절이 "무겁게" 느껴지는 효과. **추정값**이라 학습 때 ×0.5~1.5로 흔든다(`ARMATURE_RANDOM_SCALE`) |
| `STIFFNESS_*`, `DAMPING_*` | PD 게인. 토크 = Kp(목표 − 현재) − Kd·속도 |
| `RATED_*` | 연속 정격. 시뮬에 안 쓰이고, 토크 로그의 RMS를 비교하는 합격선 |
| `PAYLOAD_MASS_RANGE` | 탑재물 0~40 kg |
| `ROBOT_TOTAL_MASS`, `STATIC_TRIPOD_LOAD`, `FEET_FORCE_LIMIT` | (신규) 198.2 kg, 다리당 정적 하중 779 N, 충격 판정 기준 ≈ 1168 N |

### 5) 이름 패턴 (정규식)

- `FOOT_BODY_NAMES`: 발 6개
- `NON_FOOT_BODY_NAMES`: 발이 아닌 부위. 땅에 닿으면 벌점
- 관절 묶음: `ROLL`, `HIP_PITCH`, `KNEE_PITCH`, `PRISMATIC`, `REVOLUTE`(회전 18개), `ALL_CONTROLLED`(30개)

### 6) 보조 함수 (지형 읽기)

- **`_height_scan_values`**: 187개 점마다 "지면 높이 − 몸통 높이 + 기준값". 평지·정상 높이면 0이고, 앞에 턱이 있으면 +, 구덩이면 −. ±1로 자른다.
- **`_terrain_roughness_and_masks`**: 위 값의 표준편차가 거칠기다. 이걸로 `flat_weight`/`rough_weight`를 0~1로 만든다. **경사도 표준편차가 커서 험지로 판정된다**(CPG2 gait 선택은 그래서 평면을 빼고 계산).
- **`terrain_roughness_obs`**: 거칠기 숫자 하나를 관측으로 넘긴다.
- **`_nearest_terrain_height_from_scan`**: 발마다 가장 가까운 스캔 점의 지면 높이. 발 높이에서 빼면 발이 떠 있는 높이가 된다.

### 7) 지형 적응 보상 함수

| 함수 | 쉽게 말하면 |
|---|---|
| `terrain_swing_clearance_band_l2` | 떠 있는 발이 8 cm보다 낮거나(걸림) 20 cm보다 높으면(말 발차기) 벌점. 가장 가까운 스캔 점이 25 cm 넘게 멀면 무시 |
| `terrain_adaptive_prismatic_deviation_l2` | 직동관절이 기본값에서 **얼마나 벗어나 있나**² × (평지 1.0 / 험지 0.12). 쭉 뺀 채로 버티는 걸 막는다 |
| `terrain_adaptive_prismatic_velocity_l2` | 직동관절이 **얼마나 빨리 움직이나**² × 지형 gain. 덜덜 떠는 걸 막는다 |

deviation과 velocity는 막는 대상이 달라서, 순수 RL에서는 둘 다 의미가 있다.
→ CPG2: 경사에서 수평을 맞추려면 직동을 오래 빼고 있어야 해서 deviation은 목적과 충돌한다. 직동 residual은 action_l2가 이미 규제한다. 그래서 velocity만 0.25 m/s로 정규화해서 남겼다.

### 8) 접촉 함수

| 함수 | 쉽게 말하면 | CPG 이후 |
|---|---|---|
| `feet_air_time_reward` | 발이 0.3초 넘게 떠 있다 착지하면 초과 시간만큼 보상(발 끌기 방지) | 불필요. 스윙 타이밍은 CPG가 정하고, 이 보상은 CPG와 싸울 수 있음 |
| `feet_contact_force_l2` | 발 힘이 `FEET_FORCE_LIMIT`를 넘는 부분만 벌점 | 필요(착지 충격) |
| `undesired_body_contact` | 발이 아닌 부위가 10 N 넘게 닿은 개수 | 필요(계단에서 무릎·몸통 충돌) |
| `feet_contact_state` (관측) | 발 6개 접촉 여부, 최근 3스텝 중 최대 | **꼭 필요**. CPG가 "있어야 할 접촉"을 주고 센서가 "실제 접촉"을 준다. 이 차이(오르막에서 일찍 닿음, 내리막에서 늦게 닿음)가 residual이 지형에 적응하는 핵심 정보다 |
| (삭제) `feet_support_count` | 발 3개 이상 지지면 1 | CPG의 duty ≥ 0.5가 구조적으로 보장. 원래도 가중치 0 |

### 9) IMU 함수와 자세 함수

관측용:
- `imu_ang_vel_b`: 몸통 각속도
- `imu_projected_gravity_b`: 몸통 기준 중력 방향(기울기)
- `imu_lin_acc_residual_b`: 중력을 뺀 순수 가속도(g 단위)

보상용:
- **`imu_projected_gravity_xy_l2`** → `base_leveling` 항이 쓰는 함수.
  1. 몸통 기준 "아래" 단위벡터를 본다. 수평이면 (0, 0, −1)이다.
  2. 기울면 x, y 성분이 생긴다.
  3. gx² + gy² = **sin²(몸통이 수직에서 기운 총각도)**
  4. 가중치 −0.8 기준 초당 벌점: 5° −0.006, 10° −0.024, 20° −0.094. 속도 추종 +3/s에 비하면 약한 편이다. (CPG2는 −2.0)
- `imu_ang_vel_xy_l2`: 롤/피치 흔들림 벌점
- **`imu_vertical_dynamic_acc_l2`**: 위아래 출렁임 벌점(논문의 수직 진동 지표).
  1. IMU 가속도는 실제 가속도계처럼 중력이 섞여 나온다. 수평 정지 상태면 (0, 0, +9.81)이다.
  2. 여기에 `9.81 × projected_gravity`를 더하면 정지 상태에서 0이 된다. 즉 순수 운동 가속도만 남는다.
  3. 몸통 z 성분을 g로 나눠 제곱한다.
  4. 예: tripod 1.3 Hz로 ±1 cm 출렁이면 0.07 g → 제곱 0.0046 → × 0.02 = 초당 약 1e-4로, **벌점 크기만 보면 거의 0**이다. 실제 효과는 텐서보드 `Episode_Reward/imu_vertical_dynamic_acc_l2`로 확인할 것.
- `joint_deviation_l2`: 회전관절이 기본 자세에서 벗어난 정도. → CPG2에서는 뺐다(CPG 궤적 자체를 벌함).

### 10) Scene

- **`robot`**:
  - `max_depenetration_velocity 10`: 바닥에 박혔을 때 튕겨 나오는 최대 속도
  - `enabled_self_collisions False`: 다리끼리 충돌 계산 안 함(서로 통과 가능)
  - 솔버 반복 8/2
  - 액추에이터 4그룹: implicit PD로, PhysX 안에서 계산
- **`contact_forces`**: 발만. 최근 3스텝 기록, 체공 시간 측정
- **`body_contact_forces`**: 전신 접촉 감지
- **`imu`**: base_link. `gravity_bias` 때문에 실제 가속도계처럼 중력 포함 출력
- **`height_scanner`**: `ray_alignment="yaw"`. 몸통이 기울어도 레이는 수직 아래로 쏘고 방향만 따라간다.

### 11) Commands

목표 속도를 **5~10초마다 무작위로** 새로 뽑는다. vx 0.2~0.45, vy ±0.03, 회전 0.

- 왜 바꾸나: 한 에피소드 안에서 속도를 바꾸는 법을 배우게 하려고. 안 바꾸면 등속 보행만 배운다.
- 너무 자주 바꾸면: 정착하기 전에 또 바뀐다.
- 주기를 고정하면: 정책이 주기를 외운다.

→ CPG2: 전방향, 4~8초, 5% 정지 명령. gait 전환 2초를 거치고도 정착할 시간이 남는 값이다.

### 12) Actions (순수 RL용)

`목표 = 기본 자세 + scale × 정책 출력`. → CPG2는 `cpg_ik_action`으로 통째로 바꾼다.

### 13) Observations

| 항목 | 개수 |
|---|---|
| base_lin_vel | 3 |
| IMU 3종 | 9 |
| **height_scan** | **187** |
| roughness | 1 |
| feet_contact | 6 |
| commands | 3 |
| joint_pos / joint_vel | 60 |
| last action | 30 |
| **합계** | **299** (CPG2는 +18 = 317) |

- `enable_corruption False`: 관측 노이즈 없음
- `base_lin_vel`은 시뮬에서만 정확히 알 수 있다. 실로봇에서는 추정기가 필요하다.

### 14) Events (무작위화)

| 이벤트 | 시점 | 내용 |
|---|---|---|
| `reset_base` | 리셋 | 위치 ±5 cm, yaw ±0.05 |
| `reset_robot_joints` | 리셋 | 관절 ±0.01 |
| `randomize_robot_material` | 리셋 | 마찰 0.6~1.2 (→ CPG2: 0.4~1.2) |
| **`randomize_armature`** (신규) | 학습 시작 | armature × 0.5~1.5 |
| `randomize_base_mass` | 리셋 | 탑재물 +0~40 kg |
| `push_robot` | 4~8초마다 | ±40 N, ±8 N·m. 순간 충격보다는 다음 이벤트까지 유지되는 외력으로 이해할 것. PLAY/DEMO에서는 끔 |

### 15) Rewards

`RewardsCfg`에 적힌 weight는 초기값이고, 실제 값은 `__post_init__`에서 정한다. Isaac은 매 스텝 `weight × 함수값 × dt`를 더한다.
→ CPG2는 6그룹 체계로 다시 짰다(CPG2 파일 머리말 표).

### 16) Terminations

- `time_out` 20초: 실패가 아니라 시간 만료. 적당한 이유는 3장 Q&A 참고.
- `base_height`: 월드 z 0.45 m 미만이면 종료. → CPG2는 내리막 지형 오작동 때문에 지면 기준으로 바꿨다.
- `bad_orientation`: 1.0 rad(57°) 넘게 기울면 종료

### 17) `MultiLeggedRobotEnvCfg`

- `num_envs 4096`
- `__post_init__`: 시간 설정, 센서 갱신 주기, 명령 범위, 보상 가중치 최종값
- `gpu_max_rigid_patch_count 2**20`: 메쉬 지형 접촉 버퍼
- `viewer`: (신규) 로봇 하나를 따라가는 카메라. eye (3, 3, 2)는 그 로봇 기준 상대 좌표

---

## 3. Q&A

**Q. 높이 스캐너는 티처-스튜던트 학습의 티처 전용인가?**
꼭 그렇진 않다. 실로봇도 깊이 카메라나 LiDAR로 elevation map을 만들고 격자점 높이를 읽으면 같은 값이 된다(ANYmal 방식). `document/`의 "Probabilistic Terrain Mapping…"이 바로 그 elevation mapping이다. 다만 실제 지도에는 노이즈, 가림, 위치 오차가 있어서 이식할 때 둘 중 하나를 쓴다.
- (a) 학습 때 스캔에 노이즈를 넣는다.
- (b) 정확한 스캔은 티처만 쓰고, 스튜던트는 노이즈 낀 스캔이나 고유수용감각만 쓴다.

**Q. 스캐너를 따로 분리해서 관측 차원을 줄일 수 있나?**
지형 적응 보상 상수는 보상 계산에만 쓰여서 관측 차원과 무관하다. 차원을 키우는 건 `height_scan` 187개(전체의 약 60%)다. 보조 함수들은 계산만 공유하고, 계층으로 분리돼 있지는 않다. 이미 분리된 계층은 CPG2 gait 선택기 하나다(스캔에서 평면을 뺀 거칠기를 정책 밖에서 사용).

교수님 방향대로 나누면:

| 계층 | 내용 | 차원 |
|---|---|---|
| 지형 판단 | 거칠기, 평면 맞춤 경사(pitch/roll), 앞쪽 최대 단차 | 4~5 |
| 발 주변 높이 | 각 발 기준 위치 주변 몇 점 | 6~54 |

가장 싼 첫걸음은 **critic에만 원본 187개를 주는 것**(비대칭 actor-critic)이다. → 4장 미결정 사항.

**Q. armature가 추정값인데 넣는 게 나은가?**
넣는 게 낫다. 감속비가 크면 모터 관성이 감속비²로 커져서 링크 관성과 비슷하거나 더 크다(무릎 armature 0.95 vs 아랫다리 약 0.6 kg·m²). 0이면 관절이 비현실적으로 가볍고, 가속 토크가 빠져서 토크 로그가 작게 나온다. 정확한 값보다 크기 수준이 중요해서, 이번에 ×0.5~1.5 무작위화를 넣었다. 하모닉 + 웜기어로 가면 감속비가 바뀌니 값을 다시 잡아야 한다.

**Q. PD 게인 단위와 수준은?**
회전: Kp는 N·m/rad, Kd는 N·m·s/rad. 직동: Kp는 N/m, Kd는 N·s/m.

| 관절 | Kp | 오차당 힘 | 고유진동수 |
|---|---|---|---|
| roll | 2000 | 35 N·m/° | 약 3 Hz |
| hip | 4000 | 70 N·m/° | 약 3.7 Hz |
| knee | 2000 | 35 N·m/° | 약 5.7 Hz |
| prismatic | 50000 | 50 N/mm | 약 4.3 Hz |

Kd는 임계감쇠 2√(Kp·I)에 맞춰져 있다. 240 kg급으로는 약간 부드러운 편이다.

tripod 정적 하중(다리당 약 780 N)으로 보면:
- 직동: 관절당 약 1.4 cm, 직렬 두 개라 **약 2.8 cm 처짐**. 실제 셀프락 볼스크류보다 물렁하다.
- 무릎(CPG2 굽힌 자세): 780 N × 0.17 m ≈ **130 N·m**로, 정격 120을 정적 하중만으로 넘는다. → 4장.

**Q. 왜 timeout이 20초인가?**
- 할인율 0.99, 50 Hz면 정책이 내다보는 시간이 약 2초라, 20초는 이미 충분히 길다.
- 짧을수록 리셋이 잦아서 다양한 상태를 겪는다. legged_gym 계열 표준도 20초다.
- 커리큘럼 판정(8 m 패치의 절반 4 m = 0.2 m/s × 20초)과 맞물려 있어서, 바꾸면 이 기준도 같이 바꿔야 한다.
- 장시간 안정성은 평가로 확인한다(데모 71초).

---

## 4. 미결정 사항 (다음 단계)

1. **관측 차원 축소.** 비대칭 actor-critic → 스캐너 2계층 분리. 바꾸면 기존 체크포인트와 관측이 달라진다.
2. **무릎 토크.** CPG2의 무릎 굽힌 자세는 정적 130 N·m로 정격(120)을 넘는다. 결과로 보고하고 무릎 사양을 올릴지, `stance_height`를 높여 굽힘을 줄일지 정해야 한다.
3. **직동 강성.** 시뮬 Kp 50000 N/m에서 2.8 cm 처짐. 셀프락을 흉내 내려면 Kp를 높이는 것도 검토. 안정성은 armature 60 kg 기준으로 200 Hz에서 여유가 있다.
4. **`imu_vertical_dynamic_acc_l2` 가중치.** 현재 크기는 거의 0이다. 논문 주장과 맞는지 항별 보상 로그로 확인.
5. **액추에이터 확정 후 armature, Kp/Kd 재산정.** 하모닉 + 웜기어 감속비 기준.
