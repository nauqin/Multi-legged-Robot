"""CPG + RL residual 학습용 환경 설정.

놓을 위치:
    source/Multi_Legged_Robot/Multi_Legged_Robot/tasks/manager_based/
        multi_legged_robot/multi_legged_robot_cpg_env_cfg.py

원본 multi_legged_robot_env_cfg.py 는 건드리지 않는다.
상속해서 actions / observations / rewards / curriculum 만 갈아끼운다.
베이스라인(순수 RL)과 나란히 돌려 비교하기 위해서다.

값이 정해지는 순서 (뒤가 앞을 덮어쓴다):
    기반 cfg __post_init__ -> 이 파일의 __post_init__ -> PLAY __post_init__
    -> play.py -> CLI 인자
이 task 의 튜닝 값은 아래 "튜닝 값" 블록에 모아 둔다.

--------------------------------------------------------------------------
12,300 iteration 험지 학습에서 확인된 것과 그에 따른 수정
--------------------------------------------------------------------------
[잘 된 것] CPG 자체는 설계대로 동작했다.
    - 여섯 다리 모두 0.550 Hz (설계 0.557)
    - tripod 위상 오차 최대 0.04 주기 (2%)
    - 베이스라인의 "세 다리 보행" 은 구조적으로 사라졌다

[문제 1] 5,200 iteration 에서 정책 붕괴.
    terrain_levels 가 2.5 까지 올라갔다가 급락 후 0.2 에 고착.
    -> 계단 높이를 낮춘다. PPO 의 desired_kl 도 0.005 로 조일 것.

[문제 2] 제자리 회전.  -> track_ang_vel_z 를 되살린다.

[문제 3] 주기적인 "펑" 튀어오름 (커맨드 재샘플링 시 보폭 점프).
    -> cpg_action.py 의 d_step_tau 로 저역통과, 재샘플링 (5, 10) 초.

[문제 4, 5] prismatic 하한 충돌 / 레벨링 예산 부족.
    -> h = 0.84, k1 = 0.08, residual_scale_prismatic = 0.04
       (단, |action| <= 1 일 때만 성립한다. 문제 6 참고)

--------------------------------------------------------------------------
8,000 iteration 험지 학습 (2026-09-15_18-55-04) 에서 확인된 것과 수정
--------------------------------------------------------------------------
[문제 6] 탐색 노이즈 폭주 -> 두 번의 붕괴 (1,200~2,000 / 4,000~4,300).
    Loss/entropy 가 22 -> 63. 행동 표준편차로 환산하면 0.5 -> 약 2.0.
    residual 스케일이 작아서 노이즈가 보상에 별 영향을 주지 않으니,
    엔트로피 보너스만 남아 표준편차를 계속 키웠다.
    게다가 액션 클리핑이 없어 residual 이 설계 한도(스케일 x 1)를 넘었고,
    그러면 prismatic 이 하한 0 m 에 다시 닿는다.
    -> rsl_rl_ppo_cfg.py: entropy_coef 0, clip_actions 1.0

[문제 7] 멈춰 서 있어도 속도 추종 보상의 85% 를 받았다.
    std = sqrt(0.25) 는 커맨드 0.12~0.28 m/s 에 비해 너무 넓다.
    발이 걸려 멈춰도 손해가 거의 없으니 빠져나오려는 동기가 약하다.
    -> std = sqrt(0.02). 멈추면 13%, 오차 0.1 m/s 면 61%.

[문제 8] 커리큘럼 레벨업 기준(4 m 고정)을 느린 커맨드는 넘을 수 없었다.
    0.20 m/s 미만은 20 초에 4 m 를 갈 수 없다. 최종 평균 레벨 1.6 / 7.
    -> 커맨드가 요구한 거리 대비 비율로 판정한다.

[문제 9] 평가 지형 난이도가 무작위였다.
    play.py 는 행 수를 num_envs / 16 으로 줄인다. 16 대면 1 행이라
    패치마다 난이도가 0~1 에서 무작위로 정해져, 학습 때 못 가 본
    난이도까지 섞였다.
    -> PLAY 에서 difficulty_range 를 고정한다 (HUGO_PLAY_DIFFICULTY).
       외력 이벤트(push_robot)도 평가에서는 끈다.
"""

from __future__ import annotations

import math
import os

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .cpg_action import CPGActionCfg, cpg_oscillator_state
from .multi_legged_robot_env_cfg import (
    MultiLeggedRobotEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    ROLL_SPLAY,
)


##
# 튜닝 값
##

# --- CPG ------------------------------------------------------------------
# 입각기 다리 길이 (hip -> 발 링크 원점).
# 제약:  h - k1 - D_level >= L0 = 0.66     (prismatic 은 신장만 된다)
#        h + D_level      <= 0.96
CPG_STANCE_HEIGHT = 0.84
CPG_SWING_LIFT = 0.08
CPG_STANCE_PUSH = 0.008

# 0.557 Hz. 다리 1개가 26.7 kg 이라 스윙 관성이 크고,
# prismatic 속도 한계 0.25 m/s 가 상시 제약이다 (이 값에서 56% 사용).
CPG_OMEGA = 3.5

CPG_ALPHA = 50.0        # 2 * alpha * mu^2 * dt = 0.5  (dt = 1/200)
CPG_COUPLING = 0.5      # 크면 진폭이 mu 보다 부풀어 보폭이 의도보다 커진다

# residual 스케일. 정책 출력 1.0 이 이만큼의 보정이다.
# 체크포인트를 이어 학습할 때는 바꾸지 말 것 (출력의 의미가 달라진다).
RESIDUAL_SCALE_REVOLUTE = 0.08   # [rad]
RESIDUAL_SCALE_PRISMATIC = 0.04  # [m]

# 중립 자세에서의 prismatic 값. (0.84 - 0.66) / 2 = 0.09
CPG_PRISMATIC_NEUTRAL = (CPG_STANCE_HEIGHT - 0.66) / 2.0

# 스폰 높이. 발바닥이 hip 아래 (h + 발 두께 0.02) 에 오고, 3 cm 를 더 띄운다.
CPG_SPAWN_HEIGHT = CPG_STANCE_HEIGHT + 0.02 + 0.03

# --- 커맨드 ---------------------------------------------------------------
# v = 2 * d_step * f / beta,   f = 0.557 Hz,  beta = 0.5
#   0.12 m/s -> d_step 0.054      0.28 m/s -> d_step 0.126
CMD_LIN_VEL_X = (0.12, 0.28)
CMD_LIN_VEL_Y = (-0.03, 0.03)
CMD_RESAMPLING_S = (5.0, 10.0)   # 고정 주기면 정책이 주기를 외운다

# --- 속도 추종 보상 폭 (exp(-err^2 / std^2)) ------------------------------
# sqrt(0.25) 일 때 커맨드 0.2 m/s 에서 멈춰 있어도 85% 를 받았다.
#   lin sqrt(0.02): 정지 13%, 오차 0.10 -> 61%, 오차 0.05 -> 88%
#   yaw sqrt(0.10): 오차 0.3 rad/s -> 41%  (tripod 고유의 흔들림을 감안해 완만하게)
TRACK_LIN_STD = math.sqrt(0.02)
TRACK_ANG_STD = math.sqrt(0.10)

# --- 지형 (CPG 발 들림 8 cm 에 맞춤) --------------------------------------
STAIRS_HEIGHT_RANGE = (0.02, 0.07)
GRID_HEIGHT_RANGE = (0.02, 0.07)
# 하한은 0 근처여야 한다. 0.10 이면 가장 쉬운 행도 약 6도 경사다.
SLOPE_RANGE = (0.0, 0.30)

# --- 커리큘럼 -------------------------------------------------------------
# 커맨드가 요구한 거리(|cmd| x 에피소드 길이) 대비 비율로 판정한다.
CURRICULUM_UP_RATIO = 0.8
CURRICULUM_DOWN_RATIO = 0.5
# 요구 거리가 이보다 짧은 env (정지 커맨드) 는 레벨업 대상에서 뺀다. [m]
CURRICULUM_MIN_EXPECTED = 1.0

# --- 평가 -----------------------------------------------------------------
# 평가 지형 난이도 (0~1). 학습 레벨 L (0~7) 은 대략 difficulty = L / 8.
#   HUGO_PLAY_DIFFICULTY=0.4 python scripts/rsl_rl/play.py ...
PLAY_DIFFICULTY = float(os.environ.get("HUGO_PLAY_DIFFICULTY", "0.2"))


##
# Curriculum
##

def terrain_levels_vel(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """커맨드가 요구한 거리 대비 실제 이동 거리로 지형 레벨을 올리고 내린다.

    레벨업:   요구 거리의 80% 이상 (단, 타일 절반 4 m 를 넘을 필요는 없다)
    레벨다운: 요구 거리의 50% 미만 (넘어져 일찍 끝난 에피소드도 여기 해당)

    예전에는 레벨업 기준이 4 m 고정이라 0.20 m/s 미만 커맨드는
    구조적으로 레벨업이 불가능했다.

    제자리를 돌면 distance 가 작아 레벨다운 대상이 된다. 의도한 동작이며,
    track_ang_vel_z 와 함께 회전 도피 전략을 막는 두 번째 장치다.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    expected = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s

    half_tile = terrain.cfg.terrain_generator.size[0] / 2
    up_distance = torch.clamp(expected * CURRICULUM_UP_RATIO, max=half_tile)

    move_up = (distance > up_distance) & (expected > CURRICULUM_MIN_EXPECTED)
    move_down = distance < expected * CURRICULUM_DOWN_RATIO
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class CPGCurriculumCfg:
    """잘 걷는 환경만 험지 레벨이 올라간다. 평지에서 시작한다."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


##
# Actions
##

@configclass
class CPGActionsCfg:
    """액션 항이 하나뿐이다. 30차원 residual 을 받는다.

    순서는 cpg_action.py 의 _residual_ids 와 같다:
        roll 6 -> hip_pitch 6 -> knee_pitch 6 -> prismatic1 6 -> prismatic2 6
    다리 순서는 R1 R2 R3 L1 L2 L3.

    CPGActionCfg 의 기본값(h 0.76, omega 6.3 등)은 초기 실험값이다.
    실제로 쓰이는 값은 여기서 넣는 값이다.
    """

    cpg = CPGActionCfg(
        asset_name="robot",

        gait="tripod",
        omega=CPG_OMEGA,
        alpha=CPG_ALPHA,
        coupling=CPG_COUPLING,

        stance_height=CPG_STANCE_HEIGHT,
        swing_lift=CPG_SWING_LIFT,
        stance_push=CPG_STANCE_PUSH,
        roll_nominal=ROLL_SPLAY,

        leg_length_min=0.66,
        prismatic_stroke=0.15,

        use_command_speed=True,
        command_name="base_velocity",
        d_step_max=0.20,
        # 커맨드가 계단처럼 바뀌어도 보폭은 부드럽게 따라가게 한다.
        # 이게 없으면 재샘플링 순간마다 다리가 채찍처럼 휘둘린다.
        d_step_tau=0.3,

        enable_cpg=True,
        enable_residual=True,
        # 회전 관절: CPG 가 hip pitch 를 약 0.12 rad 쓴다. 스케일이 크면
        # 탐색 노이즈가 CPG 신호를 덮는다. 다리끼리 관통하므로 키우지 말 것.
        residual_scale_revolute=RESIDUAL_SCALE_REVOLUTE,
        # prismatic: 계단 7 cm 를 다리 길이로 보상하려면 관절당 0.04
        # (다리 길이 0.08) 가 필요하다.
        residual_scale_prismatic=RESIDUAL_SCALE_PRISMATIC,
    )


##
# Observations
##

@configclass
class CPGPolicyCfg(ObservationsCfg.PolicyCfg):
    """기존 관측에 발진기 상태를 덧붙인다.

    이게 없으면 정책은 지금이 유각기인지 입각기인지 알 수 없다.
    """

    cpg_state = ObsTerm(
        func=cpg_oscillator_state,
        params={"action_term_name": "cpg"},
    )


@configclass
class CPGObservationsCfg:
    policy: CPGPolicyCfg = CPGPolicyCfg()


##
# Rewards
##

@configclass
class CPGRewardsCfg(RewardsCfg):
    """항 구성은 그대로 두고 가중치만 env cfg 의 __post_init__ 에서 바꾼다.

    여기서 바꾸지 않는 이유는 부모 __post_init__ 이 나중에 실행되면서
    값을 되돌려 놓기 때문이다.
    """

    pass


##
# Environment
##

@configclass
class MultiLeggedRobotCPGEnvCfg(MultiLeggedRobotEnvCfg):
    """CPG 가 기본 보행을, RL 이 지형 보정을 담당하는 환경."""

    observations: CPGObservationsCfg = CPGObservationsCfg()
    actions: CPGActionsCfg = CPGActionsCfg()
    rewards: CPGRewardsCfg = CPGRewardsCfg()
    curriculum: CPGCurriculumCfg = CPGCurriculumCfg()

    def __post_init__(self) -> None:
        # 부모가 시뮬 설정, 커맨드 범위, 보상 가중치를 모두 세팅한다.
        # 그 다음에 CPG 용으로 덮어써야 한다. 순서가 중요하다.
        super().__post_init__()

        # ---------------------------------------------------------------
        # 1. 기본 자세를 CPG 중립 자세에 맞춘다
        # ---------------------------------------------------------------
        # 맞춰 두지 않으면 joint_deviation_l2 가 CPG 궤적 자체를 벌주고,
        # 리셋 직후 차이만큼 밀어내느라 과도구간이 생긴다.
        self.scene.robot.init_state.joint_pos = {
            "L.*joint1_roll": ROLL_SPLAY,
            "R.*joint1_roll": -ROLL_SPLAY,
            ".*joint2_pitch": 0.0,
            ".*joint3_pitch": 0.0,
            ".*prismatic1": CPG_PRISMATIC_NEUTRAL,
            ".*prismatic2": CPG_PRISMATIC_NEUTRAL,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, CPG_SPAWN_HEIGHT)

        # ---------------------------------------------------------------
        # 2. 리듬을 만들게 하던 항 -> 끈다
        # ---------------------------------------------------------------
        # CPG 가 유각/입각 주기를 구조적으로 보장하므로 불필요하다.
        self.rewards.feet_air_time.weight = 0.0

        # ---------------------------------------------------------------
        # 3. 속도 추종 / 자세
        # ---------------------------------------------------------------
        self.rewards.track_lin_vel_xy.weight = 3.0
        # 멈춰 서 있는 것을 확실히 손해로 만든다. [문제 7]
        self.rewards.track_lin_vel_xy.params["std"] = TRACK_LIN_STD

        # 회전 억제. 이걸 껐던 것이 제자리 회전의 직접 원인이었다.
        self.rewards.track_ang_vel_z.weight = 1.0
        self.rewards.track_ang_vel_z.params["std"] = TRACK_ANG_STD

        # 몸통 수평. prismatic 레벨링의 목적 함수다. 강하게 유지.
        # (2026-11) 이름 변경: terrain_adaptive_base_leveling -> base_leveling.
        # 예전 값 -1.0 x gain 0.8 과 같은 -0.8.
        self.rewards.base_leveling.weight = -0.8

        # ---------------------------------------------------------------
        # 4. CPG 궤적과 충돌하는 항 -> 낮춘다
        # ---------------------------------------------------------------
        # 기본 자세 대비 편차 벌점. CPG 는 정의상 계속 편차를 만든다.
        self.rewards.joint_deviation_l2.weight = -0.01

        # 유각기 발 높이 밴드. CPG 가 swing_lift 로 이미 정한다.
        self.rewards.terrain_swing_clearance.weight = -0.10

        # prismatic 억제. 레벨링이 목적이므로 변위 벌점은 사실상 끄고,
        # 속도 벌점만 남겨 급격한 펌핑을 막는다.
        self.rewards.terrain_adaptive_prismatic_deviation.weight = -0.01
        self.rewards.terrain_adaptive_prismatic_velocity.weight = -0.02

        # ---------------------------------------------------------------
        # 5. residual 을 작게 유지시키는 항
        # ---------------------------------------------------------------
        # 액션이 곧 residual 이므로 action_l2 는 "CPG 를 존중하라" 는 뜻이다.
        # 너무 크면 "아무것도 하지 마라" 가 되어 추종 보상을 압도한다.
        self.rewards.action_l2.weight = -0.008
        self.rewards.action_rate_l2.weight = -0.008

        # 넘어지는 비용. -5.0 일 때는 사실상 없는 항이었다.
        self.rewards.is_terminated.weight = -20.0

        # 착지 충격. h 를 올렸으므로 낙하 높이가 커진다.
        self.rewards.feet_contact_force_l2.weight = -0.02

        # ---------------------------------------------------------------
        # 6. 부모 값 그대로 두는 항 (참고용)
        # ---------------------------------------------------------------
        #   undesired_body_contact       -1.0
        #   joint_torques_l2             -2e-8   하드웨어 사양 도출용. 건드리지 말 것
        #   joint_acc_l2                 -1e-7
        #   joint_pos_limits             -0.25
        #   imu_ang_vel_xy_l2            -0.07
        #   imu_vertical_dynamic_acc_l2  -0.02
        #   imu_projected_gravity_xy_l2   0.0    base_leveling 으로 대체
        #   is_alive                     부모 값

        # ---------------------------------------------------------------
        # 7. 커맨드
        # ---------------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = CMD_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = CMD_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = CMD_RESAMPLING_S

        # ---------------------------------------------------------------
        # 8. 지형 + 커리큘럼
        # ---------------------------------------------------------------
        # random_grid_terrain_cfg.py 의 높이 값은 여기서 덮어쓴다.
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains["pyramid_stairs"].step_height_range = STAIRS_HEIGHT_RANGE
        gen.sub_terrains["random_grid_rough"].grid_height_range = GRID_HEIGHT_RANGE
        gen.sub_terrains["pyramid_slope"].slope_range = SLOPE_RANGE
        gen.curriculum = True

        self.scene.terrain.max_init_terrain_level = 0


@configclass
class MultiLeggedRobotCPGEnvCfg_PLAY(MultiLeggedRobotCPGEnvCfg):
    """재생/평가용.

    지형 종류는 그대로 두고 난이도만 고정한다. 커리큘럼 함수가
    terrain_generator 를 참조하므로 평지로 바꾸면 None 참조로 죽는다.

    play.py 가 지형 격자 크기, 패치 배치, 커리큘럼 해제를 담당하므로
    max_init_terrain_level 은 여기서 설정하지 않는다 (설정해도 무시된다).

    에피소드는 20 초라 모든 로봇이 20 초마다 동시에 리셋(재스폰)된다.
    그 순간의 튀어오름은 평가 아티팩트이니 정책 문제와 구분할 것.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.20)
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)

        # 난이도 고정. play.py 가 행 수를 줄여도 모든 패치가 같은 난이도가 된다.
        gen = self.scene.terrain.terrain_generator
        gen.difficulty_range = (PLAY_DIFFICULTY, PLAY_DIFFICULTY)
        gen.use_cache = False

        # 평가 중 외력은 끈다. 넣으면 지형 대응과 외란 대응이 섞인다.
        if getattr(self.events, "push_robot", None) is not None:
            self.events.push_robot = None
