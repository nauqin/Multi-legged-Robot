"""CPG + RL residual 학습용 환경 설정.

놓을 위치:
    source/Multi_Legged_Robot/Multi_Legged_Robot/tasks/manager_based/
        multi_legged_robot/multi_legged_robot_cpg_env_cfg.py

원본 multi_legged_robot_env_cfg.py 는 건드리지 않는다.
상속해서 actions / observations / rewards / curriculum 만 갈아끼운다.
베이스라인(순수 RL)과 나란히 돌려 비교하기 위해서다.

--------------------------------------------------------------------------
12,300 iteration 험지 학습에서 확인된 것과 그에 따른 수정
--------------------------------------------------------------------------
[잘 된 것] CPG 자체는 설계대로 동작했다.
    - 여섯 다리 모두 0.550 Hz (설계 0.557)
    - tripod 위상 오차 최대 0.04 주기 (2%)
    - 베이스라인의 "세 다리 보행" 은 구조적으로 사라졌다

[문제 1] 5,200 iteration 에서 정책 붕괴.
    terrain_levels 가 2.5 까지 올라갔다가 급락 후 0.2 에 고착.
    커리큘럼이 너무 빨리 올라가 대량 실패 -> 큰 음의 보상 -> 붕괴.
    -> 계단 높이를 낮춘다. PPO 의 desired_kl 도 0.005 로 조일 것.

[문제 2] 제자리 회전.
    error_vel_yaw 0.835 rad/s 인데 track_ang_vel_z 가 0.0 이었다.
    커맨드는 "돌지 마라" 인데 아무도 안 말렸고, 제자리를 돌면 원점에서
    안 멀어지니 커리큘럼 레벨다운까지 피하는 도피 전략이 됐다.
    -> track_ang_vel_z 를 되살린다.

[문제 3] 주기적인 "펑" 튀어오름.
    속도 커맨드가 8 초마다 계단처럼 바뀌는데 d_step 이 즉시 반영했다.
    최악의 경우 보폭이 한 물리 스텝(5 ms)에 70 mm 점프 (발끝 14 m/s).
    -> cpg_action.py 의 d_step_tau 로 저역통과시킨다.
    -> 재샘플링 주기도 고정 8 초에서 (5, 10) 으로 흩는다.

[문제 4] prismatic 이 하한(0 m)에 3.2% 시간 붙어 있었다.
    h=0.80, k1=0.11 이면 CPG 최소 d 가 0.015 인데 residual 이 더해져
    음수가 됐다. 하드 스톱 충격이 몸통을 튀어오르게 한다.

[문제 5] 계단을 내려올 때 몸통을 수평으로 잡을 prismatic 예산이 없었다.
    d 중립 0.070 에 residual 0.02 -> 다리 길이 조절폭 +-0.04.
    계단 7 cm 를 보상하기에 부족하다.

    문제 4, 5 를 같이 푸는 값:  h = 0.84, k1 = 0.08
        d 중립 0.090,  최악의 경우에도 d_min 0.010 (하한 충돌 없음)
        레벨링 여유 +-0.08,  prismatic 속도는 한계의 56% (기존 77%)
"""

from __future__ import annotations

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
# CPG 파라미터
##

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

# 중립 자세에서의 prismatic 값. (0.84 - 0.66) / 2 = 0.09
CPG_PRISMATIC_NEUTRAL = (CPG_STANCE_HEIGHT - 0.66) / 2.0

# 스폰 높이. 발바닥이 hip 아래 (h + 발 두께 0.02) 에 오고, 3 cm 를 더 띄운다.
CPG_SPAWN_HEIGHT = CPG_STANCE_HEIGHT + 0.02 + 0.03


##
# Curriculum
##

def terrain_levels_vel(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """커맨드 속도 기준으로 지형 레벨을 올리고 내린다.

    한 에피소드에 지형 타일 절반 이상을 전진했으면 레벨업,
    커맨드 거리의 절반도 못 갔으면 레벨다운.

    제자리를 돌면 distance 가 작아 레벨다운 대상이 된다. 의도한 동작이며,
    track_ang_vel_z 와 함께 회전 도피 전략을 막는 두 번째 장치다.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (
        distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
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
        # 탐색 노이즈가 CPG 신호를 덮는다. 실측 roll 편차가 이미 0.47 까지
        # 났으므로 (PD 오버슈트) 더 키우지 말 것. 다리끼리 관통한다.
        residual_scale_revolute=0.08,
        # prismatic: 계단 7 cm 를 다리 길이로 보상하려면 관절당 0.04
        # (다리 길이 0.08) 가 필요하다. 0.02 로는 레벨링이 불가능했다.
        residual_scale_prismatic=0.04,
    )


##
# Observations
##

@configclass
class CPGPolicyCfg(ObservationsCfg.PolicyCfg):
    """기존 관측에 발진기 상태를 덧붙인다.

    이게 없으면 정책은 지금이 유각기인지 입각기인지 알 수 없다.
    보정을 언제 넣어야 할지 모르는 채로 학습하게 된다.
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
        # 남겨 두면 CPG 주기와 다른 주기를 유도해 서로 싸운다.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.support_contact_count.weight = 0.0

        # ---------------------------------------------------------------
        # 3. 자세 / 방향
        # ---------------------------------------------------------------
        # 회전 억제. 이걸 껐던 것이 제자리 회전의 직접 원인이었다.
        # CPG 가 회전을 "명령" 할 수 없는 것과, 회전을 "억제" 할 필요가
        # 없는 것은 다르다. residual 은 얼마든지 로봇을 돌릴 수 있다.
        self.rewards.track_ang_vel_z.weight = 1.0

        # 몸통 수평. prismatic 레벨링의 목적 함수다. 강하게 유지.
        self.rewards.terrain_adaptive_base_leveling.weight = -1.0

        # ---------------------------------------------------------------
        # 4. CPG 궤적과 충돌하는 항 -> 낮춘다
        # ---------------------------------------------------------------
        # 기본 자세 대비 편차 벌점. CPG 는 정의상 계속 편차를 만든다.
        self.rewards.joint_deviation_l2.weight = -0.01

        # 유각기 발 높이 밴드. CPG 가 swing_lift 로 이미 정한다.
        self.rewards.terrain_swing_clearance.weight = -0.10

        # prismatic 억제. 이 항이 -0.10 이면 "prismatic 쓰지 마라" 와
        # "몸통 수평 유지해라"(-1.0) 가 서로 모순된다. 레벨링이 목적이므로
        # 변위 벌점은 사실상 끄고, 속도 벌점만 남겨 급격한 펌핑을 막는다.
        self.rewards.terrain_adaptive_prismatic_deviation.weight = -0.01
        self.rewards.terrain_adaptive_prismatic_velocity.weight = -0.02

        # ---------------------------------------------------------------
        # 5. residual 을 작게 유지시키는 항
        # ---------------------------------------------------------------
        # 액션이 곧 residual 이므로 action_l2 는 "CPG 를 존중하라" 는 뜻이다.
        # 너무 크면 "아무것도 하지 마라" 가 되어 추종 보상을 압도한다.
        # (-0.02 로 두었을 때 정규화 합 -1.01 vs 추종 +0.83 이었다)
        self.rewards.action_l2.weight = -0.008
        self.rewards.action_rate_l2.weight = -0.008

        # 넘어지는 비용. -5.0 일 때는 사실상 없는 항이었다.
        self.rewards.is_terminated.weight = -20.0

        # 착지 충격. h 를 올렸으므로 낙하 높이가 커진다.
        self.rewards.feet_contact_force_l2.weight = -0.02

        # ---------------------------------------------------------------
        # 6. 그대로 두는 항
        # ---------------------------------------------------------------
        #   track_lin_vel_xy        3.0     주 목표
        #   undesired_body_contact  -1.0
        #   joint_torques_l2        -2e-8   하드웨어 사양 도출용. 건드리지 말 것
        #   joint_acc_l2            -1e-7
        #   joint_pos_limits        -0.25
        #   imu_ang_vel_xy_l2       -0.07

        # ---------------------------------------------------------------
        # 7. 커맨드
        # ---------------------------------------------------------------
        # v = 2 * d_step * f / beta,   f = 0.557 Hz,  beta = 0.5
        #   0.12 m/s -> d_step 0.054      0.28 m/s -> d_step 0.126
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.28)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.03, 0.03)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # 고정 8 초면 정책이 주기를 외운다. 흩어 놓는다.
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)

        # ---------------------------------------------------------------
        # 8. 지형 + 커리큘럼
        # ---------------------------------------------------------------
        # 발 들림 8 cm 에 맞춰 최대 높이를 제한한다.
        # 원래 계단 0.23 m 는 CPG 클리어런스로 넘을 수 없고,
        # 0.10 m 까지 올렸을 때 레벨 2.5 부근에서 정책이 붕괴했다.
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.07)
        gen.sub_terrains["random_grid_rough"].grid_height_range = (0.02, 0.07)
        gen.sub_terrains["pyramid_slope"].slope_range = (0.10, 0.30)
        gen.curriculum = True

        self.scene.terrain.max_init_terrain_level = 0


@configclass
class MultiLeggedRobotCPGEnvCfg_PLAY(MultiLeggedRobotCPGEnvCfg):
    """재생/평가용. 환경 수를 줄이고 무작위화를 끈다.

    지형은 그대로 둔다. 커리큘럼 함수가 terrain_generator 를 참조하므로
    여기서 평지로 바꾸면 None 참조로 죽는다.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.20)
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        # 평지만 보고 싶으면 0, 험지를 보고 싶으면 3 이상.
        self.scene.terrain.max_init_terrain_level = 3
