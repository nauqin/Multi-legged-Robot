"""CPG2: CPG + 수치 IK + RL residual 환경 (2026-11 최종 발표용).

v1 (multi_legged_robot_cpg_env_cfg.py) 은 비교용으로 그대로 둔다.
로봇 파라미터(URDF, 액추에이터)는 ver3.1 그대로다. 바뀐 것은 제어기와 학습 설정뿐.

==========================================================================
보상 체계  (Isaac Lab 은 weight x 값 x dt 를 더한다. 아래 '초당' 기준)
==========================================================================
설계 원칙
    - 양의 보상은 과제(속도 추종) 두 개뿐이다. 완벽하면 초당 +3.
    - 벌점은 목적별 5 그룹. 정상 보행에서 그룹 합이 과제 보상의 ~20% 를 넘지 않게
      가중치를 잡았다. 한 항이 혼자 지배하지 않게 하는 것이 목적이다
      (joint_torques_l2 가 추종 보상의 18 배였던 사례).
    - CPG 가 이미 보장하는 것(리듬, 발 들림, 지지 다리 수)은 보상에서 뺐다.
    - residual 은 'CPG 를 얼마나 고쳤나' 이므로 action_l2 하나로 규제한다.
      기본 자세 편차 벌점(joint_deviation)은 CPG 궤적 자체를 벌하므로 뺐다.

  그룹         항                         가중치    비고
  [A 과제]     track_lin_vel_xy           +2.0     exp(-e^2/0.02)
               track_ang_vel_z            +1.0     exp(-e^2/0.05)
  [B 안정]     base_leveling (중력 xy)     -2.0     몸통 수평 (P-R 레벨링의 목적)
               base_height (지면 기준)     -10.0    목표 0.74 m, 오차 3 cm -> -0.009
               imu_ang_vel_xy              -0.05
               imu_vertical_dynamic_acc    -0.02    수직 진동 (논문 지표)
  [C CPG]      action_l2 (residual 크기)   -0.01
               action_rate_l2              -0.01
               knee_extension              -2.0     굽힘 < 0.35 rad 일 때만
  [D 접촉]     feet_slip                   -0.5     디딘 발 미끄러짐
               feet_stumble                -0.5     턱에 걸림
               undesired_body_contact      -1.0
               feet_contact_force (>700N)  -0.02
  [E 안전]     joint_pos_limits            -1.0     prismatic 포함
               is_terminated               -10.0
  [F 하드웨어] joint_torques_l2            -2e-8    사양 도출용. 건드리지 말 것
               joint_acc_l2                -1e-7
               prismatic_vel (정규화)      -0.1     볼스크류 0.25 m/s

  뺀 항: is_alive, feet_air_time, support_contact_count, joint_deviation_l2,
         terrain_swing_clearance, terrain_adaptive_prismatic_deviation/velocity
         (마지막 둘은 정규화 prismatic_vel 과 residual 규제로 대체)

==========================================================================
환경 조건
==========================================================================
    지형 7 종: 평지, 랜덤 격자, 랜덤 요철, 오르막/내리막 경사, 오르막/내리막 계단
    명령: vx -0.25~0.35, vy +-0.15, wz +-0.35 (전방향), 5% 정지
    무작위화: 탑재물 0~40 kg, 마찰 0.4~1.2, 외력 +-40 N
    걸음새: 상위 규칙이 속도/거칠기로 고른다. 학습 중 명령이 바뀌면 전환도 일어난다.
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .cpg2_rewards import (
    base_height_terrain_below,
    base_height_terrain_l2,
    feet_slip_l2,
    feet_stumble,
    joint_vel_norm_l2,
    knee_extension_l2,
)
from .cpg_core import HugoPostureCfg, build_hugo
from .cpg_ik_action import CPGIKActionCfg, cpg_ik_gait_state, cpg_ik_oscillator_state
from .demo_command import ScriptedVelocityCommandCfg
from .multi_legged_robot_env_cfg import (
    ALL_CONTROLLED_JOINT_NAMES,
    FOOT_BODY_NAMES,
    KNEE_PITCH_JOINT_NAMES,
    PRISMATIC_JOINT_NAMES,
    VELOCITY_LIMIT_PRISMATIC,
    MultiLeggedRobotEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


##
# 튜닝 값
##

POSTURE = HugoPostureCfg(
    stance_height=0.72,      # hip -> 발 링크 원점. v1 0.84 -> 12 cm 낮춤
    roll_splay=0.20,
    prismatic_nominal=0.075,  # 행정 중앙: 위아래 +-0.07 m 레벨링 여유
    knee_sign=-1.0,           # Go2 식 (무릎이 뒤로)
)
FOOT_THICKNESS = 0.02
BASE_HEIGHT = POSTURE.stance_height + FOOT_THICKNESS          # 0.74 m
SPAWN_HEIGHT = BASE_HEIGHT + 0.03

# 기준 관절각은 IK 로 계산해 기본 자세로 쓴다 (joint_pos_rel 관측의 원점).
_, _, _NOMINAL_FEET, _Q_NOM = build_hugo(POSTURE)
Q_ROLL, Q_HIP, Q_PR1, Q_KNEE, Q_PR2 = [abs(v) if i == 0 else v for i, v in enumerate(_Q_NOM[0].tolist())]

RESIDUAL_SCALE_REVOLUTE = 0.10
RESIDUAL_SCALE_PRISMATIC = 0.04

CMD_LIN_VEL_X = (-0.25, 0.35)
CMD_LIN_VEL_Y = (-0.15, 0.15)
CMD_ANG_VEL_Z = (-0.35, 0.35)
CMD_RESAMPLING_S = (4.0, 8.0)
CMD_STANDING_RATIO = 0.05

TRACK_LIN_STD = math.sqrt(0.02)
TRACK_ANG_STD = math.sqrt(0.05)

# 지형 (난이도 0 -> 1 로 선형)
STEP_HEIGHT = (0.02, 0.10)      # 발 들림 0.08 + 무릎 residual
GRID_HEIGHT = (0.02, 0.10)
UNIFORM_NOISE = (0.0, 0.05)
SLOPE = (0.0, 0.35)             # tan -> 최대 약 19 도

CURRICULUM_UP_RATIO = 0.6
CURRICULUM_DOWN_RATIO = 0.3
CURRICULUM_MIN_EXPECTED = 1.0

PLAY_DIFFICULTY = float(os.environ.get("HUGO_PLAY_DIFFICULTY", "0.4"))
DEMO_TERRAIN = os.environ.get("HUGO_DEMO_TERRAIN", "mixed")


def make_terrains() -> dict:
    return {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.35, grid_height_range=GRID_HEIGHT, platform_width=2.0),
        "uneven": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=UNIFORM_NOISE, noise_step=0.01, border_width=0.25),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=SLOPE, platform_width=2.0, border_width=0.25),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=SLOPE, platform_width=2.0, border_width=0.25),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.175, step_height_range=STEP_HEIGHT, step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.175, step_height_range=STEP_HEIGHT, step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False),
    }


##
# Curriculum
##

def terrain_levels_cmd(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """명령이 요구한 거리 대비 실제 이동 거리로 레벨을 올리고 내린다.

    회전 명령이 있으면 실제 경로가 휘어 직선거리가 줄어드므로 v1 (0.8/0.5) 보다
    기준을 낮췄다 (0.6/0.3).
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    expected = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    half_tile = terrain.cfg.terrain_generator.size[0] / 2
    up_distance = torch.clamp(expected * CURRICULUM_UP_RATIO, max=half_tile)
    move_up = (distance > up_distance) & (expected > CURRICULUM_MIN_EXPECTED)
    move_down = (distance < expected * CURRICULUM_DOWN_RATIO) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


@configclass
class CPG2CurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_cmd)


##
# Actions
##

@configclass
class CPG2ActionsCfg:
    cpg = CPGIKActionCfg(
        asset_name="robot",
        posture=POSTURE,
        gait_mode="auto",
        command_name="base_velocity",
        roughness_sensor_name="height_scanner",
        residual_scale_revolute=RESIDUAL_SCALE_REVOLUTE,
        residual_scale_prismatic=RESIDUAL_SCALE_PRISMATIC,
    )


##
# Observations
##

@configclass
class CPG2PolicyCfg(ObservationsCfg.PolicyCfg):
    cpg_state = ObsTerm(func=cpg_ik_oscillator_state, params={"action_term_name": "cpg"})
    gait_state = ObsTerm(func=cpg_ik_gait_state, params={"action_term_name": "cpg"})


@configclass
class CPG2ObservationsCfg:
    policy: CPG2PolicyCfg = CPG2PolicyCfg()


##
# Rewards
##

_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)


@configclass
class CPG2RewardsCfg(RewardsCfg):
    base_height = RewTerm(
        func=base_height_terrain_l2, weight=-10.0,
        params={"target_height": BASE_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")})
    knee_extension = RewTerm(
        func=knee_extension_l2, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_PITCH_JOINT_NAMES),
                "min_bend": 0.35, "knee_sign": POSTURE.knee_sign})
    feet_slip = RewTerm(
        func=feet_slip_l2, weight=-0.5,
        params={"sensor_cfg": _FEET_CONTACT,
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES)})
    feet_stumble = RewTerm(func=feet_stumble, weight=-0.5, params={"sensor_cfg": _FEET_CONTACT})
    prismatic_vel = RewTerm(
        func=joint_vel_norm_l2, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=PRISMATIC_JOINT_NAMES),
                "vel_limit": VELOCITY_LIMIT_PRISMATIC})


@configclass
class CPG2TerminationsCfg(TerminationsCfg):
    pass


##
# Environment
##

@configclass
class MultiLeggedRobotCPG2EnvCfg(MultiLeggedRobotEnvCfg):
    observations: CPG2ObservationsCfg = CPG2ObservationsCfg()
    actions: CPG2ActionsCfg = CPG2ActionsCfg()
    rewards: CPG2RewardsCfg = CPG2RewardsCfg()
    terminations: CPG2TerminationsCfg = CPG2TerminationsCfg()
    curriculum: CPG2CurriculumCfg = CPG2CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- 1. 기본 자세 = IK 기준 자세 ---------------------------------
        self.scene.robot.init_state.joint_pos = {
            "L.*joint1_roll": Q_ROLL,
            "R.*joint1_roll": -Q_ROLL,
            ".*joint2_pitch": Q_HIP,
            ".*joint3_pitch": Q_KNEE,
            ".*prismatic1": Q_PR1,
            ".*prismatic2": Q_PR2,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
        self.observations.policy.height_scan.params["nominal_height"] = BASE_HEIGHT

        # ---- 2. 보상 체계 (머리말 표와 같다) -----------------------------
        r = self.rewards
        # [A] 과제
        r.track_lin_vel_xy.weight = 2.0
        r.track_lin_vel_xy.params["std"] = TRACK_LIN_STD
        r.track_ang_vel_z.weight = 1.0
        r.track_ang_vel_z.params["std"] = TRACK_ANG_STD
        # [B] 안정
        r.terrain_adaptive_base_leveling.weight = -2.0
        r.base_height.weight = -10.0
        r.imu_ang_vel_xy_l2.weight = -0.05
        r.imu_vertical_dynamic_acc_l2.weight = -0.02
        # [C] CPG 존중
        r.action_l2.weight = -0.01
        r.action_rate_l2.weight = -0.01
        r.knee_extension.weight = -2.0
        # [D] 접촉
        r.feet_slip.weight = -0.5
        r.feet_stumble.weight = -0.5
        r.undesired_body_contact.weight = -1.0
        r.feet_contact_force_l2.weight = -0.02
        # [E] 안전
        r.joint_pos_limits.weight = -1.0
        r.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES)
        r.is_terminated.weight = -10.0
        # [F] 하드웨어
        r.joint_torques_l2.weight = -2.0e-8
        r.joint_acc_l2.weight = -1.0e-7
        r.prismatic_vel.weight = -0.1
        # 뺀 항
        for name in ("is_alive", "feet_air_time", "support_contact_count", "imu_projected_gravity_xy_l2",
                     "joint_deviation_l2", "terrain_swing_clearance",
                     "terrain_adaptive_prismatic_deviation", "terrain_adaptive_prismatic_velocity"):
            setattr(r, name, None)

        # ---- 3. 종료: 월드 z 대신 지면 기준 높이 --------------------------
        self.terminations.base_height = DoneTerm(
            func=base_height_terrain_below,
            params={"minimum_height": 0.40, "sensor_cfg": SceneEntityCfg("height_scanner")})

        # ---- 4. 명령 (전방향) --------------------------------------------
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = CMD_LIN_VEL_X
        c.ranges.lin_vel_y = CMD_LIN_VEL_Y
        c.ranges.ang_vel_z = CMD_ANG_VEL_Z
        c.resampling_time_range = CMD_RESAMPLING_S
        c.rel_standing_envs = CMD_STANDING_RATIO

        # ---- 5. 지형 + 커리큘럼 ------------------------------------------
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = make_terrains()
        gen.curriculum = True
        gen.cache_dir = "/tmp/isaaclab/hugo_cpg2_terrains"
        self.scene.terrain.max_init_terrain_level = 1

        # ---- 6. 무작위화 -------------------------------------------------
        self.events.randomize_robot_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.randomize_robot_material.params["dynamic_friction_range"] = (0.4, 1.2)


@configclass
class MultiLeggedRobotCPG2EnvCfg_PLAY(MultiLeggedRobotCPG2EnvCfg):
    """평가: 무작위 명령, 난이도 고정 (HUGO_PLAY_DIFFICULTY), 외력 끔."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        gen = self.scene.terrain.terrain_generator
        gen.difficulty_range = (PLAY_DIFFICULTY, PLAY_DIFFICULTY)
        gen.use_cache = False
        self.events.push_robot = None


@configclass
class MultiLeggedRobotCPG2EnvCfg_DEMO(MultiLeggedRobotCPG2EnvCfg_PLAY):
    """발표 영상: 모든 env 가 같은 시나리오(demo_command.DEFAULT_SCHEDULE)를 따른다.

        HUGO_DEMO_TERRAIN=flat|grid|uneven|slope_up|slope_down|stairs_up|stairs_down|mixed
        HUGO_PLAY_DIFFICULTY=0.5
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = ScriptedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=old.ranges,
        )
        self.episode_length_s = self.commands.base_velocity.schedule[-1][0] + 1.0
        gen = self.scene.terrain.terrain_generator
        if DEMO_TERRAIN != "mixed":
            terr = make_terrains()
            only = terr[DEMO_TERRAIN]
            only.proportion = 1.0
            gen.sub_terrains = {DEMO_TERRAIN: only}
        # 시나리오는 넘어져도 끝까지 보여주도록 탑재물은 중간값으로 고정
        self.events.randomize_base_mass.params["mass_distribution_params"] = (20.0, 20.0)
