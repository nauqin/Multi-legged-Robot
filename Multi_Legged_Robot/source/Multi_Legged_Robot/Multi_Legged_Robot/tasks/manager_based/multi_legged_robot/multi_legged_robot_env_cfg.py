# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

BUMPS-TERRAIN training configuration (fluctuation measurement, rough condition).

Purpose of this version:
- adapt the flat anti-fluctuation policy (flat_antifluc2) to random bumpy
  terrain matching the evaluation condition (HfRandomUniformTerrain, +-mm scale)
- resume from flat_antifluc2 model_3000 (obs 231 / act 30 unchanged)
- rewards are the antifluc2 set; base_height_l2 now uses the height scanner
  so the height target follows the local ground instead of absolute world z

Differences vs. the flat config (each one is a lesson from the rough phase):
- terrain: plane -> bumps generator (train seed != eval seed, so the policy
  learns the terrain *type*, not the exact eval surface)
- contact_forces narrowed to referenced bodies only (full `Robot/.*` on a
  triangle mesh floods PhysX getMaterialFromInternalFaceIndex warnings and
  stalls the physics step)
- gpu_max_rigid_patch_count raised (default overflows on mesh terrain)
- base_height_l2 gets sensor_cfg (absolute z target is wrong on uneven ground)

Run (resume from flat_antifluc2):
    python scripts/rsl_rl/train.py --task Hugo-Hexapod-v0 --headless \
        --num_envs 2000 \
        --resume --load_run 2026-07-27_23-12-47_flat_antifluc2 \
        --checkpoint model_3000.pt \
        --run_name bumps_antifluc
    (mesh terrain + 4096 envs caused GPU OOM before; 2000 is the verified safe count)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# IsaacLab official MDP terms
import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
import Multi_Legged_Robot.tasks.manager_based.multi_legged_robot.mdp as hugo_mdp
from isaaclab.sensors import (
    ContactSensorCfg,
    ImuCfg,
    RayCasterCfg,
    patterns,
)

from isaaclab.terrains import TerrainImporterCfg


##
# Paths and design-level constants
##

ROBOT_USD_PATH = "/home/ubin/Hugo_Project/usd files/hugo_hexapod_ver2/hugo_hexapod_ver2.usd"

# 몸통 목표 높이. bumps에서는 sensor_cfg를 통해 "발밑 지면 대비" 높이로 해석된다.
TARGET_BODY_HEIGHT = 1.02

# 스폰 높이. use_terrain_origins=True라 각 패치 원점 위에서 스폰된다.
# bumps 요철이 최대 +-5cm 수준이므로 검증된 1.8을 그대로 유지.
INITIAL_BODY_HEIGHT = 1.8

# 50 Hz action period: sim.dt=1/200, decimation=4
SIM_DT = 1.0 / 200.0
DECIMATION = 4

# imu용 중력
GRAVITY_MAG = 9.81


##
# Bumps terrain settings (학습용)
##

# 학습 요철 진폭 [mm]. 측정(30mm)보다 약간 크게 잡아 여유를 둔다.
# 50mm 가혹 조건까지 측정할 계획이면 50으로 올려서 재학습.
TRAIN_BUMPS_AMPLITUDE_MM = 40.0

# 학습용 seed. 측정 지형(EVAL_SEED=20260725)과 반드시 달라야 한다 —
# 같으면 "그 지형을 외운" 정책이라는 지적을 피할 수 없다.
TRAIN_TERRAIN_SEED = 77

_AMP = TRAIN_BUMPS_AMPLITUDE_MM / 1000.0

# 학습용 지형: 측정과 같은 지형 *종류* (HfRandomUniformTerrain),
# 다른 seed. 커리큘럼 없음 (단일 난이도라 필요 없음).
# 패치는 4x8=32개, env들이 패치를 공유한다 (학습에서는 문제없음 —
# 1 env : 1 patch는 측정 전용 요구사항).
HUGO_BUMPS_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        seed=TRAIN_TERRAIN_SEED,
        curriculum=False,
        difficulty_range=(1.0, 1.0),
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=0.0,
        num_rows=4,
        num_cols=8,
        horizontal_scale=0.1,
        vertical_scale=0.002,   # 2mm 양자화 (측정 지형과 동일)
        slope_threshold=0.75,
        color_scheme="none",
        use_cache=False,        # 파라미터 바꿀 때 옛 지형 재사용 사고 방지
        sub_terrains={
            "bumps": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(-_AMP, _AMP),
                noise_step=max(_AMP / 5.0, 0.005),
                border_width=0.25,
            ),
        },
    ),
    use_terrain_origins=True,
    max_init_terrain_level=None,   # 난이도 1개뿐이라 무의미하지만 명시
    env_spacing=4.0,
    debug_vis=False,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75)),
)


##
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod (bumps terrain)."""

    terrain = HUGO_BUMPS_TERRAIN_IMPORTER_CFG

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,

            activate_contact_sensors=True,

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=10.0,
            ),

            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 이 로봇은 더미/구형 링크가 설계상 겹쳐 있어 self-collision을
                # 켜면 스폰 직후 뒤집힌다 (실험으로 확인). 반드시 False 유지.
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, INITIAL_BODY_HEIGHT),
            joint_pos={
                ".*joint1_roll": 0.0,
                ".*joint2_pitch": 0.0,
                ".*joint3_pitch": 0.0,
                ".*prismatic1": 0.0,
                ".*prismatic2": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=[".*prismatic.*"],
                stiffness=3000.0,
                damping=300.0,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
            ),

            "revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*joint.*"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
            ),
        },
    )

    # 삼각형 메시 지형이므로 참조되는 body만으로 범위 축소 필수.
    # `Robot/.*` 전체로 걸면 PhysX getMaterialFromInternalFaceIndex 경고가
    # 폭주해 물리 스텝이 로깅에 막힌다 (러프 단계에서 확인된 문제).
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*(feet|outside|inside|base).*",
        history_length=3,
        track_air_time=True,
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, GRAVITY_MAG),
    )

    # bumps는 요철이 +-5cm 이내라 평지형 스캐너 설정(오프셋 0.30, 5 m)으로 충분.
    # (20 m 오프셋은 계단/급경사에서 레이 시작점이 지형에 묻히는 것을 막는
    #  설정이었고, 이 지형에서는 불필요)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=SIM_DT * DECIMATION,
        history_length=1,
        debug_vis=False,
        mesh_prim_paths=["/World/ground/terrain"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(1.8, 1.2)),
        max_distance=5.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )


##
# Commands
##

@configclass
class CommandsCfg:
    """Command specifications."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.30, 0.70),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.15, 0.15),
        ),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications (30-dim: roll 6 + pitch 12 + prismatic 12)."""

    roll_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*roll"],
        scale=0.05,
        use_default_offset=True,
    )

    pitch_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*pitch"],
        scale=0.2,
        use_default_offset=True,
    )

    # bumps에서 발밑 요철을 prismatic으로 흡수하는 것이 이 로봇의 설계 의도.
    prismatic_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*prismatic.*"],
        scale=0.04,
        use_default_offset=True,
    )


##
# Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (231-dim, flat/rough 구성과 동일 — 체크포인트 호환)."""

        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # revolute joint states
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"])},
        )

        # prismatic joint states
        prismatic_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*prismatic.*"])},
        )
        prismatic_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*prismatic.*"])},
        )

        # imu
        imu_ang_vel_b = ObsTerm(
            func=hugo_mdp.imu_ang_vel_b,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )
        imu_projected_gravity_b = ObsTerm(
            func=hugo_mdp.imu_projected_gravity_b,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        # feet states
        feet_contact = ObsTerm(
            func=hugo_mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_feet"),
                "threshold": 1.0,
            },
        )
        feet_contact_force = ObsTerm(
            func=hugo_mdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_feet")},
        )

        # scan
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        # command
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # previous action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events / Domain randomization
##

@configclass
class EventCfg:
    """Event terms for reset/randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.01, 0.01),
        },
    )

    randomize_robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms — flat_antifluc2 세트를 그대로 유지.

    유일한 변경: base_height_l2에 sensor_cfg 추가.
    bumps에서는 절대 z 목표가 지면 요철만큼 오차를 내장하므로,
    height scanner 기준 상대 높이로 판정하도록 한다.
    """

    # alive / termination
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.05,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    # velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    # vertical fluctuation penalty (antifluc2 강화값)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.5,
    )

    # roll/pitch angular velocity penalty (antifluc2 강화값)
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.25,
    )

    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time,
        # 주의: 이 항은 이 로봇에서 조작 이력이 많음 (weight 1.0 실험 실패,
        # 장기 학습 시 호핑 유발 확인). 건드리지 말고 유지할 것.
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_feet"),
            "command_name": "base_velocity",
            "threshold": 1.0,
        },
    )

    undesired_contacts = RewTerm(
        func=locomotion_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(outside|inside|base).*"),
            "threshold": 1.0,
        },
    )

    # posture stability
    # bumps는 국소 요철이라 몸통은 계속 수평이 맞다 (경사와 다름) — -5.0 유지.
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
    )

    # smoothness
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.02,
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.001,
    )

    # energy-like terms
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
    )

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
        },
    )

    # anti-fluctuation: 발밑 지면 대비 몸통 높이 유지.
    # sensor_cfg가 있으면 base_height_l2는 스캔된 지면 높이만큼 목표를 보정한다
    # -> 측정 지표 z_rel과 정확히 같은 양을 최적화하게 된다.
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "target_height": TARGET_BODY_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # bumps 요철은 +-5cm 이내라 절대 z 기준 0.65도 오작동하지 않지만,
    # 러프 단계에서 만든 상대 높이 termination을 쓰는 것이 더 정확하다.
    base_height = DoneTerm(
        func=hugo_mdp.root_height_below_minimum_rel,
        params={
            "minimum_height": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Environment configuration
##

@configclass
class MultiLeggedRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration for Hugo hexapod (bumps)."""

    scene: MultiLeggedRobotSceneCfg = MultiLeggedRobotSceneCfg(
        num_envs=4096,   # 명령줄 --num_envs 2000 권장 (메시 지형 + 4096은 OOM 이력)
        env_spacing=4.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""

        # sim.dt = 1/200, decimation = 4 -> 50 Hz policy rate.
        self.decimation = DECIMATION
        self.episode_length_s = 20.0

        # simulation
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = (SIM_DT * DECIMATION)

        # --- PhysX GPU buffers ------------------------------------------
        # 삼각형 메시 지형 + 수천 env 접촉이면 기본값(5*2**15=163840)이 넘쳐
        # "Patch buffer overflow" 에러가 매 스텝 쏟아지고 iteration time이
        # 몇 배로 늘어난다 (러프 단계에서 확인). 또 터지면 2**21로.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        # viewer
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)