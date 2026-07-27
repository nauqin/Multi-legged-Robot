# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

FLAT-TERRAIN training configuration (fluctuation measurement baseline).

Purpose of this version:
- train flat walking with the full sensor/observation set (231-dim obs, 30-dim action)
- observation/action dimensions match the rough-terrain runs, so checkpoints
  from either side can be loaded across configs
- anti-fluctuation direct rewards (base_height_l2 etc.) are NOT included here;
  add them later and resume from a converged walking checkpoint
  (adding them to fresh training collapses into a standing-only policy —
  verified experimentally)

Notes:
- prismatic_pos action IS active (despite older comments saying otherwise).
- Curriculum is intentionally absent: plane terrain has no difficulty levels.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
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

# anti-fluctuation 단계에서 base_height_l2 보상을 추가할 때 사용.
# 지금은 정의만 해두고 보상에는 연결하지 않는다 (fresh 학습에 걸면 서 있기
# 국소해로 붕괴하는 것을 실험으로 확인함).
# 주의: 연결 전에 play로 잘 걷는 정책의 실제 몸통 높이를 재서 이 값이 맞는지
#       확인할 것. 자연 보행 높이와 어긋난 목표는 걸음을 뒤틀리게 한다.
TARGET_BODY_HEIGHT = 1.02

# 검증된 평지 spawn 높이.
INITIAL_BODY_HEIGHT = 1.8

# 50 Hz action period: sim.dt=1/200, decimation=4
SIM_DT = 1.0 / 200.0
DECIMATION = 4

# imu용 중력
GRAVITY_MAG = 9.81


##
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod (flat terrain)."""

    # 검증된 평지 설정: 원래 flat 학습(2033 iter, reward 59.76 수렴)과 동일.
    # physics_material을 명시해 마찰 결합 방식이 기본값으로 바뀌지 않게 한다
    # (로봇 쪽 마찰 랜덤화 0.6~1.2와의 결합이 검증된 run과 같아야 함).
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

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

    # 평지(해석적 평면)에서는 전 링크 접촉 센서가 문제를 일으키지 않는다.
    # (러프 삼각형 메시로 갈 때만 범위 축소 필요)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
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

    # 평지용 스캐너: 낮은 오프셋(0.30)과 5 m max_distance면 충분.
    # (20 m 오프셋은 계단/경사에서 시작점이 지형에 묻히는 것을 막기 위한
    #  러프 전용 설정이었음)
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

    # Roll 관절 (자세 유지)
    roll_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*roll"],
        scale=0.05,
        use_default_offset=True,
    )

    # Pitch 관절 (보행)
    pitch_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*pitch"],
        scale=0.2,
        use_default_offset=True,
    )

    # 활성 상태. anti-fluctuation 단계에서 높이 제어를 담당할 관절이므로 유지.
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
        """Policy observations (231-dim, rough-terrain 구성과 동일)."""

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
    """Reward terms (검증된 평지 세트).

    anti-fluctuation 직접 보상은 여기 넣지 않는다.
    걷기 수렴 후 base_height_l2를 추가하고 resume으로 다듬을 것.
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

    # vertical fluctuation penalty
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.5,
    )

    # roll/pitch angular velocity penalty
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

    # posture stability (평지 검증값)
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
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "target_height": TARGET_BODY_HEIGHT,   # ①에서 확인/수정한 값
            "asset_cfg": SceneEntityCfg("robot"),
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

    # 평지에서는 절대 z 기준이 정확함 (지면 z=0).
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.65,
            "asset_cfg": SceneEntityCfg("robot"),
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
    """Manager-based RL environment configuration for Hugo hexapod (flat)."""

    scene: MultiLeggedRobotSceneCfg = MultiLeggedRobotSceneCfg(
        num_envs=4096,
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

        # viewer
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)