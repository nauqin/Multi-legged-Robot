# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

Temporary revolute-only training configuration.

Purpose of this version:
- train basic flat/mild-terrain walking using revolute joints first
- keep prismatic joints at their default 0 position as much as possible
- remove direct anti-fluctuation rewards/observations for now
- keep simple posture/vertical velocity stabilization terms
- avoid contact sensors for now

Important:
- This cfg removes prismatic joints from the policy action space.
- For true prismatic locking, also modify the URDF prismatic joint limits to
  lower="0.0", upper="0.0" or lower="0.0", upper="0.001".
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


##
# Paths and design-level constants
##

TERRAIN_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/terrain.usd"
ROBOT_URDF_PATH = "/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod.urdf"

# Prismatic을 잠깐 사용하지 않는 revolute-only 단계에서는
# TARGET_BODY_HEIGHT 기반 anti-fluctuation 직접 보상은 제거한다.
# 나중에 prismatic/height control을 다시 사용할 때 아래 값을 복구하면 됨.
# TARGET_BODY_HEIGHT = 1.02

# 사용자가 확인한 것처럼 로봇이 terrain에 끼어 튕겨나가는 경우가 있어
# 초기 spawn 높이는 당분간 높게 유지한다.
# 단, 너무 높으면 낙하 충격이 커질 수 있으므로 안정화되면 1.20, 1.10 등으로 낮춰 실험 권장.
INITIAL_BODY_HEIGHT = 1.75

# 50 Hz action period: sim.dt=1/200, decimation=4
SIM_DT = 1.0 / 200.0
DECIMATION = 4


##
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod."""

    # Current mixed terrain USD.
    # If a pure flat terrain USD exists, use it here for first-stage walking training.
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TERRAIN_USD_PATH,
        ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=ROBOT_URDF_PATH,
            make_instanceable=True,
            fix_base=False,

            # Contact sensors are intentionally disabled for now.
            activate_contact_sensors=False,

            # URDF importer requires joint_drive gains.
            # Actual PD-like behavior is handled by ImplicitActuatorCfg below.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=10.0,
            ),

            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
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

                # Prismatic joints are initialized at zero.
                # For true locking, also lock them in the URDF joint limits.
                ".*prismatic1": 0.0,
                ".*prismatic2": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            # Keep prismatic actuator enabled to hold default position as much as possible.
            # However, prismatic joints are removed from the policy action space below.
            # Best practice: lock prismatic limits in URDF to 0.0~0.001 as well.
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=[".*prismatic.*"],
                stiffness=3000.0,
                damping=300.0,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
            ),

            # Revolute joints are the only joints controlled by the policy in this cfg.
            "revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*joint.*"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
            ),
        },
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
    """Command specifications for revolute-only walking."""

    # Standing command is disabled to avoid learning a standing-only policy.
    # Forward-only command is used to force basic walking attempts.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.15, 0.45),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.15, 0.15),
        ),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for revolute-only training.

    Prismatic action is temporarily disabled.
    Only roll/pitch revolute joints are controlled by the policy.
    """

    # Disabled for first-stage revolute-only walking.
    # prismatic_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*prismatic.*"],
    #     scale=0.04,
    #     use_default_offset=True,
    # )

    revolute_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*joint.*"],
        scale=0.15,
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
        """Policy observations for revolute-only walking.

        Removed for now:
        - base_height observation
        - body_height_error observation

        Kept:
        - base velocity
        - projected gravity
        - command
        - revolute joint states
        - previous action
        """

        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # command
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # revolute joint states only
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"])},
        )

        # previous action: now contains revolute action only
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
    """Event terms for reset/randomization.

    For first-stage revolute-only walking:
    - reset pose/velocity is kept mild
    - joint reset randomization is applied only to revolute joints
    - prismatic joints are not randomized
    - external push is disabled for now
    """

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

    # Apply reset randomization only to revolute joints.
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

    # Disabled for first-stage walking.
    # External disturbance can make the policy prefer standing/stabilization before it learns gait.
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (-30.0, 30.0),
    #         "torque_range": (-5.0, 5.0),
    #     },
    # )


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for first-stage revolute-only walking.

    Removed for now:
    - body_height_error_l2
    - body_height_abs_error

    Kept:
    - velocity tracking
    - posture stabilization
    - vertical velocity stabilization
    - smoothness/energy penalties
    """

    # alive / termination
    # Lower than 0.1 to reduce standing-only incentive.
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.05,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    # velocity tracking: strengthened to encourage forward walking attempts.
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

    # Direct anti-fluctuation rewards are temporarily disabled.
    # body_height_error_l2 = RewTerm(...)
    # body_height_abs_error = RewTerm(...)

    # vertical fluctuation penalty: suppress bouncing motion
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    # roll/pitch angular velocity penalty
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # posture stability: keep body reasonably flat
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
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

    # Joint limit penalty only for revolute joints.
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
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

    # Previous 0.35 was too low: the body could collapse without immediate termination.
    # Since spawn height is high, this only affects after the robot lands/settles.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.65,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Previous 1.5 rad was very loose. 1.0 rad is still permissive but terminates clear falls earlier.
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
    """Manager-based RL environment configuration for Hugo hexapod."""

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

        # Policy/control frequency:
        # sim.dt = 1/200, decimation = 4 -> 50 Hz policy rate.
        self.decimation = DECIMATION
        self.episode_length_s = 20.0

        # simulation
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation

        # viewer
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
