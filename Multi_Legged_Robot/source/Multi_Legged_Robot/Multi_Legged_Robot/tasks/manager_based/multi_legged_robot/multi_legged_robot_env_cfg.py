# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

This configuration is designed for the current early-stage simulation purpose:
- train Hugo P-R hexapod locomotion on the current mixed terrain.usd
- avoid sensors for now
- include implementable anti-fluctuation reward using root body height
- include basic domain randomization: friction, mass, external disturbance
- separate prismatic and revolute action spaces
"""

from __future__ import annotations

import math

import torch

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

TERRAIN_USD_PATH = "/home/ubin/Hugo_Project/usd files/terrain.usd"
ROBOT_URDF_PATH = "/home/ubin/Hugo_Project/usd files/hugo_hexapod.urdf"

# URDF 분석 기준:
# base_link에서 발바닥까지 약 1.02 m 정도이므로, 목표 몸체 높이를 1.05 m 근처로 둔다.
# 너무 높으면 다리가 공중에서 오래 떨어지고, 너무 낮으면 terrain에 파묻힐 수 있음.
TARGET_BODY_HEIGHT = 1.05

# 초기 spawn 높이.
# 복합 지형에서 약간의 요철/경사를 고려하여 목표 높이보다 조금 높게 시작.
INITIAL_BODY_HEIGHT = 1.75

# 현재 단계에서는 센서 없이 1차 locomotion 학습을 안정화하는 것이 목적.
# 계획서상 50 Hz action period에 맞추기 위해 sim.dt=1/200, decimation=4를 사용.
SIM_DT = 1.0 / 200.0
DECIMATION = 4


##
# Custom observation and reward terms
##

def base_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root body height z in world frame.

    Shape:
        (num_envs, 1)
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def body_height_error_obs(
    env,
    target_height: float = TARGET_BODY_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root body height error z_body - z_target.

    Shape:
        (num_envs, 1)
    """
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - target_height).unsqueeze(-1)


def body_height_error_l2(
    env,
    target_height: float = TARGET_BODY_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty term for anti-fluctuation.

    This implements:
        (z_body - z_target)^2

    Reward weight should be negative.
    """
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def body_height_abs_error(
    env,
    target_height: float = TARGET_BODY_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute body height error.

    This is useful as an auxiliary logged reward-like term.
    Reward weight should be negative and small if used.
    """
    asset = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)


##
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod."""

    # Current mixed terrain USD.
    # This file already contains flat, slope, rough, and stairs-like regions.
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TERRAIN_USD_PATH,
        ),
    )

    # Robot articulation from URDF.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=ROBOT_URDF_PATH,
            make_instanceable=True,
            fix_base=False,

            # Contact sensors are intentionally disabled for now.
            # ContactSensorCfg caused contact reporter API errors with the current URDF/USD conversion.
            activate_contact_sensors=False,

            # URDF importer requires joint_drive gains.
            # Actual PD-like behavior is handled by IsaacLab ImplicitActuatorCfg below.
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
                ".*prismatic1": 0.0,
                ".*prismatic2": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            # Prismatic axes: rough design-level actuator setting.
            # These values can be swept later for hardware specification feedback.
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=[".*prismatic.*"],
                stiffness=3000.0,
                damping=300.0,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
            ),

            # Revolute axes: rough design-level actuator setting.
            "revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*joint.*"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
            ),
        },
    )

    # Light
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
    """Command specifications for the MDP."""

    # Current phase:
    # start with slow velocity commands on mixed terrain.
    # If learning is stable, gradually increase these ranges later.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.3),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.2, 0.2),
        ),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Current URDF contains more than the final conceptual 12 action variables.
    Therefore, we use the current URDF joint structure directly.

    We split action terms by joint type:
    - prismatic axes: target linear position
    - revolute axes: target joint angle
    """

    prismatic_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*prismatic.*"],
        scale=0.00,
        use_default_offset=True,
    )

    revolute_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*joint.*"],
        scale=0.3,
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
        """Policy observations.

        Sensor-free first-stage policy:
        - base velocity
        - projected gravity
        - body height and height error
        - command
        - joint states
        - previous action
        """

        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # anti-fluctuation related state
        base_height = ObsTerm(
            func=base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        body_height_error = ObsTerm(
            func=body_height_error_obs,
            params={
                "target_height": TARGET_BODY_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # command
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # joint state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

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
    """Event terms for reset/randomization.

    Current implementable domain randomization:
    - initial base pose and velocity
    - initial joint offset
    - foot/body friction randomization
    - base mass randomization
    - external force/torque disturbance

    Sensor noise, communication delay, LiDAR/RGB-D noise, and detailed actuator model uncertainty
    are intentionally postponed.
    """

    # Initial base pose reset.
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

    # Weak joint reset randomization.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.01, 0.01),
        },
    )

    # Friction randomization.
    # Plan target range: friction coefficient 0.3 ~ 1.2.
    # Use a milder range first to avoid early instability on mixed terrain.
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

    # Base mass randomization.
    # Plan target: body/link mass ±15%.
    # Start with base_link only to reduce risk of unstable URDF conversion behavior.
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    # External disturbance.
    # Plan target: random force up to 50 N.
    # Use moderate disturbance at first; can increase to full range after stable walking.
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-30.0, 30.0),
            "torque_range": (-5.0, 5.0),
        },
    )


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for Hugo hexapod locomotion.

    Main objectives:
    1. velocity tracking
    2. anti-fluctuation via body height error
    3. posture stability
    4. smooth and energy-aware actuation
    """

    # alive / termination
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.1,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    # velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    # Anti-fluctuation: direct body height error penalty.
    # This is the currently implementable version of:
    #     r_height = -alpha * (z_body - z_target)^2
    body_height_error_l2 = RewTerm(
        func=body_height_error_l2,
        weight=-0.3,
        params={
            "target_height": TARGET_BODY_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Auxiliary anti-fluctuation term.
    # Penalizes absolute height deviation more directly.
    # Kept small to avoid overwhelming learning.
    body_height_abs_error = RewTerm(
        func=body_height_abs_error,
        weight=-0.2,
        params={
            "target_height": TARGET_BODY_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # vertical fluctuation penalty: suppress bouncing motion
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.25,
    )

    # roll/pitch angular velocity penalty
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # posture stability: keep body flat
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.25,
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

    # joint limit penalty
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
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

    # Body too low means collapse or severe terrain penetration.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Early debugging: keep this relatively loose.
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 1.5,
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