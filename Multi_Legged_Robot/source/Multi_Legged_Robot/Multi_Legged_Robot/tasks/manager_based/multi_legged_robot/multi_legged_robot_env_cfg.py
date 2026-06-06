# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

USD-based revolute-only training configuration with contact sensors and IMU.

Current purpose:
- load robot from verified USD file
- enable contact sensors
- enable IMU sensor on base_link
- train basic walking using revolute joints first
- keep prismatic joints near zero
- keep IMU out of observations/rewards until sensor values are verified
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
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp


##
# Paths and constants
##

TERRAIN_MODE = "flat"  # "flat" or "mixed"

TERRAIN_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/terrain.usd"
ROBOT_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod_ver2/hugo_hexapod_ver2.usd"

INITIAL_BODY_HEIGHT = 1.75

SIM_DT = 1.0 / 200.0
DECIMATION = 4


##
# Body name patterns
##

FOOT_BODY_NAMES = [".*_feet"]

NON_FOOT_BODY_NAMES = [
    "base_link",
    ".*_hip_dummy",
    ".*_sphere1",
    ".*_inside",
    ".*_outside",
    ".*_sphere2_base",
    ".*_sphere2",
    ".*_inside2",
    ".*_outside2",
]


##
# Custom reward functions
##

def feet_support_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    min_contacts: int = 3,
):
    contact_sensor = env.scene[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    contacts = force_norm > threshold
    num_contacts = torch.sum(contacts, dim=1)

    return (num_contacts >= min_contacts).float()


def feet_air_time_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.35,
    command_threshold: float = 0.1,
):
    contact_sensor = env.scene[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time_reward = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(air_time_reward * first_contact.float(), dim=1)

    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.norm(command[:, :2], dim=1)
    reward *= command_xy_norm > command_threshold

    return reward


def feet_contact_force_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 600.0,
):
    contact_sensor = env.scene[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    excess_force = torch.clamp(force_norm - max_force, min=0.0)
    penalty = torch.mean((excess_force / max_force) ** 2, dim=1)

    return penalty


def undesired_body_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 10.0,
):
    contact_sensor = env.scene[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    contacts = force_norm > threshold
    contact_count = torch.sum(contacts.float(), dim=1)

    return contact_count


def undesired_body_contact_force_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 100.0,
):
    contact_sensor = env.scene[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    excess_force = torch.clamp(force_norm - max_force, min=0.0)
    penalty = torch.mean((excess_force / max_force) ** 2, dim=1)

    return penalty


def joint_deviation_l2(
    env,
    asset_cfg: SceneEntityCfg,
):
    asset = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.mean((joint_pos - default_joint_pos) ** 2, dim=1)


def contact_force_distribution(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    max_force: float = 800.0,
):
    contact_sensor = env.scene[sensor_cfg.name]

    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)
    force_norm = torch.clamp(force_norm, max=max_force)

    contacts = force_norm > threshold
    contact_count = torch.sum(contacts, dim=1).clamp(min=1)

    masked_force = force_norm * contacts.float()
    mean_force = torch.sum(masked_force, dim=1, keepdim=True) / contact_count.unsqueeze(-1)

    variance = torch.sum(((masked_force - mean_force) * contacts.float()) ** 2, dim=1) / contact_count
    normalized_variance = variance / (mean_force.squeeze(-1).clamp(min=1.0) ** 2)

    return normalized_variance


##
# Scene
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod."""

    if TERRAIN_MODE == "flat":
        terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

    elif TERRAIN_MODE == "mixed":
        terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.UsdFileCfg(
                usd_path=TERRAIN_USD_PATH,
            ),
        )

    else:
        raise ValueError(f"Unknown TERRAIN_MODE: {TERRAIN_MODE}. Use 'flat' or 'mixed'.")

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # If the robot explodes immediately, set this back to False.
                enabled_self_collisions=False, # 무조건 False로 설정할 것. 그렇지 않으면 충돌 발생
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

    # Foot contact sensor.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_feet",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    # Full-body contact sensor.
    body_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
    )

    # IMU sensor attached to base_link.
    #
    # The sensor is attached to the base rigid body.
    # For now, it is only added to the scene and is not used in observations/rewards.
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
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
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*joint.*"],
                )
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*joint.*"],
                )
            },
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##

@configclass
class EventCfg:
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
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.05,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

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

    feet_air_time = RewTerm(
        func=feet_air_time_reward,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.35,
            "command_threshold": 0.1,
        },
    )

    support_contact_count = RewTerm(
        func=feet_support_count,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 5.0,
            "min_contacts": 3,
        },
    )

    feet_contact_force_l2 = RewTerm(
        func=feet_contact_force_l2,
        weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "max_force": 600.0,
        },
    )

    undesired_body_contact = RewTerm(
        func=undesired_body_contact,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "body_contact_forces",
                body_names=NON_FOOT_BODY_NAMES,
            ),
            "threshold": 10.0,
        },
    )

    undesired_body_contact_force_l2 = RewTerm(
        func=undesired_body_contact_force_l2,
        weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg(
                "body_contact_forces",
                body_names=NON_FOOT_BODY_NAMES,
            ),
            "max_force": 100.0,
        },
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.03,
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.005,
    )

    joint_deviation_l2 = RewTerm(
        func=joint_deviation_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
        },
    )

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


##
# Terminations
##

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

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
# Environment
##

@configclass
class MultiLeggedRobotEnvCfg(ManagerBasedRLEnvCfg):
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
        self.decimation = DECIMATION
        self.episode_length_s = 20.0

        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.body_contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt

        if TERRAIN_MODE == "flat":
            self.rewards.flat_orientation_l2.weight = -3.0
            self.rewards.feet_air_time.weight = 0.1
            self.rewards.support_contact_count.weight = 0.4
            self.rewards.feet_contact_force_l2.weight = -0.02

            self.rewards.undesired_body_contact.weight = -0.3
            self.rewards.undesired_body_contact_force_l2.weight = -0.02

            self.rewards.action_rate_l2.weight = -0.03
            self.rewards.action_l2.weight = -0.005
            self.rewards.joint_deviation_l2.weight = -0.05

        elif TERRAIN_MODE == "mixed":
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05
            self.rewards.support_contact_count.weight = 0.25
            self.rewards.feet_contact_force_l2.weight = -0.02

            self.rewards.undesired_body_contact.weight = -0.2
            self.rewards.undesired_body_contact_force_l2.weight = -0.01

            self.rewards.action_rate_l2.weight = -0.03
            self.rewards.action_l2.weight = -0.005
            self.rewards.joint_deviation_l2.weight = -0.03

        else:
            raise ValueError(f"Unknown TERRAIN_MODE: {TERRAIN_MODE}. Use 'flat' or 'mixed'.")

        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)