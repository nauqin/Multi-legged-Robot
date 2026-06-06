# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

USD-based revolute-only training configuration with contact sensors.

Current purpose:
- load robot from verified USD file
- enable contact sensors
- train basic walking using revolute joints first
- keep prismatic joints near zero
- add first-stage contact-based rewards:
  1) feet_air_time
  2) support_contact_count
  3) feet_contact_force_l2
  4) undesired_body_contact
- support terrain mode switching:
  1) flat  : simple ground plane
  2) mixed : prebuilt USD terrain with flat / rough / slope / stairs, etc.

Notes:
- contact_forces tracks only foot bodies: .*_feet
- body_contact_forces tracks all robot bodies.
- undesired_body_contact uses only non-foot bodies from body_contact_forces.
- This penalizes contact on base / leg links / body parts.
- With self-collisions enabled, it can also penalize foot-body or leg-leg collision-like contacts.
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# IsaacLab official MDP terms
import isaaclab.envs.mdp as mdp


##
# Paths and design-level constants
##

# ------------------------------------------------------------
# Terrain mode
# ------------------------------------------------------------
# "flat"  : use simple flat ground plane
# "mixed" : use prebuilt USD terrain
# ------------------------------------------------------------
TERRAIN_MODE = "flat"     # "flat" or "mixed"

TERRAIN_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/terrain.usd"
ROBOT_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod_ver2/hugo_hexapod_ver2.usd"

# Initial spawn height.
INITIAL_BODY_HEIGHT = 1.75

# 50 Hz action period: sim.dt=1/200, decimation=4
SIM_DT = 1.0 / 200.0
DECIMATION = 4


##
# Body name patterns
##

# Feet bodies verified from contact sensor test:
# ['L1_feet', 'L2_feet', 'L3_feet', 'R1_feet', 'R2_feet', 'R3_feet']
FOOT_BODY_NAMES = [".*_feet"]

# Non-foot body names.
# These are used for undesired contact penalty.
#
# The goal is to penalize:
# - base touching terrain
# - leg links other than feet touching terrain
# - if self collision is enabled, foot/body or leg/body contact-like interactions
#
# If your actual USD body names differ, run check_contact_sensor.py with
# body_contact_forces and print body_names.
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
# Custom reward functions using contact sensors
##

def feet_support_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    min_contacts: int = 3,
):
    """Reward when at least min_contacts feet are in contact.

    For a hexapod, encouraging at least 3 contacts helps form a stable support set.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Shape: (num_envs, num_feet, 3)
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
    """Reward feet that stay in the air for a short time before making contact.

    Local replacement for mdp.feet_air_time.

    Reward is given only when:
    - a foot makes first contact in this step
    - the commanded xy velocity is large enough

    Negative air-time reward is clipped to zero for early learning stability.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # True for feet that newly established contact within this env step.
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # Shape: (num_envs, num_feet)
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # Reward only positive air time beyond threshold.
    air_time_reward = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(air_time_reward * first_contact.float(), dim=1)

    # Do not reward stepping when command is nearly zero.
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.norm(command[:, :2], dim=1)
    reward *= command_xy_norm > command_threshold

    return reward


def feet_contact_force_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 600.0,
):
    """Penalty for excessive foot contact force.

    This discourages hard foot impacts.
    The returned value is positive, so use a negative reward weight.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Shape: (num_envs, num_feet, 3)
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
    """Penalty when non-foot bodies make contact.

    This is similar in spirit to undesired contact penalties used in legged locomotion.

    It penalizes contact forces detected on:
    - base_link
    - hip / sphere / inside / outside leg links
    - all non-foot leg structures

    Important:
    - This function does not use the foot contact sensor.
    - It uses body_contact_forces, which tracks all robot bodies.
    - The SceneEntityCfg body_names should select only non-foot bodies.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Shape: (num_envs, num_non_foot_bodies, 3)
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    # Count how many non-foot bodies are contacting something.
    contacts = force_norm > threshold
    contact_count = torch.sum(contacts.float(), dim=1)

    return contact_count


def undesired_body_contact_force_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 100.0,
):
    """Penalty for large contact force on non-foot bodies.

    This is softer than pure contact count.
    It becomes large only when non-foot body contact force is high.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Shape: (num_envs, num_non_foot_bodies, 3)
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    excess_force = torch.clamp(force_norm - max_force, min=0.0)
    penalty = torch.mean((excess_force / max_force) ** 2, dim=1)

    return penalty


def joint_deviation_l2(
    env,
    asset_cfg: SceneEntityCfg,
):
    """Penalty for revolute joints deviating too far from default joint positions.

    This helps reduce strange folded-leg motions while still allowing movement.
    """
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
    """Penalty for uneven force distribution among contacting feet.

    This is prepared for the next stage.
    Do not enable it until the robot starts showing stable walking.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Shape: (num_envs, num_feet, 3)
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
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod."""

    # ------------------------------------------------------------
    # Terrain selection
    # ------------------------------------------------------------
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
        raise ValueError(
            f"Unknown TERRAIN_MODE: {TERRAIN_MODE}. "
            "Please use 'flat' or 'mixed'."
        )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,

            # Required for ContactSensor.
            # This enables PhysX contact reporter on the robot rigid bodies.
            activate_contact_sensors=True,

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=10.0,
            ),

            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # IMPORTANT:
                # To penalize foot-body or leg-body contacts, self collisions should be enabled.
                #
                # If the simulation becomes unstable immediately after enabling this,
                # it likely means some collision geometries overlap in the default pose.
                # In that case:
                #   1) temporarily set this back to False
                #   2) use undesired body-ground contact penalty only
                #   3) fix collision geometry / self-collision filters in the USD later
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
                ".*prismatic1": 0.0,
                ".*prismatic2": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            # Keep prismatic actuator enabled to hold default position.
            # Prismatic joints are removed from policy action space below.
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=[".*prismatic.*"],
                stiffness=3000.0,
                damping=300.0,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
            ),

            # Revolute joints are controlled by the policy.
            "revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*joint.*"],
                stiffness=2000.0,
                damping=100.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
            ),
        },
    )

    # ------------------------------------------------------------
    # Foot contact sensor
    # ------------------------------------------------------------
    # Tracks only feet.
    # Used for:
    # - feet_air_time
    # - support_contact_count
    # - feet_contact_force_l2
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_feet",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    # ------------------------------------------------------------
    # Full-body contact sensor
    # ------------------------------------------------------------
    # Tracks all robot bodies.
    # Used for:
    # - undesired_body_contact
    # - undesired_body_contact_force_l2
    #
    # We select non-foot bodies later using SceneEntityCfg body_names.
    body_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
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
        """Policy observations for revolute-only walking."""

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
    """Reward terms for first-stage revolute-only walking with contact sensors."""

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

    # Contact-based gait reward.
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

    # Hexapod support reward.
    support_contact_count = RewTerm(
        func=feet_support_count,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 5.0,
            "min_contacts": 3,
        },
    )

    # Penalize strong foot impacts.
    feet_contact_force_l2 = RewTerm(
        func=feet_contact_force_l2,
        weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "max_force": 600.0,
        },
    )

    # Penalize non-foot body contacts.
    #
    # This is the important new term.
    # It discourages using base / leg links as support surfaces
    # and reduces strange folded-leg walking.
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

    # Softer force-based version of non-foot contact penalty.
    #
    # This helps when contact_count alone is too binary.
    # If learning becomes too conservative, reduce this weight first.
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
        weight=-0.03,
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.005,
    )

    # Penalize strange folded joint configurations.
    joint_deviation_l2 = RewTerm(
        func=joint_deviation_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
        },
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

    # Next-stage reward.
    # Enable only after the robot starts producing stable walking.
    #
    # contact_force_distribution = RewTerm(
    #     func=contact_force_distribution,
    #     weight=-0.03,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
    #         "threshold": 5.0,
    #         "max_force": 800.0,
    #     },
    # )


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

    # Do not add body-contact termination yet.
    # We first use undesired_body_contact as a reward penalty.
    # If the robot still abuses body contacts, we can add a termination later.


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

        # Contact sensor update periods.
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.body_contact_forces.update_period = self.sim.dt

        # ------------------------------------------------------------
        # Terrain-mode-specific reward tuning
        # ------------------------------------------------------------
        if TERRAIN_MODE == "flat":
            self.rewards.flat_orientation_l2.weight = -3.0
            self.rewards.feet_air_time.weight = 0.1
            self.rewards.support_contact_count.weight = 0.4
            self.rewards.feet_contact_force_l2.weight = -0.02

            # New body-contact penalties.
            # Start mild-to-medium.
            # If the robot still uses body contacts, increase undesired_body_contact to -0.5.
            # If it becomes too stiff or refuses to move, reduce to -0.1.
            self.rewards.undesired_body_contact.weight = -0.3
            self.rewards.undesired_body_contact_force_l2.weight = -0.02

            # Naturalness penalties.
            self.rewards.action_rate_l2.weight = -0.03
            self.rewards.action_l2.weight = -0.005
            self.rewards.joint_deviation_l2.weight = -0.05

        elif TERRAIN_MODE == "mixed":
            self.rewards.flat_orientation_l2.weight = -1.0
            self.rewards.feet_air_time.weight = 0.05
            self.rewards.support_contact_count.weight = 0.25
            self.rewards.feet_contact_force_l2.weight = -0.02

            # Mixed terrain needs more freedom, so start slightly weaker.
            self.rewards.undesired_body_contact.weight = -0.2
            self.rewards.undesired_body_contact_force_l2.weight = -0.01

            self.rewards.action_rate_l2.weight = -0.03
            self.rewards.action_l2.weight = -0.005
            self.rewards.joint_deviation_l2.weight = -0.03

        else:
            raise ValueError(
                f"Unknown TERRAIN_MODE: {TERRAIN_MODE}. "
                "Please use 'flat' or 'mixed'."
            )

        # viewer
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)