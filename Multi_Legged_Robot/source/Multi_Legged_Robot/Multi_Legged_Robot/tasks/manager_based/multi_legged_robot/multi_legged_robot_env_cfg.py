# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

This version adds:
- IMU sensor on base_link
- foot contact sensor
- full-body contact sensor
- front RGB-D camera on base_link/front_camera

Important:
- Camera is added only as a scene sensor.
- Camera is NOT used in observations or rewards yet.
- Use scripts/check_camera_sensor.py to verify camera data first.
- enabled_self_collisions is kept False because True caused unnatural stretching.
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
from isaaclab.sensors import ContactSensorCfg, ImuCfg, CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp


##
# Paths and constants
##

TERRAIN_MODE = "mixed"  # "flat" or "mixed"

TERRAIN_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/terrain.usd"
ROBOT_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod_ver2/hugo_hexapod_ver2.usd"

INITIAL_BODY_HEIGHT = 1.75

# 200 Hz physics, 50 Hz policy
SIM_DT = 1.0 / 200.0
DECIMATION = 4

GRAVITY_MAG = 9.81


##
# Camera constants
##

# Camera is attached to base_link/front_camera.
# Convention:
# - "world": camera forward axis +X, up axis +Z.
# - pitch down 30 deg around +Y:
#   quaternion = (cos(15deg), 0, sin(15deg), 0)
CAMERA_PITCH_DOWN_30DEG_QUAT = (0.9659258, 0.0, 0.2588190, 0.0)

# Front/head-like position relative to base_link.
# If the camera appears too far/too close, tune x/z only.
CAMERA_OFFSET_POS = (0.35, 0.0, 0.08)


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
# Contact reward functions
##

def feet_support_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    min_contacts: int = 3,
):
    """Reward when at least min_contacts feet are in contact.

    Currently weight is set to 0.0 in __post_init__ for mixed terrain.
    """
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
    threshold: float = 0.30,
    command_threshold: float = 0.1,
):
    """Reward feet that stay in the air briefly before making contact."""
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
    max_force: float = 700.0,
):
    """Penalty for excessive foot contact force."""
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
    """Penalty when non-foot bodies make contact."""
    contact_sensor = env.scene[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = torch.norm(forces_w, dim=-1)

    contacts = force_norm > threshold
    contact_count = torch.sum(contacts.float(), dim=1)

    return contact_count


##
# IMU observation functions
##

def imu_ang_vel_b(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """IMU angular velocity in body frame."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.ang_vel_b


def imu_projected_gravity_b(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """Gravity direction projected into body frame from IMU."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.projected_gravity_b


def imu_lin_acc_residual_b(
    env,
    sensor_cfg: SceneEntityCfg,
    gravity: float = GRAVITY_MAG,
):
    """Body-frame dynamic acceleration with gravity component removed."""
    imu = env.scene[sensor_cfg.name]
    residual_acc_b = imu.data.lin_acc_b + gravity * imu.data.projected_gravity_b
    return residual_acc_b / gravity


##
# IMU reward functions
##

def imu_projected_gravity_xy_l2(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """Penalty for roll/pitch tilt using IMU projected gravity."""
    imu = env.scene[sensor_cfg.name]
    projected_gravity = imu.data.projected_gravity_b
    return torch.sum(projected_gravity[:, :2] ** 2, dim=1)


def imu_ang_vel_xy_l2(
    env,
    sensor_cfg: SceneEntityCfg,
):
    """Penalty for body roll/pitch angular velocity from IMU."""
    imu = env.scene[sensor_cfg.name]
    ang_vel_b = imu.data.ang_vel_b
    return torch.sum(ang_vel_b[:, :2] ** 2, dim=1)


def imu_vertical_dynamic_acc_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    gravity: float = GRAVITY_MAG,
):
    """Penalty for vertical dynamic acceleration in body frame."""
    imu = env.scene[sensor_cfg.name]
    residual_acc_b = imu.data.lin_acc_b + gravity * imu.data.projected_gravity_b
    return (residual_acc_b[:, 2] / gravity) ** 2

##
# Camera observation functions
##

def camera_depth_grid(
    env,
    sensor_cfg: SceneEntityCfg,
    grid_h: int = 3,
    grid_w: int = 5,
    max_depth: float = 5.0,
):
    """Convert depth image into low-dimensional grid features.

    Output shape:
        (num_envs, grid_h * grid_w)

    The depth is normalized to [0, 1].
    Smaller value means closer terrain/object.
    """
    camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["depth"]

    # depth: (N, H, W, 1) -> (N, H, W)
    if depth.dim() == 4:
        depth = depth[..., 0]

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = torch.clamp(depth, min=0.0, max=max_depth)

    num_envs, height, width = depth.shape

    cell_h = height // grid_h
    cell_w = width // grid_w

    features = []

    for i in range(grid_h):
        for j in range(grid_w):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < grid_h - 1 else height
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < grid_w - 1 else width

            patch = depth[:, y0:y1, x0:x1]
            patch_mean = torch.mean(patch, dim=(1, 2))
            features.append(patch_mean / max_depth)

    return torch.stack(features, dim=1)


def camera_depth_near_min(
    env,
    sensor_cfg: SceneEntityCfg,
    max_depth: float = 5.0,
):
    """Minimum depth in the lower-center region.

    This approximates the distance to the near-front terrain/obstacle.
    """
    camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["depth"]

    if depth.dim() == 4:
        depth = depth[..., 0]

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = torch.clamp(depth, min=0.0, max=max_depth)

    _, height, width = depth.shape

    # lower-center area
    y0 = int(height * 0.55)
    y1 = height
    x0 = int(width * 0.35)
    x1 = int(width * 0.65)

    patch = depth[:, y0:y1, x0:x1]
    near_min = torch.amin(patch, dim=(1, 2))

    return (near_min / max_depth).unsqueeze(-1)


def camera_depth_valid_ratio(
    env,
    sensor_cfg: SceneEntityCfg,
    max_depth: float = 5.0,
):
    """Ratio of valid finite depth pixels."""
    camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["depth"]

    if depth.dim() == 4:
        depth = depth[..., 0]

    valid = torch.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    ratio = torch.mean(valid.float(), dim=(1, 2))

    return ratio.unsqueeze(-1)


##
# Camera reward functions
##

def camera_near_obstacle_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    min_safe_depth: float = 0.35,
    max_depth: float = 5.0,
):
    """Penalty if the lower-center camera view has very close terrain/object.

    This is a pre-contact risk penalty.
    It does not replace undesired_body_contact.
    """
    camera = env.scene[sensor_cfg.name]
    depth = camera.data.output["depth"]

    if depth.dim() == 4:
        depth = depth[..., 0]

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = torch.clamp(depth, min=0.0, max=max_depth)

    _, height, width = depth.shape

    y0 = int(height * 0.55)
    y1 = height
    x0 = int(width * 0.35)
    x1 = int(width * 0.65)

    patch = depth[:, y0:y1, x0:x1]
    near_min = torch.amin(patch, dim=(1, 2))

    # If near_min < min_safe_depth, penalty increases.
    penalty = torch.clamp(min_safe_depth - near_min, min=0.0) / min_safe_depth

    return penalty


##
# Joint / posture reward functions
##

def joint_deviation_l2(
    env,
    asset_cfg: SceneEntityCfg,
):
    """Penalty for revolute joints deviating too far from default joint positions."""
    asset = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.mean((joint_pos - default_joint_pos) ** 2, dim=1)


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

    # Front RGB-D camera.
    #
    # Camera prim is spawned as a child of base_link:
    #   /World/envs/env_*/Robot/base_link/front_camera
    #
    # Using convention="world":
    # - camera forward axis: +X
    # - camera up axis: +Z
    #
    # The quaternion below pitches the camera down by about 30 degrees,
    # so it can see the ground in front of the robot.
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        height=120,
        width=160,
        data_types=["rgb", "depth"],
        depth_clipping_behavior="max",
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS,
            rot=CAMERA_PITCH_DOWN_30DEG_QUAT,
            convention="world",
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
    """Command specifications for mixed-terrain revolute-only walking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.15, 0.40),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.12, 0.12),
        ),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for revolute-only training."""

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
        """Policy observations.

        Camera is NOT included here yet.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        imu_ang_vel_b = ObsTerm(
            func=imu_ang_vel_b,
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
            },
        )

        imu_projected_gravity_b = ObsTerm(
            func=imu_projected_gravity_b,
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
            },
        )

        imu_lin_acc_residual_b = ObsTerm(
            func=imu_lin_acc_residual_b,
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "gravity": GRAVITY_MAG,
            },
        )

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
    """Compact reward terms for mixed-terrain walking.

    Camera is NOT used in rewards yet.
    """

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.03,
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )

    feet_air_time = RewTerm(
        func=feet_air_time_reward,
        weight=0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.30,
            "command_threshold": 0.1,
        },
    )

    support_contact_count = RewTerm(
        func=feet_support_count,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 5.0,
            "min_contacts": 3,
        },
    )

    feet_contact_force_l2 = RewTerm(
        func=feet_contact_force_l2,
        weight=-0.015,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "max_force": 700.0,
        },
    )

    undesired_body_contact = RewTerm(
        func=undesired_body_contact,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg(
                "body_contact_forces",
                body_names=NON_FOOT_BODY_NAMES,
            ),
            "threshold": 10.0,
        },
    )

    imu_projected_gravity_xy_l2 = RewTerm(
        func=imu_projected_gravity_xy_l2,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("imu"),
        },
    )

    imu_ang_vel_xy_l2 = RewTerm(
        func=imu_ang_vel_xy_l2,
        weight=-0.06,
        params={
            "sensor_cfg": SceneEntityCfg("imu"),
        },
    )

    imu_vertical_dynamic_acc_l2 = RewTerm(
        func=imu_vertical_dynamic_acc_l2,
        weight=-0.04,
        params={
            "sensor_cfg": SceneEntityCfg("imu"),
            "gravity": GRAVITY_MAG,
        },
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.04,
    )

    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.006,
    )

    joint_deviation_l2 = RewTerm(
        func=joint_deviation_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*joint.*"]),
        },
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.15,
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

        self.decimation = DECIMATION
        self.episode_length_s = 20.0

        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.body_contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt
        self.scene.front_camera.update_period = self.sim.dt

        if TERRAIN_MODE == "flat":
            self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.65)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)

            self.rewards.track_lin_vel_xy.weight = 3.0
            self.rewards.track_ang_vel_z.weight = 0.6

            self.rewards.feet_air_time.weight = 0.10
            self.rewards.support_contact_count.weight = 0.0
            self.rewards.feet_contact_force_l2.weight = -0.015

            self.rewards.undesired_body_contact.weight = -0.6

            self.rewards.imu_projected_gravity_xy_l2.weight = -0.8
            self.rewards.imu_ang_vel_xy_l2.weight = -0.06
            self.rewards.imu_vertical_dynamic_acc_l2.weight = -0.04

            self.rewards.action_rate_l2.weight = -0.04
            self.rewards.action_l2.weight = -0.006
            self.rewards.joint_deviation_l2.weight = -0.05
            self.rewards.joint_pos_limits.weight = -0.15

        elif TERRAIN_MODE == "mixed":
            self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.40)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.12, 0.12)

            self.rewards.track_lin_vel_xy.weight = 3.0
            self.rewards.track_ang_vel_z.weight = 0.6

            self.rewards.feet_air_time.weight = 0.10
            self.rewards.support_contact_count.weight = 0.0
            self.rewards.feet_contact_force_l2.weight = -0.015

            self.rewards.undesired_body_contact.weight = -0.6

            self.rewards.imu_projected_gravity_xy_l2.weight = -0.8
            self.rewards.imu_ang_vel_xy_l2.weight = -0.06
            self.rewards.imu_vertical_dynamic_acc_l2.weight = -0.04

            self.rewards.action_rate_l2.weight = -0.04
            self.rewards.action_l2.weight = -0.006
            self.rewards.joint_deviation_l2.weight = -0.05
            self.rewards.joint_pos_limits.weight = -0.15

        else:
            raise ValueError(f"Unknown TERRAIN_MODE: {TERRAIN_MODE}. Use 'flat' or 'mixed'.")

        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)