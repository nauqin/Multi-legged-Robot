# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

This version includes:
- one external customizable terrain configuration from random_grid_terrain_cfg.py
- IMU sensor on base_link
- foot contact sensor
- full-body contact sensor
- height scanner using RayCasterCfg
- foot contact state observation
- roll / hip-pitch / knee-pitch / prismatic separated actions
- bounded terrain swing clearance reward
- terrain roughness based prismatic-use regulation
- terrain-adaptive base leveling reward
- no RGB-D camera code

Design intent:
- On flat terrain, prismatic joints are discouraged strongly.
- On rough terrain, prismatic joints are allowed more, but still regularized.
- Base link is encouraged to stay level on both flat and rough terrain.
- Swing-foot clearance has both a lower and upper bound to prevent exaggerated kicking.
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

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
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from .random_grid_terrain_cfg import HUGO_RANDOM_GRID_TERRAIN_IMPORTER_CFG


##
# Paths and constants
##

ROBOT_USD_PATH = "/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod_ver3_1/hugo_hexapod_ver3_1.usd"

# ver3.1 geometry:
#   segment 0.33 x 2 + prismatic default 0.03 x 2 + foot 0.02  -> sole 0.74 m below hip
#   roll splay 0.2 rad                                          -> body center 0.725 m
NOMINAL_LEG_DROP = 0.74
ROLL_SPLAY = 0.2
NOMINAL_BODY_HEIGHT = 0.725
PRISMATIC_DEFAULT = 0.03

# spawn slightly above nominal so the robot settles instead of being pushed out of the mesh
INITIAL_BODY_HEIGHT = 0.80

# 200 Hz physics, 50 Hz policy
SIM_DT = 1.0 / 200.0
DECIMATION = 4

GRAVITY_MAG = 9.81
ENV_SPACING = 4.0


##
# Height scanner constants
##

HEIGHT_SCANNER_OFFSET_POS = (0.10, 0.0, 0.40)
HEIGHT_SCANNER_SIZE = (1.6, 1.0)
HEIGHT_SCANNER_RESOLUTION = 0.10
HEIGHT_SCANNER_MAX_DISTANCE = 5.0
HEIGHT_SCANNER_UPDATE_PERIOD = SIM_DT * DECIMATION

HEIGHT_SCAN_NOMINAL_HEIGHT = NOMINAL_BODY_HEIGHT
HEIGHT_SCAN_SCALE = 1.0

# TerrainImporter with terrain_type="generator" usually creates the mesh under /World/ground/terrain.
HEIGHT_SCANNER_MESH_PRIM_PATHS = ["/World/ground/terrain"]


##
# Terrain roughness / adaptive reward constants
##

# Roughness is computed from the standard deviation of normalized height_scan.
# Below FLAT_ROUGHNESS_THRESHOLD: mostly flat.
# Above ROUGH_ROUGHNESS_THRESHOLD: mostly non-flat.
# Between the two thresholds: smooth transition.
FLAT_ROUGHNESS_THRESHOLD = 0.025
ROUGH_ROUGHNESS_THRESHOLD = 0.060

# Swing-foot terrain clearance target band.
# Too low -> foot dragging / stumbling.
# Too high -> horse-kick behavior.
SWING_CLEARANCE_MIN = 0.08
SWING_CLEARANCE_MAX = 0.20

# Prismatic regulation.
# The reward function internally uses strong gain on flat terrain and weak gain on rough terrain.
PRISMATIC_FLAT_USAGE_GAIN = 1.00
PRISMATIC_ROUGH_USAGE_GAIN = 0.12
PRISMATIC_VEL_FLAT_GAIN = 0.50
PRISMATIC_VEL_ROUGH_GAIN = 0.08

# Base leveling gain.
# This keeps base_link level in both flat and rough terrain, with stronger pressure on rough terrain
# because prismatic joints should help the base stay parallel while feet adapt to height differences.
BASE_LEVEL_FLAT_GAIN = 0.80
BASE_LEVEL_ROUGH_GAIN = 0.80


##
# Actuator specification (ver3.1)
##

# Reference build: RobStride 04 QDD module (peak 120 N.m, rated 40 N.m, 9:1, 1.42 kg)
# plus an additional reduction stage per joint.
#
#   roll  : 04 + 2:1  -> peak 240 N.m, rated  80 N.m
#   hip   : 04 + 4:1  -> peak 480 N.m, rated 160 N.m
#   knee  : 04 + 3:1  -> peak 360 N.m, rated 120 N.m
#   prism : SMC LEY63 class ball screw, lead 5 mm
#
# effort_limit_sim is intentionally set ABOVE the reference peak so that the
# recorded torque is the torque the gait actually needs, not a clipped value.
# Model selection is done afterwards by comparing the logged RMS torque with
# RATED_* below. Clipping here would make the spec-derivation logs meaningless.
EFFORT_LIMIT_REVOLUTE = 500.0
EFFORT_LIMIT_PRISMATIC = 1500.0

VELOCITY_LIMIT_ROLL = 5.0
VELOCITY_LIMIT_HIP = 5.0
VELOCITY_LIMIT_KNEE = 5.0
VELOCITY_LIMIT_PRISMATIC = 0.25

# armature = rotor inertia x (reduction ratio)^2
# rotor inertia is ESTIMATED from module geometry (not from a datasheet).
# Revisit once the manufacturer provides the rotor inertia.
ARMATURE_ROLL = 0.42
ARMATURE_HIP = 1.70
ARMATURE_KNEE = 0.95
ARMATURE_PRISMATIC = 60.0

# Kp: sized so that the static load deflection stays small.
# Kd: critical damping, 2*sqrt(Kp * (armature + link inertia)).
STIFFNESS_ROLL, DAMPING_ROLL = 2000.0, 120.0
STIFFNESS_HIP, DAMPING_HIP = 4000.0, 300.0
STIFFNESS_KNEE, DAMPING_KNEE = 2000.0, 120.0
STIFFNESS_PRISMATIC, DAMPING_PRISMATIC = 50000.0, 3500.0

# Continuous (thermal) limits of the reference build.
# These are NOT enforced in simulation. They are the pass/fail lines used when
# reading the torque logs: if the RMS torque of a joint exceeds its rated value,
# that joint overheats on the real robot and the actuator must be re-selected.
RATED_TORQUE_ROLL = 80.0
RATED_TORQUE_HIP = 160.0
RATED_TORQUE_KNEE = 120.0
RATED_FORCE_PRISMATIC = 500.0

# Payload: cobot arm + controller + welding feeder/torch + cabling.
PAYLOAD_MASS_RANGE = (0.0, 40.0)


##
# Body / joint name patterns
##

FOOT_BODY_NAMES = [".*_feet"]

NON_FOOT_BODY_NAMES = [
    "base_link",
    ".*_hip_dummy",
    ".*_sphere1",
    ".*_inside",
    ".*_outside",
    ".*_sphere2",
    ".*_inside2",
    ".*_outside2",
]

ROLL_JOINT_NAMES = [".*joint1_roll"]
HIP_PITCH_JOINT_NAMES = [".*joint2_pitch"]
KNEE_PITCH_JOINT_NAMES = [".*joint3_pitch"]
REVOLUTE_JOINT_NAMES = [".*joint.*"]
PRISMATIC_JOINT_NAMES = [".*prismatic.*"]
ALL_CONTROLLED_JOINT_NAMES = [".*joint.*", ".*prismatic.*"]


##
# Height scan / terrain helper functions
##

def _height_scan_values(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    nominal_height: float = HEIGHT_SCAN_NOMINAL_HEIGHT,
    scale: float = HEIGHT_SCAN_SCALE,
):
    """Return normalized relative terrain heights around the robot.

    relative_height = terrain_z - base_z + nominal_height

    Flat ground with base at nominal height should be close to zero.
    """
    sensor = env.scene[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    ray_hits_z = sensor.data.ray_hits_w[..., 2]

    if hasattr(asset.data, "root_pos_w"):
        base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    else:
        base_z = asset.data.root_state_w[:, 2].unsqueeze(1)

    relative_height = ray_hits_z - base_z + nominal_height
    relative_height = torch.nan_to_num(relative_height, nan=0.0, posinf=0.0, neginf=0.0)
    relative_height = torch.clamp(relative_height / scale, min=-1.0, max=1.0)

    return relative_height


def _terrain_roughness_and_masks(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    flat_threshold: float = FLAT_ROUGHNESS_THRESHOLD,
    rough_threshold: float = ROUGH_ROUGHNESS_THRESHOLD,
):
    """Compute terrain roughness and smooth flat/rough weights.

    Returns:
        roughness:    (num_envs,)
        flat_weight:  (num_envs,), 1.0 means flat-like.
        rough_weight: (num_envs,), 1.0 means rough-like.
    """
    scan = _height_scan_values(env, sensor_cfg, asset_cfg)
    roughness = torch.std(scan, dim=1)

    denom = max(rough_threshold - flat_threshold, 1.0e-6)
    rough_weight = torch.clamp((roughness - flat_threshold) / denom, min=0.0, max=1.0)
    flat_weight = 1.0 - rough_weight

    return roughness, flat_weight, rough_weight


def terrain_roughness_obs(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
):
    """Policy observation: scalar roughness from height scanner."""
    roughness, _, _ = _terrain_roughness_and_masks(env, sensor_cfg, asset_cfg)
    return roughness.unsqueeze(1)


def _nearest_terrain_height_from_scan(
    sensor,
    query_pos_w: torch.Tensor,
):
    """Find nearest height-scanner ray hit height for each query point.

    Args:
        sensor: RayCaster sensor.
        query_pos_w: Tensor with shape (num_envs, num_points, 3).

    Returns:
        terrain_z: Tensor with shape (num_envs, num_points).
        nearest_dist_xy: Tensor with shape (num_envs, num_points).
    """
    ray_hits_w = sensor.data.ray_hits_w

    ray_xy = ray_hits_w[..., :2]
    ray_z = ray_hits_w[..., 2]

    query_xy = query_pos_w[..., :2]

    valid = torch.isfinite(ray_z)

    # Distance from each query point to each ray hit point in XY plane.
    # shape: (num_envs, num_points, num_rays)
    dist_xy_sq = torch.sum((query_xy.unsqueeze(2) - ray_xy.unsqueeze(1)) ** 2, dim=-1)

    inf = torch.full_like(dist_xy_sq, float("inf"))
    dist_xy_sq = torch.where(valid.unsqueeze(1), dist_xy_sq, inf)

    nearest_ids = torch.argmin(dist_xy_sq, dim=-1)

    terrain_z = torch.gather(ray_z, dim=1, index=nearest_ids)
    nearest_dist_xy = torch.sqrt(
        torch.gather(dist_xy_sq, dim=2, index=nearest_ids.unsqueeze(-1)).squeeze(-1)
    )

    terrain_z = torch.nan_to_num(terrain_z, nan=0.0, posinf=0.0, neginf=0.0)
    nearest_dist_xy = torch.nan_to_num(nearest_dist_xy, nan=1.0, posinf=1.0, neginf=1.0)

    return terrain_z, nearest_dist_xy


##
# Terrain-aware reward functions
##

def terrain_swing_clearance_band_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    min_clearance: float = SWING_CLEARANCE_MIN,
    max_clearance: float = SWING_CLEARANCE_MAX,
    contact_threshold: float = 1.0,
):
    """Penalty when swing-foot clearance is outside a target band.

    This replaces the previous minimum-only clearance reward.

    Good behavior:
        - Swing foot is high enough to avoid dragging/stumbling.
        - Swing foot is not excessively high, preventing horse-kick-like motions.

    Penalty:
        clearance < min_clearance -> low clearance penalty
        clearance > max_clearance -> excessive clearance penalty
    """
    height_sensor = env.scene[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene[contact_sensor_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    terrain_z, nearest_dist_xy = _nearest_terrain_height_from_scan(height_sensor, foot_pos_w)

    foot_z = foot_pos_w[..., 2]
    clearance = foot_z - terrain_z

    forces = (
        contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    contacts = forces > contact_threshold
    swing_mask = ~contacts

    # If nearest ray is too far, do not trust this reward strongly.
    valid_near = nearest_dist_xy < 0.25

    low_deficit = torch.clamp(min_clearance - clearance, min=0.0) / max(min_clearance, 1.0e-6)
    high_excess = torch.clamp(clearance - max_clearance, min=0.0) / max(max_clearance, 1.0e-6)

    penalty = low_deficit**2 + high_excess**2
    penalty = penalty * swing_mask.float() * valid_near.float()

    denom = torch.sum(swing_mask.float() * valid_near.float(), dim=1).clamp(min=1.0)
    return torch.sum(penalty, dim=1) / denom


def terrain_adaptive_prismatic_deviation_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    prismatic_asset_cfg: SceneEntityCfg,
    flat_usage_gain: float = PRISMATIC_FLAT_USAGE_GAIN,
    rough_usage_gain: float = PRISMATIC_ROUGH_USAGE_GAIN,
):
    """Terrain-adaptive penalty for prismatic joint displacement.

    On flat terrain:
        Strongly penalize prismatic displacement so the robot learns not to use it.

    On rough terrain:
        Penalize prismatic displacement weakly, allowing it to level the base.
    """
    robot = env.scene[asset_cfg.name]
    _, flat_weight, rough_weight = _terrain_roughness_and_masks(env, sensor_cfg, asset_cfg)

    prismatic_pos = robot.data.joint_pos[:, prismatic_asset_cfg.joint_ids]
    default_pos = robot.data.default_joint_pos[:, prismatic_asset_cfg.joint_ids]

    usage = torch.mean((prismatic_pos - default_pos) ** 2, dim=1)
    adaptive_gain = flat_weight * flat_usage_gain + rough_weight * rough_usage_gain

    return adaptive_gain * usage


def terrain_adaptive_prismatic_velocity_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    prismatic_asset_cfg: SceneEntityCfg,
    flat_vel_gain: float = PRISMATIC_VEL_FLAT_GAIN,
    rough_vel_gain: float = PRISMATIC_VEL_ROUGH_GAIN,
):
    """Terrain-adaptive penalty for prismatic joint velocity.

    This prevents rapid prismatic pumping, especially on flat terrain.
    """
    robot = env.scene[asset_cfg.name]
    _, flat_weight, rough_weight = _terrain_roughness_and_masks(env, sensor_cfg, asset_cfg)

    prismatic_vel = robot.data.joint_vel[:, prismatic_asset_cfg.joint_ids]
    vel_usage = torch.mean(prismatic_vel**2, dim=1)

    adaptive_gain = flat_weight * flat_vel_gain + rough_weight * rough_vel_gain
    return adaptive_gain * vel_usage


def terrain_adaptive_base_leveling_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    imu_sensor_cfg: SceneEntityCfg,
    flat_level_gain: float = BASE_LEVEL_FLAT_GAIN,
    rough_level_gain: float = BASE_LEVEL_ROUGH_GAIN,
):
    """Terrain-adaptive base roll/pitch leveling penalty.

    This encourages the base_link to remain parallel/level while the legs adapt.
    It is applied on both flat and rough terrain, but stronger on rough terrain
    so prismatic joints are useful for maintaining base attitude.
    """
    imu = env.scene[imu_sensor_cfg.name]
    _, flat_weight, rough_weight = _terrain_roughness_and_masks(env, sensor_cfg, asset_cfg)

    projected_gravity = imu.data.projected_gravity_b
    level_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)

    adaptive_gain = flat_weight * flat_level_gain + rough_weight * rough_level_gain
    return adaptive_gain * level_error


##
# Contact reward / observation functions
##

def feet_support_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
    min_contacts: int = 3,
):
    """Reward when at least min_contacts feet are in contact."""
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


def feet_contact_state(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
):
    """Return binary foot contact state as policy observation."""
    contact_sensor = env.scene[sensor_cfg.name]

    forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )

    contacts = forces > threshold
    return contacts.float()


##
# IMU observation functions
##

def imu_ang_vel_b(env, sensor_cfg: SceneEntityCfg):
    """IMU angular velocity in body frame."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.ang_vel_b


def imu_projected_gravity_b(env, sensor_cfg: SceneEntityCfg):
    """Gravity direction projected into body frame from IMU."""
    imu = env.scene[sensor_cfg.name]
    return imu.data.projected_gravity_b


def imu_lin_acc_residual_b(env, sensor_cfg: SceneEntityCfg, gravity: float = GRAVITY_MAG):
    """Body-frame dynamic acceleration with gravity component removed."""
    imu = env.scene[sensor_cfg.name]
    residual_acc_b = imu.data.lin_acc_b + gravity * imu.data.projected_gravity_b
    return residual_acc_b / gravity


##
# Height scanner observation function
##

def height_scan(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    nominal_height: float = HEIGHT_SCAN_NOMINAL_HEIGHT,
    scale: float = HEIGHT_SCAN_SCALE,
):
    """Return normalized terrain height scan around the robot."""
    return _height_scan_values(env, sensor_cfg, asset_cfg, nominal_height, scale)


##
# IMU reward functions
##

def imu_projected_gravity_xy_l2(env, sensor_cfg: SceneEntityCfg):
    """Penalty for roll/pitch tilt using IMU projected gravity."""
    imu = env.scene[sensor_cfg.name]
    projected_gravity = imu.data.projected_gravity_b
    return torch.sum(projected_gravity[:, :2] ** 2, dim=1)


def imu_ang_vel_xy_l2(env, sensor_cfg: SceneEntityCfg):
    """Penalty for body roll/pitch angular velocity from IMU."""
    imu = env.scene[sensor_cfg.name]
    ang_vel_b = imu.data.ang_vel_b
    return torch.sum(ang_vel_b[:, :2] ** 2, dim=1)


def imu_vertical_dynamic_acc_l2(env, sensor_cfg: SceneEntityCfg, gravity: float = GRAVITY_MAG):
    """Penalty for vertical dynamic acceleration in body frame."""
    imu = env.scene[sensor_cfg.name]
    residual_acc_b = imu.data.lin_acc_b + gravity * imu.data.projected_gravity_b
    return (residual_acc_b[:, 2] / gravity) ** 2


##
# Joint / posture reward functions
##

def joint_deviation_l2(env, asset_cfg: SceneEntityCfg):
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

    terrain = HUGO_RANDOM_GRID_TERRAIN_IMPORTER_CFG

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
                # Splay the legs outward so the support polygon is wider than
                # the 0.6 m body. ver2 kept roll at 0 with scale 0.05, which
                # made the roll joints effectively unused.
                "L.*joint1_roll": ROLL_SPLAY,
                "R.*joint1_roll": -ROLL_SPLAY,
                ".*joint2_pitch": 0.0,
                ".*joint3_pitch": 0.0,
                # Park the prismatic joints away from the lower limit so they
                # can absorb terrain in BOTH directions. In ver2 the default
                # was the hard stop at 0, so any load pinned them there.
                ".*prismatic1": PRISMATIC_DEFAULT,
                ".*prismatic2": PRISMATIC_DEFAULT,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Split per joint role: each joint now carries its own reduction
            # ratio, reflected inertia (armature) and torque envelope.
            "roll": ImplicitActuatorCfg(
                joint_names_expr=ROLL_JOINT_NAMES,
                stiffness=STIFFNESS_ROLL,
                damping=DAMPING_ROLL,
                armature=ARMATURE_ROLL,
                effort_limit_sim=EFFORT_LIMIT_REVOLUTE,
                velocity_limit_sim=VELOCITY_LIMIT_ROLL,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=HIP_PITCH_JOINT_NAMES,
                stiffness=STIFFNESS_HIP,
                damping=DAMPING_HIP,
                armature=ARMATURE_HIP,
                effort_limit_sim=EFFORT_LIMIT_REVOLUTE,
                velocity_limit_sim=VELOCITY_LIMIT_HIP,
            ),
            "knee_pitch": ImplicitActuatorCfg(
                joint_names_expr=KNEE_PITCH_JOINT_NAMES,
                stiffness=STIFFNESS_KNEE,
                damping=DAMPING_KNEE,
                armature=ARMATURE_KNEE,
                effort_limit_sim=EFFORT_LIMIT_REVOLUTE,
                velocity_limit_sim=VELOCITY_LIMIT_KNEE,
            ),
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=PRISMATIC_JOINT_NAMES,
                stiffness=STIFFNESS_PRISMATIC,
                damping=DAMPING_PRISMATIC,
                armature=ARMATURE_PRISMATIC,
                effort_limit_sim=EFFORT_LIMIT_PRISMATIC,
                velocity_limit_sim=VELOCITY_LIMIT_PRISMATIC,
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_feet",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=True,
    )

    body_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
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

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=HEIGHT_SCANNER_UPDATE_PERIOD,
        history_length=1,
        debug_vis=False,
        mesh_prim_paths=HEIGHT_SCANNER_MESH_PRIM_PATHS,
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=HEIGHT_SCANNER_RESOLUTION,
            size=HEIGHT_SCANNER_SIZE,
        ),
        max_distance=HEIGHT_SCANNER_MAX_DISTANCE,
        offset=RayCasterCfg.OffsetCfg(
            pos=HEIGHT_SCANNER_OFFSET_POS,
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
    """Command specifications for walking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.20, 0.45),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(0.0, 0.0),
        ),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Separated action specifications.

    The prismatic action is now enabled.
    Keep its scale small at first. Increase gradually only after stable behavior appears.
    """

    # scale = how many rad (or m) a policy output of 1.0 corresponds to.
    # Reachable target = default pose +/- scale.
    roll_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ROLL_JOINT_NAMES,
        # 0.05 rad (2.9 deg) in ver2 left the roll joints unusable.
        scale=0.20,
        use_default_offset=True,
    )

    hip_pitch_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HIP_PITCH_JOINT_NAMES,
        # 0.5 m stride on a 0.72 m leg needs about +/-0.35 rad.
        scale=0.30,
        use_default_offset=True,
    )

    knee_pitch_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=KNEE_PITCH_JOINT_NAMES,
        scale=0.30,
        use_default_offset=True,
    )

    prismatic_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=PRISMATIC_JOINT_NAMES,
        # default 0.03 +/- 0.05 -> 0.00 to 0.08 m of the 0.15 m stroke.
        scale=0.05,
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
        """Policy observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        imu_ang_vel_b = ObsTerm(
            func=imu_ang_vel_b,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        imu_projected_gravity_b = ObsTerm(
            func=imu_projected_gravity_b,
            params={"sensor_cfg": SceneEntityCfg("imu")},
        )

        # If this value causes noisy policy behavior, remove this observation first.
        imu_lin_acc_residual_b = ObsTerm(
            func=imu_lin_acc_residual_b,
            params={"sensor_cfg": SceneEntityCfg("imu"), "gravity": GRAVITY_MAG},
        )

        height_scan = ObsTerm(
            func=height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_height": HEIGHT_SCAN_NOMINAL_HEIGHT,
                "scale": HEIGHT_SCAN_SCALE,
            },
        )

        terrain_roughness = ObsTerm(
            func=terrain_roughness_obs,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        feet_contact = ObsTerm(
            func=feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
                "threshold": 1.0,
            },
        )

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Include both revolute and prismatic joints because prismatic actions are now enabled.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES)},
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES)},
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES),
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

    # Payload, not manufacturing tolerance.
    # ver2 used scale (0.85, 1.15), i.e. about +/-6 kg on a 40 kg base, which
    # does not represent the 40 kg cobot-arm payload this robot is designed for.
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": PAYLOAD_MASS_RANGE,
            "operation": "add",
        },
    )

    # External disturbance. Present in the mid-generation cfg, dropped later.
    # Without it the policy is only ever tested against terrain, never against
    # a push, which matters for a robot carrying a moving arm.
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-40.0, 40.0),
            "torque_range": (-8.0, 8.0),
        },
    )


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for terrain-adaptive walking."""

    is_alive = RewTerm(func=mdp.is_alive, weight=0.03)

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    feet_air_time = RewTerm(
        func=feet_air_time_reward,
        weight=0.05,
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
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact_forces", body_names=NON_FOOT_BODY_NAMES),
            "threshold": 10.0,
        },
    )

    # Kept for logging/comparison, but disabled in __post_init__ to avoid duplicating base leveling.
    imu_projected_gravity_xy_l2 = RewTerm(
        func=imu_projected_gravity_xy_l2,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("imu")},
    )

    terrain_adaptive_base_leveling = RewTerm(
        func=terrain_adaptive_base_leveling_l2,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "imu_sensor_cfg": SceneEntityCfg("imu"),
            "flat_level_gain": BASE_LEVEL_FLAT_GAIN,
            "rough_level_gain": BASE_LEVEL_ROUGH_GAIN,
        },
    )

    imu_ang_vel_xy_l2 = RewTerm(
        func=imu_ang_vel_xy_l2,
        weight=-0.07,
        params={"sensor_cfg": SceneEntityCfg("imu")},
    )

    imu_vertical_dynamic_acc_l2 = RewTerm(
        func=imu_vertical_dynamic_acc_l2,
        weight=-0.02,
        params={"sensor_cfg": SceneEntityCfg("imu"), "gravity": GRAVITY_MAG},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.035)

    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.004)

    joint_deviation_l2 = RewTerm(
        func=joint_deviation_l2,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REVOLUTE_JOINT_NAMES)},
    )

    # Revolute only. With the prismatic default at the hard stop (ver2) this
    # term punished the robot for simply being loaded. The default is now at
    # 0.03 m, but the prismatic joints are already regulated by the two
    # terrain-adaptive terms below, so a third limit penalty on them would
    # suppress the mechanism this project is meant to study.
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=REVOLUTE_JOINT_NAMES)},
    )

    terrain_swing_clearance = RewTerm(
        func=terrain_swing_clearance_band_l2,
        weight=-0.30,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "min_clearance": SWING_CLEARANCE_MIN,
            "max_clearance": SWING_CLEARANCE_MAX,
            "contact_threshold": 1.0,
        },
    )

    terrain_adaptive_prismatic_deviation = RewTerm(
        func=terrain_adaptive_prismatic_deviation_l2,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "prismatic_asset_cfg": SceneEntityCfg("robot", joint_names=PRISMATIC_JOINT_NAMES),
            "flat_usage_gain": PRISMATIC_FLAT_USAGE_GAIN,
            "rough_usage_gain": PRISMATIC_ROUGH_USAGE_GAIN,
        },
    )

    # --- actuator load terms -------------------------------------------
    # Required for hardware spec derivation. Without a torque cost the policy
    # has no reason to economise torque, so the logged torque only reflects
    # whatever the effort limit allows and cannot be used to size an actuator.
    # Weight is deliberately small: it should shape the gait, not dominate it.
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES)},
    )

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINT_NAMES)},
    )

    terrain_adaptive_prismatic_velocity = RewTerm(
        func=terrain_adaptive_prismatic_velocity_l2,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
            "prismatic_asset_cfg": SceneEntityCfg("robot", joint_names=PRISMATIC_JOINT_NAMES),
            "flat_vel_gain": PRISMATIC_VEL_FLAT_GAIN,
            "rough_vel_gain": PRISMATIC_VEL_ROUGH_GAIN,
        },
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # ver2 used 0.70 against a 1.02 m nominal height, i.e. the robot could sag
    # 30 cm and keep going. Scaled to the ver3.1 nominal height of 0.725 m.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment
##

@configclass
class MultiLeggedRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration for Hugo hexapod."""

    scene: MultiLeggedRobotSceneCfg = MultiLeggedRobotSceneCfg(
        num_envs=4096,
        env_spacing=ENV_SPACING,
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
        self.scene.height_scanner.update_period = HEIGHT_SCANNER_UPDATE_PERIOD

        # Current early terrain-adaptive setting.
        # Keep yaw disabled until forward gait becomes stable.
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.03, 0.03)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_ang_vel_z.weight = 1.0

        # Air-time is kept weak because too much air-time caused kicking behavior.
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.support_contact_count.weight = 0.0
        self.rewards.feet_contact_force_l2.weight = -0.015
        self.rewards.undesired_body_contact.weight = -1.0

        # Replaced by terrain_adaptive_base_leveling.
        self.rewards.imu_projected_gravity_xy_l2.weight = 0.0
        self.rewards.terrain_adaptive_base_leveling.weight = -1.0

        self.rewards.imu_ang_vel_xy_l2.weight = -0.07
        self.rewards.imu_vertical_dynamic_acc_l2.weight = -0.02

        self.rewards.action_rate_l2.weight = -0.035
        self.rewards.action_l2.weight = -0.004
        self.rewards.joint_deviation_l2.weight = -0.08
        self.rewards.joint_pos_limits.weight = -0.25

        # Bounded clearance penalty. Weight is intentionally around 0.3 as requested.
        self.rewards.terrain_swing_clearance.weight = -0.30

        # Prismatic usage control:
        # - strong penalty on flat terrain
        # - weak penalty on rough terrain
        self.rewards.terrain_adaptive_prismatic_deviation.weight = -1.0
        self.rewards.terrain_adaptive_prismatic_velocity.weight = -0.05

        # Actuator load shaping for spec derivation.
        # If the gait becomes sluggish, lower joint_torques_l2 first.
        self.rewards.joint_torques_l2.weight = -2.0e-8
        self.rewards.joint_acc_l2.weight = -1.0e-7

        # Mesh terrain with thousands of envs overflows the default contact
        # patch buffer, which both spams errors and slows each iteration.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
