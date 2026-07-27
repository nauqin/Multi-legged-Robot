# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Hugo hexapod manager-based RL environment.

Stage 2: rough terrain + adaptive terrain curriculum.

Changes vs. the flat-terrain version:
  1. CurriculumCfg (terrain_levels_vel) added and registered.
  2. Scene terrain switched to HUGO_MIXED_ROUGH_TERRAIN_IMPORTER_CFG.
  3. height_scanner raised (offset z = 20 m) and max_distance = 25 m,
     height_scan observation clipped to (-1, 1).
  4. base_height termination now measured relative to the terrain under the
     height scanner instead of absolute world z.
  5. contact_forces prim_path narrowed to the bodies actually referenced.
  6. flat_orientation_l2 weight -5.0 -> -1.0 (slopes need body tilt).
  7. Softer actuator gains, lower spawn height, lower depenetration velocity.

Notes:
- Prismatic joints are still in the action space (`prismatic_pos`). The
  docstring of the previous version claimed they were disabled, but the term
  was live during the flat run, so removing it now would change the action
  dimension and break `--resume` from the flat checkpoint. Leave it unless you
  intend to train from scratch.
- Items 7 (actuator gains) and the height-scan offset materially change the
  dynamics/observation distribution, so they are NOT compatible with resuming
  the flat checkpoint. See the constants below.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

from isaaclab.terrains import TerrainImporterCfg  # noqa: F401  (flat fallback below)
from .random_grid_terrain_cfg import (
    HUGO_MIXED_ROUGH_TERRAIN_IMPORTER_CFG
)


##
# Paths and design-level constants
##

# No longer used: terrain now comes from the procedural generator.
# Kept only for reference / manual USD experiments.
TERRAIN_USD_PATH = "/home/ubin/Hugo_Project/usd files/terrain.usd"
ROBOT_USD_PATH = "/home/ubin/Hugo_Project/usd files/hugo_hexapod_ver2/hugo_hexapod_ver2.usd"

# Prismatic을 잠깐 사용하지 않는 revolute-only 단계에서는
# TARGET_BODY_HEIGHT 기반 anti-fluctuation 직접 보상은 제거한다.
# TARGET_BODY_HEIGHT = 1.02

# 1.8 -> 1.2.
# 지형이 생성형으로 바뀌면서 spawn 위치가 sub-terrain origin 위로 잡히므로
# 더 이상 높게 띄울 필요가 없다. 낙하 충격도 줄어든다.
INITIAL_BODY_HEIGHT = 1.2

# 50 Hz action period: sim.dt=1/200, decimation=4
SIM_DT = 1.0 / 200.0
DECIMATION = 4

# imu용 중력
GRAVITY_MAG = 9.81


##
# Height scanner constants
##

# 레이 시작점을 몸통보다 20 m 위로 올린다 (IsaacLab rough preset과 동일).
# 계단/경사에서 시작점이 지형 내부에 묻히는 것을 방지한다.
# 주의: RayCaster는 offset을 ray start에만 적용하고 data.pos_w에는 적용하지
#       않으므로, mdp.height_scan 값은 여전히 "base_link 기준 지면 높이"이다.
HEIGHT_SCAN_RAY_OFFSET_Z = 20.0

# 20 m 시작점에서 지면(최대 몇 m 위/아래)까지 닿아야 하므로 25 m.
HEIGHT_SCAN_MAX_DISTANCE = 25.0

# mdp.height_scan = base_z - hit_z - HEIGHT_SCAN_OFFSET
#
# 0.5  : 평지 학습과 동일한 값. flat 체크포인트에서 resume 하려면 이 값을 유지.
# 1.0  : Hugo의 공칭 몸통 높이(~1.0 m)에 맞춘 값. clip=(-1, 1)과 함께 쓰면
#        낙차/융기를 대칭으로 ±1 m까지 관측할 수 있어 rough terrain에 유리하다.
#        (0.5로 두면 낙차는 0.5 m까지만 보이고 나머지는 전부 1.0으로 포화된다.)
# fresh 학습을 시작한다면 1.0을 권장.
HEIGHT_SCAN_OFFSET = 0.5


##
# Actuator gain constants
##

# 기존(평지) 값: revolute 2000/100, prismatic 3000/300.
# rough terrain에서는 접촉 충격이 커서 과도하게 뻣뻣한 PD가 튐/발산을 유발한다.
#
# 경고: 이 값을 바꾸면 flat 체크포인트의 정책이 학습한 관절 응답 특성이
#       달라진다. `--resume`을 쓸 계획이라면 아래 4개를 기존 값으로 되돌리고,
#       게인 변경은 별도 fresh run에서 검증하는 편이 안전하다.
REVOLUTE_STIFFNESS = 2000.0
REVOLUTE_DAMPING = 100.0
PRISMATIC_STIFFNESS = 3000.0
PRISMATIC_DAMPING = 300.0



##
# Scene definition
##

@configclass
class MultiLeggedRobotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Hugo hexapod."""

    # Procedurally generated mixed rough terrain with a difficulty curriculum.
    # Rows = difficulty levels, columns = terrain types.
    # Requires CurriculumCfg.terrain_levels below to actually be promoted.
    terrain = HUGO_MIXED_ROUGH_TERRAIN_IMPORTER_CFG.replace(
        prim_path="/World/ground",
    )

    # --- Flat fallback ---------------------------------------------------
    # 평지로 되돌리려면 위 terrain을 주석 처리하고 아래를 활성화한 뒤,
    # CurriculumCfg 등록도 함께 제거해야 한다 (terrain_levels_vel은
    # terrain_generator가 없으면 동작하지 않음).
    #
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    # )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,

            activate_contact_sensors=True,

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 10.0 -> 1.0.
                # 울퉁불퉁한 메시와 겹쳐 스폰될 때 10 m/s로 튕겨나가는 것을 방지.
                max_depenetration_velocity=1.0,
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
                ".*prismatic1": 0.0,
                ".*prismatic2": 0.0,
            },
            joint_vel={".*": 0.0},
        ),

        actuators={
            "prismatic": ImplicitActuatorCfg(
                joint_names_expr=[".*prismatic.*"],
                stiffness=PRISMATIC_STIFFNESS,
                damping=PRISMATIC_DAMPING,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
            ),

            "revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*joint.*"],
                stiffness=REVOLUTE_STIFFNESS,
                damping=REVOLUTE_DAMPING,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
            ),
        },
    )

    # 참조되는 body는 `.*_feet` (feet_air_time, feet_contact*) 와
    # `.*(outside|inside|base).*` (undesired_contacts) 뿐이므로 범위를 좁힌다.
    # `Robot/.*`로 전 링크에 걸면 삼각형 메시 지형에서 PhysX
    # `getMaterialFromInternalFaceIndex` 경고가 폭주해 물리 스텝이 로깅에 막힌다.
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

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=SIM_DT * DECIMATION,
        history_length=1,
        debug_vis=False,
        mesh_prim_paths=["/World/ground/terrain"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(1.8, 1.2)),
        max_distance=HEIGHT_SCAN_MAX_DISTANCE,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.20, 0.0, HEIGHT_SCAN_RAY_OFFSET_Z),
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
    """Command specifications for revolute-only walking."""

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
    """Action specifications.

    NOTE: `prismatic_pos` is intentionally left enabled. It was active during
    the flat run, so the flat checkpoint's action head includes it. Removing it
    changes the action dimension and makes `--resume` fail.
    """

    # Roll 관절 (안정적인 자세 유지용)
    roll_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*roll"],
        scale=0.05,
        use_default_offset=True,
    )

    # Pitch 관절 (보행 동작 수행용)
    pitch_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*pitch"],
        scale=0.2,
        use_default_offset=True,
    )

    # 주석과 달리 실제로 활성 상태였음. 위 NOTE 참고.
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
        """Policy observations. Term order and dimensions are unchanged from the
        flat-terrain config so that checkpoints stay loadable."""

        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)

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
        # clip은 필수: 레이가 지형을 빗나가면 hit z = -inf -> height_scan = +inf
        # -> 신경망에서 NaN. clip이 있으면 1.0으로 포화된다.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": HEIGHT_SCAN_OFFSET,
            },
            clip=(-1.0, 1.0),
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

    # push_robot = EventTerm(...)  # still disabled at this stage


##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms."""

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
        weight=-0.25,
    )

    # roll/pitch angular velocity penalty
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time,
        # 평지에서 평균 -0.0139로 음수였음. rough에서 발을 충분히 들지 않으면
        # 0.5 -> 1.0으로 올리거나 threshold를 낮춰 재조정.
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
    # -5.0 -> -1.0.
    # 경사면에서는 몸통이 지형을 따라 기울어야 하는데, -5.0이면 등반 자체가
    # 손해가 되어 커리큘럼이 낮은 레벨에 고착된다.
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.5,
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

    # 절대 z 기준(mdp.root_height_below_minimum)은 경사/계단에서 오작동하므로
    # height_scanner 기준 상대 높이로 교체.
    # 기준이 절대 -> 상대로 바뀌었으므로 임계값도 0.65 -> 0.45.
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
            # 최대 경사 0.4 (= 21.8도 = 0.38 rad)이므로 1.0 rad은 여전히 여유가 있다.
            "limit_angle": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


##
# Curriculum
##

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    `terrain_levels_vel` promotes an env to a harder row when the robot walked
    more than half a sub-terrain (4 m), and demotes it when it walked less than
    half of what its command asked for. Without this term the rows above
    `max_init_terrain_level` are generated but never visited.

    The function lives in the locomotion velocity mdp package, not in
    `isaaclab.envs.mdp`.

    Watch `Curriculum/terrain_levels` in the log:
      - stuck at 0~2  -> lower the terrain difficulty
      - saturates at 7 quickly -> raise it
    """

    terrain_levels = CurrTerm(func=locomotion_mdp.terrain_levels_vel)


##
# Environment configuration
##

@configclass
class MultiLeggedRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration for Hugo hexapod."""

    scene: MultiLeggedRobotSceneCfg = MultiLeggedRobotSceneCfg(
        num_envs=4096,
        # use_terrain_origins=True 이므로 실제 배치는 sub-terrain origin이 결정한다.
        env_spacing=4.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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
        # 평면은 해석적 충돌이라 접촉 패치가 거의 없지만, 삼각형 메시 지형에서
        # 4096 envs x 6족 로봇이 접촉하면 패치 수가 기본값을 쉽게 넘긴다.
        #
        # 기본값 5 * 2**15 = 163840을 넘기면 "Patch buffer overflow" 에러가
        # 매 스텝 쏟아지고, carb 동기 로깅이 물리 스텝을 막아 iteration time이
        # 몇 배로 늘어난다 (예전 getMaterialFromInternalFaceIndex 건과 동일 메커니즘).
        #
        # 계단/경사 비중이 커지는 상위 커리큘럼 레벨에서 더 늘어나므로 여유 있게 잡는다.
        # 또 터지면 2**21로. 다른 버퍼 관련 에러가 뜨면 해당 필드를 각각 올린다:
        #   gpu_max_rigid_contact_count        (기본 2**23)
        #   gpu_found_lost_pairs_capacity      (기본 2**21)
        #   gpu_total_aggregate_pairs_capacity (기본 2**21)
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        # Terrain physics material is applied per-terrain; make sure the
        # generator-based importer keeps the curriculum enabled.
        if getattr(self.scene.terrain, "terrain_generator", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # viewer
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)