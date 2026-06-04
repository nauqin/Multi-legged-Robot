# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
import isaaclab.sim as sim_utils

# =========================================================================
# 1. Hugo Hexapod (6족 보행 로봇) 본체 및 구동기(모터) 설정
# =========================================================================
HUGO_HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # ★ 주의: 아래 경로를 선생님 컴퓨터에 있는 실제 hugo_hexapod.usd 절대 경로로 반드시 수정하세요!
        usd_path="/home/ubin/hugo_hexapod/hugo_hexapod.usd", 
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8), # 시뮬레이션 시작 시 로봇을 0.8m 높이(공중)에 띄움
        joint_pos={
            ".*_roll": 0.0,
            ".*_pitch": 0.0,
            ".*_prismatic.*": 0.0,
        },
    ),
    # 30개의 관절을 회전형(Revolute)과 선형(Prismatic)으로 분리하여 물리적 강성(Stiffness) 부여
    actuators={
        "revolute_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_roll", ".*_pitch"],
            stiffness=500.0,
            damping=50.0,
        ),
        "prismatic_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_prismatic.*"],
            stiffness=3000.0, # 거대한 체중을 버티기 위해 선형 실린더 강성을 매우 높게 설정
            damping=300.0,
        ),
    },
)

# =========================================================================
# 2. 센서 부착 (IMU, 접촉 센서, 라이다)
# =========================================================================
# 발끝 접촉 센서 (6개 다리 끝)
HUGO_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_sphere.*", 
    update_period=0.0,
    history_length=3,
    track_air_time=True,
)

# 몸통 기울기 감지용 IMU
HUGO_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    update_period=0.0,
)

# 지형 높이 스캐너 (라이다)
HUGO_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

@configclass
class HugoHexapodFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

@configclass
class HugoHexapodRoughEnvCfg_PLAY(HugoHexapodRoughEnvCfg): # 이름 변경!
    def __post_init__(self):
        super().__post_init__()
        # ... (이하 내용은 동일하게 유지) ...

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
