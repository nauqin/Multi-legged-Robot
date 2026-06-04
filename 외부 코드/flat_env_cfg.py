# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
import isaaclab.sim as sim_utils

# =========================================================================
# 1. 로봇 본체 및 구동기(모터) 설정
# =========================================================================
HUGO_HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubin/hugo_hexapod/hugo_hexapod.usd", # 절대 경로
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
            ".*_roll": 0.0,
            ".*_pitch": 0.0,
            ".*_prismatic.*": 0.0,
        },
    ),
    actuators={
        "revolute_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_roll", ".*_pitch"],
            stiffness=500.0, damping=50.0,
        ),
        "prismatic_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_prismatic.*"],
            stiffness=3000.0, damping=300.0,
        ),
    },
)

# =========================================================================
# 2. 평지용 센서 부착 (라이다 제거, 속도 향상!)
# =========================================================================
HUGO_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*feet",
    update_period=0.0,
    history_length=3,
    track_air_time=True,
)

HUGO_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link", update_period=0.0,
)

# =========================================================================
# 3. 평지 환경(Flat Env) 클래스 정의
# =========================================================================
@configclass
class HugoHexapodFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        
        # self.events.reset_robot_joints = None
        # self.events.reset_base = None
        # 베이스 링크 변경
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

        # 진짜 평지
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # 센서
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=3,
            track_air_time=True,
        )
        #self.scene.imu = None
        self.scene.imu = HUGO_IMU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/base_link")
        self.scene.height_scanner = None
        # ★ 핵심: 평지에서는 높이 스캐너(라이다)를 꺼버려서 연산량을 대폭 줄입니다.
        #self.scene.height_scanner = None

        # 로봇
        self.scene.robot = HUGO_HEXAPOD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # observation
        self.observations.policy.height_scan = None

        # termination
        # self.terminations.base_contact = None

        # reward
        self.rewards.undesired_contacts = None        
        
        

@configclass
class HugoHexapodFlatEnvCfg_PLAY(HugoHexapodFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 플레이(테스트) 모드일 때는 환경 수를 줄이고 밀기(Push) 이벤트를 끕니다.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None