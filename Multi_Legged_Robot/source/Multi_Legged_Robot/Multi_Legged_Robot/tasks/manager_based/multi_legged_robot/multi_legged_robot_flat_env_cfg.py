# multi_legged_robot_flat_env_cfg.py

from isaaclab.utils import configclass

# 방금 전까지 작업했던 기본(Rough/Mixed) 환경 설정을 불러옵니다.
from .multi_legged_robot_env_cfg import MultiLeggedRobotEnvCfg


@configclass
class MultiLeggedRobotFlatEnvCfg(MultiLeggedRobotEnvCfg):
    def __post_init__(self):
        # 1. 부모 클래스(기본 설정)의 초기화를 먼저 실행
        super().__post_init__()

        # 2. 지형(Terrain)을 완벽한 평지로 덮어쓰기
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 3. 평지에 맞게 보상(Reward) 가중치 덮어쓰기
        # 평지이므로 수평 유지에 대한 페널티를 아주 강하게 줍니다.
        self.rewards.flat_orientation_l2.weight = -5.0
        
        # 발 체공 시간 보상을 여기서 명시적으로 올려줍니다. (걷기 유도)
        self.rewards.feet_air_time.weight = 0.5

        # (참고) 만약 나중에 높이 스캐너나 커리큘럼을 기본 파일에 추가하게 된다면, 
        # 평지 환경에서는 아래 코드로 꺼주면 됩니다.
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # self.curriculum.terrain_levels = None


@configclass
class MultiLeggedRobotFlatEnvCfg_PLAY(MultiLeggedRobotFlatEnvCfg):
    def __post_init__(self) -> None:
        # 1. 부모(Flat) 클래스 초기화
        super().__post_init__()

        # 2. PLAY(평가/테스트) 모드 전용 설정
        # 학습이 끝난 후 눈으로 볼 때는 로봇 50마리만 띄워서 가볍게 봅니다.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # 평가 시에는 외부 밀치기나 센서 노이즈 등 방해 요소를 끕니다.
        self.observations.policy.enable_corruption = False
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None