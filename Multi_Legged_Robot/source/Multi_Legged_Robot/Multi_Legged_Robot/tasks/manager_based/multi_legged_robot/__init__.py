import gymnasium as gym

from . import agents

# 1. 기존 환경 (험지)
gym.register(
    id="Hugo-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_legged_robot_env_cfg:MultiLeggedRobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# 2. 새로 추가하는 평지(Flat) 전용 환경
from .multi_legged_robot_flat_env_cfg import MultiLeggedRobotFlatEnvCfg, MultiLeggedRobotFlatEnvCfg_PLAY

# # 평지 학습용 환경 등록
# gym.register(
#     id="Hugo-Hexapod-Flat-v0",  # 이름을 기존 스타일에 맞춰서 예쁘게 맞췄습니다!
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": MultiLeggedRobotFlatEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg", # 중요: PPO 설정 추가
#     },
# )

# # 평지 평가(Play)용 환경 등록
# gym.register(
#     id="Hugo-Hexapod-Flat-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": MultiLeggedRobotFlatEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg", # 중요: PPO 설정 추가
#     },
# )