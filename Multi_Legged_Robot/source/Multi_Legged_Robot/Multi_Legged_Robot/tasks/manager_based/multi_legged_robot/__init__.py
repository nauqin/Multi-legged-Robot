import gymnasium as gym

from . import agents
from .multi_legged_robot_cpg_env_cfg import (
    MultiLeggedRobotCPGEnvCfg,
    MultiLeggedRobotCPGEnvCfg_PLAY,
)

# 베이스라인 (순수 RL). 9000 iteration 체크포인트가 여기 쌓여 있다.
gym.register(
    id="Hugo-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_legged_robot_env_cfg:MultiLeggedRobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# CPG + RL residual
gym.register(
    id="Hugo-Hexapod-CPG-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MultiLeggedRobotCPGEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgCPG",
    },
)

gym.register(
    id="Hugo-Hexapod-CPG-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MultiLeggedRobotCPGEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgCPG",
    },
)
