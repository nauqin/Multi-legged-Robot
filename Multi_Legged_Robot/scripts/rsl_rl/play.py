# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--terrain_seed",
    type=int,
    default=None,
    help="Override the terrain generator seed. Use a value different from the training seed for evaluation.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Multi_Legged_Robot.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ------------------------------------------------------------------
    # 1 env : 1 patch — terrain grid sizing
    #
    # use_terrain_origins=True 이면 TerrainImporter가 env_spacing을 무시하고
    # env 원점을 sub-terrain 패치 중심으로 잡는다. 따라서 패치 수보다 env가
    # 많으면 여러 로봇이 같은 좌표에 겹쳐서 스폰된다.
    # 여기서 패치 수(num_rows x num_cols)를 num_envs 이상으로 키운다.
    # ------------------------------------------------------------------
    terrain_cfg = getattr(env_cfg.scene, "terrain", None)
    terrain_gen_cfg = getattr(terrain_cfg, "terrain_generator", None) if terrain_cfg is not None else None

    if terrain_gen_cfg is not None:
        num_patch_cols = terrain_gen_cfg.num_cols
        num_patch_rows = math.ceil(env_cfg.scene.num_envs / num_patch_cols)

        terrain_gen_cfg.num_rows = num_patch_rows
        terrain_gen_cfg.num_cols = num_patch_cols

        # 캐시된 옛 지형(다른 rows/cols)을 재사용하지 않도록 한다.
        terrain_gen_cfg.use_cache = False

        # 평가 지형은 학습 지형과 다른 seed 여야 "지형을 외운 정책"이라는
        # 지적을 피할 수 있다.
        if args_cli.terrain_seed is not None:
            terrain_gen_cfg.seed = args_cli.terrain_seed

        # 아래에서 레벨/타입을 직접 지정하므로 초기 레벨 제한은 해제한다.
        terrain_cfg.max_init_terrain_level = None

        print(
            f"[INFO] Terrain grid set to {num_patch_rows} x {num_patch_cols} "
            f"= {num_patch_rows * num_patch_cols} patches for {env_cfg.scene.num_envs} envs "
            f"(seed={terrain_gen_cfg.seed})."
        )

        # terrain_levels_vel 커리큘럼이 reset 때 env_origins를 다시 섞어버리므로
        # 평가 중에는 끈다.
        if hasattr(env_cfg, "curriculum") and getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
            env_cfg.curriculum.terrain_levels = None
            print("[INFO] Disabled terrain_levels curriculum for deterministic placement.")
    else:
        print("[WARN] No terrain generator found. Skipping 1 env : 1 patch setup.")

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ------------------------------------------------------------------
    # 1 env : 1 patch — deterministic placement
    #
    # env i 는 patch (i // num_cols, i % num_cols) 에 배치된다.
    # env_origins 는 in-place 로 갱신해야 scene.env_origins 와 같은 텐서를
    # 계속 가리킨다.
    # ------------------------------------------------------------------
    if terrain_gen_cfg is not None:
        terrain = env.unwrapped.scene.terrain
        num_envs = env.unwrapped.num_envs
        patch_origins = terrain.terrain_origins  # (rows, cols, 3)

        if patch_origins is None:
            raise RuntimeError(
                "terrain.terrain_origins is None. "
                "1 env : 1 patch requires terrain_type='generator' with use_terrain_origins=True."
            )

        num_patch_rows, num_patch_cols = patch_origins.shape[0], patch_origins.shape[1]
        if num_envs > num_patch_rows * num_patch_cols:
            raise ValueError(
                f"num_envs ({num_envs}) exceeds the number of terrain patches "
                f"({num_patch_rows} x {num_patch_cols} = {num_patch_rows * num_patch_cols})."
            )

        env_ids = torch.arange(num_envs, device=patch_origins.device)
        terrain.terrain_levels = (env_ids // num_patch_cols).long()
        terrain.terrain_types = (env_ids % num_patch_cols).long()
        terrain.env_origins[:] = patch_origins[terrain.terrain_levels, terrain.terrain_types]

        # 새 원점으로 로봇을 다시 배치한다.
        env.unwrapped.reset()

        print(f"[INFO] Assigned {num_envs} envs to {num_envs} distinct terrain patches.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
