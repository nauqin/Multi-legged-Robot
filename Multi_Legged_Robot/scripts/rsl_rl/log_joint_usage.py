"""관절 사용량 로깅 스크립트.

학습된 정책을 굴리면서 각 관절이 실제로 얼마나 움직이는지 기록한다.
목적: CPG 매핑에서 joint3_pitch(무릎)를 0으로 고정해도 되는지 판단.

놓을 위치:  scripts/rsl_rl/log_joint_usage.py
           (play.py 와 같은 디렉터리. cli_args.py 를 import 하므로 필수)

사용 예:
    # 평지 정책 (ubin 브랜치)
    python scripts/rsl_rl/log_joint_usage.py --task Hugo-Hexapod-v0 \
        --num_envs 64 --steps 2000 --headless --out joint_usage_flat.npz

    # 험지 정책 (hy 브랜치)
    python scripts/rsl_rl/log_joint_usage.py --task Hugo-Hexapod-v0 \
        --num_envs 64 --steps 2000 --headless --out joint_usage_rough.npz

    # 특정 체크포인트 지정
    python scripts/rsl_rl/log_joint_usage.py --task Hugo-Hexapod-v0 \
        --checkpoint logs/rsl_rl/.../model_3100.pt --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from pathlib import Path

parser = argparse.ArgumentParser(description="Log per-joint usage from a trained policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps to log.")
parser.add_argument("--warmup", type=int, default=100, help="Steps to discard at the start.")
parser.add_argument("--out", type=str, default="joint_usage.npz", help="Output npz path.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Multi_Legged_Robot.tasks  # noqa: F401

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")


def resolve_checkpoint(log_root_path):
    """--checkpoint 가 있으면 그것을, 없으면 가장 최근 run 의 최신 모델을 쓴다."""
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)

    runs = sorted(
        [p for p in Path(log_root_path).iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if len(runs) == 0:
        raise FileNotFoundError(f"No runs found in {log_root_path}")

    models = sorted(runs[-1].glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(models) == 0:
        raise FileNotFoundError(f"No model_*.pt found in {runs[-1]}")
    return str(models[-1])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = resolve_checkpoint(log_root_path)
    print(f"[INFO] checkpoint: {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------------------------------------------------------------
    # 관절 이름과 기본값
    # ---------------------------------------------------------------
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy().copy()

    print(f"[INFO] {len(joint_names)} joints found")

    # ---------------------------------------------------------------
    # 롤아웃
    # ---------------------------------------------------------------
    pos_log, vel_log, done_log = [], [], []

    obs = env.get_observations()
    total = args_cli.warmup + args_cli.steps

    for step in range(total):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        if step >= args_cli.warmup:
            pos_log.append(robot.data.joint_pos.clone().cpu().numpy())
            vel_log.append(robot.data.joint_vel.clone().cpu().numpy())
            done_log.append(dones.clone().cpu().numpy())

        if step % 200 == 0:
            print(f"  step {step}/{total}")

    env.close()

    pos = np.stack(pos_log)            # (T, num_envs, num_joints)
    vel = np.stack(vel_log)
    done = np.stack(done_log)          # (T, num_envs)

    np.savez_compressed(
        args_cli.out,
        joint_pos=pos.astype(np.float32),
        joint_vel=vel.astype(np.float32),
        dones=done.astype(np.bool_),
        default_pos=default_pos.astype(np.float32),
        joint_names=np.array(joint_names),
        checkpoint=np.array([resume_path]),
    )
    print(f"\n[INFO] saved: {args_cli.out}   shape {pos.shape}")

    summarize(pos, default_pos, joint_names)


# =====================================================================
# 요약 출력
# =====================================================================

def summarize(pos, default_pos, joint_names):
    """관절 종류별로 실제 가동 범위를 집계한다.

    핵심 질문: joint3_pitch(무릎)가 joint2_pitch(고관절)에 비해
    얼마나 움직이는가. 무시할 만큼 작으면 CPG IK에서 0으로 고정해도 된다.
    """
    dev = pos - default_pos[None, None, :]          # 기본자세 대비 편차
    flat = dev.reshape(-1, dev.shape[-1])

    groups = {
        "joint1_roll":  r"joint1_roll$",
        "joint2_pitch": r"joint2_pitch$",
        "joint3_pitch": r"joint3_pitch$",
        "prismatic1":   r"prismatic1$",
        "prismatic2":   r"prismatic2$",
    }

    print()
    print("=" * 78)
    print("관절 그룹별 사용량 (기본 자세 대비 편차)")
    print("=" * 78)
    print(f"{'group':<14}{'n':>4}{'std':>10}{'p05':>10}{'p95':>10}"
          f"{'range':>10}{'max|dev|':>11}")
    print("-" * 78)

    stats = {}
    for label, pattern in groups.items():
        idx = [i for i, n in enumerate(joint_names) if re.search(pattern, n)]
        if not idx:
            print(f"{label:<14}{'--':>4}  (해당 관절 없음)")
            continue

        sub = flat[:, idx]
        std = float(sub.std())
        p05, p95 = np.percentile(sub, [5, 95])
        rng = float(sub.max() - sub.min())
        mx = float(np.abs(sub).max())
        stats[label] = dict(std=std, p05=p05, p95=p95, range=rng, max=mx)

        print(f"{label:<14}{len(idx):>4}{std:>10.4f}{p05:>10.4f}"
              f"{p95:>10.4f}{rng:>10.4f}{mx:>11.4f}")

    print()
    print("판정 (joint3_pitch vs joint2_pitch)")
    print("-" * 78)
    if "joint3_pitch" in stats and "joint2_pitch" in stats:
        ratio = stats["joint3_pitch"]["std"] / max(stats["joint2_pitch"]["std"], 1e-9)
        print(f"  표준편차 비 = {ratio:.3f}")
        print(f"  무릎 최대 편차 = {stats['joint3_pitch']['max']:.4f} rad "
              f"({np.degrees(stats['joint3_pitch']['max']):.1f} deg)")
        print()
        if ratio < 0.2:
            print("  -> 무릎이 거의 놀고 있다. CPG IK에서 0 고정해도 손실이 작다.")
        elif ratio < 0.5:
            print("  -> 무릎이 보조적으로 쓰인다. 0 고정으로 시작하되,")
            print("     RL residual이 무릎을 다시 쓰는지 학습 후 확인할 것.")
        else:
            print("  -> 무릎이 적극적으로 쓰인다. 0 고정은 재검토 필요.")
            print("     굽힌 값(예: 0.3 rad)을 기준으로 두는 편이 나을 수 있다.")
    print("=" * 78)

    # 다리별로도 한 번 (좌우/앞뒤 비대칭 확인용)
    knee_idx = [(i, n) for i, n in enumerate(joint_names) if n.endswith("joint3_pitch")]
    if knee_idx:
        print()
        print("다리별 무릎 편차 표준편차")
        print("-" * 40)
        for i, name in knee_idx:
            print(f"  {name:<28}{flat[:, i].std():8.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
