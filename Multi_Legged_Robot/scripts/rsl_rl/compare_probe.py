"""RL only / CPG only / CPG + RL 성능 비교용 데이터 추출 (Isaac Lab).

놓을 위치:  scripts/rsl_rl/compare_probe.py
           (play.py 와 같은 디렉터리. cli_args.py 를 import 하므로 필수)

세 조건을 같은 로봇 · 같은 지형 · 같은 명령으로 굴리고, 정책 주기(50 Hz)마다
몸통 상태 · 발 접지 · 발진기 상태를 CSV 로 남긴다. 분석은 compare_analyze.py 가 한다.

    --mode rl       Hugo-Hexapod-v0      + 체크포인트         (순수 RL)
    --mode cpg      Hugo-Hexapod-CPG-v0  + residual 끔        (CPG 개루프)
    --mode cpg_rl   Hugo-Hexapod-CPG-v0  + 체크포인트         (CPG + RL residual)

공통 평가 조건 (인자로 바꿀 수 있음)
    지형      --terrain bumps  : random_grid 요철만, 높이 --bump (기본 ±0.03 m), 시드 고정
              --terrain flat   : 무한 평면
              --terrain mixed  : 학습 지형 그대로, --terrain_level 행에 고정
    명령      전진 --speed (기본 0.3 m/s), 횡 0, 회전 0, 에피소드 내 재샘플링 없음
    외란      push_robot 제거, 관측 노이즈 끔, 마찰 1.0 고정, 페이로드 --payload kg 고정
    커리큘럼  제거 (지형 난이도가 도중에 바뀌지 않음)

사용 예 (프로젝트 루트 ~/WS/Hugo_Multi/Multi_Legged_Robot 에서)
    # 1) RL only
    python scripts/rsl_rl/compare_probe.py --mode rl \
        --checkpoint logs/rsl_rl/hugo_base/<run>/model_9000.pt \
        --terrain bumps --headless --out eval_results/raw/rl_bumps.csv

    # 2) CPG only  (체크포인트 불필요)
    python scripts/rsl_rl/compare_probe.py --mode cpg \
        --terrain bumps --headless --out eval_results/raw/cpg_bumps.csv

    # 3) CPG + RL
    python scripts/rsl_rl/compare_probe.py --mode cpg_rl \
        --checkpoint logs/rsl_rl/hugo_cpg/<run>/model_8000.pt \
        --terrain bumps --headless --out eval_results/raw/cpg_rl_bumps.csv

    --checkpoint 를 생략하면 logs/rsl_rl/<experiment>/ 의 가장 최근 run, 최신 모델을 쓴다.
    play.py 에 넘기던 hydra 인자 "env.scene.robot.spawn.usd_path='...'" 는 --usd_path 로 준다.
    --terrain flat 으로 바꿔 한 번 더 돌리면 평지 세트가 된다.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="RL / CPG / CPG+RL comparison probe.")
parser.add_argument("--mode", type=str, required=True, choices=["rl", "cpg", "cpg_rl"])
parser.add_argument("--task", type=str, default=None,
                    help="생략하면 mode 에 맞춰 Hugo-Hexapod-v0 / Hugo-Hexapod-CPG-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42, help="환경 시드 (세 조건에 같은 값)")

# 평가 조건
parser.add_argument("--terrain", type=str, default="bumps", choices=["bumps", "flat", "mixed"])
parser.add_argument("--bump", type=float, default=0.03, help="random_grid 요철 높이 [m]")
parser.add_argument("--terrain_seed", type=int, default=7, help="지형 생성 시드 (세 조건에 같은 값)")
parser.add_argument("--terrain_level", type=int, default=2, help="--terrain mixed 일 때 고정할 행")
parser.add_argument("--speed", type=float, default=0.3, help="전진 명령 [m/s]")
parser.add_argument("--payload", type=float, default=0.0, help="base_link 추가 질량 [kg]")
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--episode_s", type=float, default=20.0)
parser.add_argument("--duration", type=float, default=62.0, help="env 당 기록 시간 [s]")
parser.add_argument("--gait", type=str, default=None, choices=[None, "tripod", "ripple", "wave"],
                    help="cpg / cpg_rl 에서 걸음새 덮어쓰기 (학습은 tripod)")
parser.add_argument("--contact_threshold", type=float, default=5.0, help="접지 판정 힘 [N]")

parser.add_argument("--usd_path", type=str, default=None,
                    help="로봇 USD 덮어쓰기 (play.py 의 env.scene.robot.spawn.usd_path=... 와 같은 역할)")
parser.add_argument("--out", type=str, default=None, help="CSV 경로 (기본 eval_results/raw/<mode>_<terrain>.csv)")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import Multi_Legged_Robot.tasks  # noqa: F401

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

LEG_ORDER = ("R1", "R2", "R3", "L1", "L2", "L3")


# =====================================================================
# 평가 조건을 env_cfg 에 적용
# =====================================================================

def _set(obj, path, value):
    """obj.a.b.c = value 를 경로가 있을 때만. 없으면 조용히 건너뛴다."""
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p) or getattr(cur, p) is None:
            return False
        cur = getattr(cur, p)
    if not hasattr(cur, parts[-1]):
        return False
    setattr(cur, parts[-1], value)
    return True


def apply_eval_conditions(env_cfg, a):
    env_cfg.scene.num_envs = a.num_envs
    env_cfg.seed = a.seed
    env_cfg.episode_length_s = a.episode_s
    if a.usd_path:
        env_cfg.scene.robot.spawn.usd_path = a.usd_path
        print(f"[INFO] robot usd: {a.usd_path}")

    # --- 관측 노이즈 · 외란 · 무작위화 -----------------------------------
    env_cfg.observations.policy.enable_corruption = False
    _set(env_cfg, "events.push_robot", None)
    _set(env_cfg, "events.base_external_force_torque", None)
    ev = getattr(env_cfg, "events", None)
    if ev is not None and getattr(ev, "randomize_robot_material", None) is not None:
        p = ev.randomize_robot_material.params
        p["static_friction_range"] = (a.friction, a.friction)
        p["dynamic_friction_range"] = (a.friction, a.friction)
        p["restitution_range"] = (0.0, 0.0)
    if ev is not None and getattr(ev, "randomize_base_mass", None) is not None:
        ev.randomize_base_mass.params["mass_distribution_params"] = (a.payload, a.payload)

    # --- 명령: 전진만, 에피소드 내 재샘플링 없음 --------------------------
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (a.speed, a.speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    _set(cmd, "heading_command", False)
    _set(cmd, "rel_standing_envs", 0.0)
    _set(cmd, "rel_heading_envs", 0.0)
    cmd.resampling_time_range = (a.episode_s, a.episode_s)

    # --- 커리큘럼 제거: 난이도가 도중에 바뀌면 조건이 흔들린다 ----------
    _set(env_cfg, "curriculum.terrain_levels", None)

    # --- 지형 --------------------------------------------------------------
    terrain = env_cfg.scene.terrain
    if a.terrain == "flat":
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
    elif a.terrain == "bumps":
        gen = terrain.terrain_generator
        gen.seed = a.terrain_seed
        gen.curriculum = False
        sub = gen.sub_terrains["random_grid_rough"]
        sub.grid_height_range = (a.bump, a.bump)
        sub.proportion = 1.0
        gen.sub_terrains = {"random_grid_rough": sub}
        terrain.max_init_terrain_level = None
    else:  # mixed: 학습 지형 그대로, 행만 고정
        gen = terrain.terrain_generator
        gen.seed = a.terrain_seed
        terrain.max_init_terrain_level = a.terrain_level

    # --- CPG 액션 항 ---------------------------------------------------
    if a.mode in ("cpg", "cpg_rl"):
        cpg = env_cfg.actions.cpg
        cpg.enable_residual = (a.mode == "cpg_rl")
        if a.gait is not None:
            if a.mode == "cpg_rl" and a.gait != "tripod":
                print(f"[WARN] cpg_rl 정책은 tripod 로 학습됐다. gait={a.gait} 는 학습 분포 밖.")
            cpg.gait = a.gait
    return env_cfg


# =====================================================================
# 체크포인트
# =====================================================================

def resolve_checkpoint(agent_cfg, a):
    if a.checkpoint:
        return retrieve_file_path(a.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] --checkpoint 없음 → 최근 run 사용: {path}")
    return path


# =====================================================================
# main
# =====================================================================

def main():
    a = args_cli
    task = a.task or ("Hugo-Hexapod-v0" if a.mode == "rl" else "Hugo-Hexapod-CPG-v0")
    out = a.out or os.path.join("eval_results", "raw", f"{a.mode}_{a.terrain}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    env_cfg = parse_env_cfg(task, device=a.device, num_envs=a.num_envs,
                            use_fabric=not a.disable_fabric)
    env_cfg = apply_eval_conditions(env_cfg, a)

    agent_cfg = load_cfg_from_registry(task, a.agent)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, a)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    uenv = env.unwrapped
    device = uenv.device
    dt = float(uenv.step_dt)

    # ---------------- 정책 -------------------------------------------
    ckpt = ""
    if a.mode in ("rl", "cpg_rl"):
        from rsl_rl.runners import OnPolicyRunner
        ckpt = resolve_checkpoint(agent_cfg, a)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=device)
        print(f"[INFO] policy loaded: {ckpt}")
    else:
        act_dim = env.num_actions
        zero = torch.zeros(a.num_envs, act_dim, device=device)
        policy = lambda obs: zero  # noqa: E731
        print("[INFO] CPG 개루프 — residual 없음, 정책 없음")

    # ---------------- 핸들 -------------------------------------------
    robot = uenv.scene["robot"]
    contact = uenv.scene["contact_forces"]
    foot_ids, foot_names = contact.find_bodies(".*_feet")
    # 다리 순서를 R1 R2 R3 L1 L2 L3 로 고정 (Isaac 은 알파벳순으로 준다)
    order = []
    for pre in LEG_ORDER:
        hit = [i for i, n in zip(foot_ids, foot_names) if n.startswith(pre)]
        if not hit:
            raise RuntimeError(f"foot body for {pre} not found in {foot_names}")
        order.append(hit[0])
    foot_ids = torch.tensor(order, dtype=torch.long, device=device)

    scanner = uenv.scene["height_scanner"] if "height_scanner" in uenv.scene.keys() else None
    cpg_term = None
    if a.mode in ("cpg", "cpg_rl"):
        cpg_term = uenv.action_manager.get_term("cpg")
        print(f"[INFO] gait={cpg_term.cfg.gait} omega={cpg_term.cfg.omega} "
              f"residual={cpg_term.cfg.enable_residual}")

    n_steps = int(round(a.duration / dt))
    print(f"[INFO] task={task} mode={a.mode} terrain={a.terrain} speed={a.speed} "
          f"num_envs={a.num_envs} dt={dt:.4f} steps={n_steps}")

    # ---------------- 루프 -------------------------------------------
    rows = []
    obs = env.get_observations()
    if isinstance(obs, (tuple, list)):     # 구버전 rsl_rl 은 (obs, extras) 튜플
        obs = obs[0]
    t0 = time.time()
    with torch.inference_mode():
        for step in range(n_steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()

            pos = robot.data.root_pos_w
            quat = robot.data.root_quat_w
            v_b = robot.data.root_lin_vel_b
            v_w = robot.data.root_lin_vel_w
            w_b = robot.data.root_ang_vel_b
            g_b = robot.data.projected_gravity_b
            roll, pitch, _ = euler_xyz_from_quat(quat)
            roll = torch.atan2(torch.sin(roll), torch.cos(roll))
            pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
            tilt = torch.acos((-g_b[:, 2]).clamp(-1.0, 1.0))

            if contact.data.net_forces_w_history is not None and contact.data.net_forces_w_history.shape[1] > 0:
                f = contact.data.net_forces_w_history[:, :, foot_ids, :].norm(dim=-1).max(dim=1).values
            else:
                f = contact.data.net_forces_w[:, foot_ids, :].norm(dim=-1)
            in_contact = (f > a.contact_threshold).float()               # (N, 6)

            if scanner is not None:
                hz = scanner.data.ray_hits_w[..., 2]
                hz = torch.where(torch.isfinite(hz), hz, torch.nan)
                ground_z = torch.nanmedian(hz, dim=1).values
            else:
                ground_z = torch.zeros(a.num_envs, device=device)

            cmdv = uenv.command_manager.get_command("base_velocity")
            ep = uenv.episode_length_buf.float()

            if cpg_term is not None:
                osc = cpg_term.oscillator_state
            else:
                osc = torch.full((a.num_envs, 12), float("nan"), device=device)

            block = torch.cat([
                torch.arange(a.num_envs, device=device, dtype=torch.float32).unsqueeze(1),
                torch.full((a.num_envs, 1), step * dt, device=device),
                ep.unsqueeze(1),
                dones.float().unsqueeze(1), time_outs.float().unsqueeze(1),
                pos[:, 2:3], ground_z.unsqueeze(1),
                v_b[:, 0:1], v_b[:, 1:2], v_w[:, 2:3],
                w_b[:, 0:1], w_b[:, 1:2], w_b[:, 2:3],
                torch.rad2deg(roll).unsqueeze(1), torch.rad2deg(pitch).unsqueeze(1),
                torch.rad2deg(tilt).unsqueeze(1),
                cmdv[:, 0:1], cmdv[:, 2:3],
                in_contact, in_contact.sum(dim=1, keepdim=True),
                f,
                osc,
            ], dim=1)
            rows.append(block.cpu().numpy().astype(np.float32))

            if step % 500 == 0:
                print(f"  step {step}/{n_steps}  ({time.time() - t0:.0f}s)")

    env.close()

    # ---------------- 저장 -------------------------------------------
    data = np.concatenate(rows, axis=0)
    header = (["env_id", "t", "ep_step", "done", "time_out",
               "z_world", "ground_z", "vx_b", "vy_b", "vz_w", "wx", "wy", "wz",
               "roll_deg", "pitch_deg", "tilt_deg", "cmd_vx", "cmd_wz"]
              + [f"c_{l}" for l in LEG_ORDER] + ["n_contact"]
              + [f"f_{l}" for l in LEG_ORDER]
              + [f"cpg_x{i+1}" for i in range(6)] + [f"cpg_y{i+1}" for i in range(6)])
    gait = cpg_term.cfg.gait if cpg_term is not None else "none"
    meta = (f"# mode={a.mode},task={task},terrain={a.terrain},bump={a.bump},speed={a.speed},"
            f"payload={a.payload},friction={a.friction},gait={gait},dt={dt:.5f},"
            f"num_envs={a.num_envs},seed={a.seed},terrain_seed={a.terrain_seed},"
            f"episode_s={a.episode_s},checkpoint={ckpt}")
    with open(out, "w", newline="") as fh:
        fh.write(meta + "\n")
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(data.tolist())
    print(f"[INFO] saved {data.shape[0]} rows → {out}")
    print(f"[INFO] early terminations: {int(((data[:, 3] > 0.5) & (data[:, 4] < 0.5)).sum())}, "
          f"time-outs: {int(((data[:, 3] > 0.5) & (data[:, 4] > 0.5)).sum())}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        simulation_app.close()
        os._exit(1)
    simulation_app.close()
