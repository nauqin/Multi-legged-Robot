# scripts/check_contact_sensor.py

from __future__ import annotations

import argparse
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


# --------------------------------------------------
# CLI
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Check Hugo foot contact sensor values.")
parser.add_argument("--task", type=str, default="Hugo-Hexapod-v0")
parser.add_argument("--num_envs", type=int, default=60)
parser.add_argument("--steps", type=int, default=300)

# Isaac Lab 기본 옵션 추가: --headless, --device 등
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# --------------------------------------------------
# Launch Isaac Sim app
# --------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------
# Import task registration
# --------------------------------------------------
# gym registry에 Hugo-Hexapod-v0가 등록되어 있어야 합니다.
# 본인 프로젝트 패키지명에 맞춰 둘 중 하나가 성공하면 됩니다.
try:
    import Multi_Legged_Robot.tasks  # noqa: F401
except Exception:
    try:
        import multi_legged_robot.tasks  # noqa: F401
    except Exception as e:
        print("[WARN] Could not import task package automatically.")
        print("       If gym.make or parse_env_cfg fails, check your package import path.")
        print(e)


# --------------------------------------------------
# Parse env cfg
# --------------------------------------------------
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

# 필요하면 viewer를 조금 보기 좋게 설정
env_cfg.viewer.eye = (5.0, 5.0, 4.0)
env_cfg.viewer.lookat = (0.0, 0.0, 0.5)


# --------------------------------------------------
# Make env
# --------------------------------------------------
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env = env.unwrapped

obs, info = env.reset()

scene = env.scene

print("\n==============================")
print("Available scene keys:")
print(list(scene.keys()))
print("==============================\n")

if "contact_forces" not in scene.keys():
    raise RuntimeError(
        "scene에 'contact_forces'가 없습니다.\n"
        "MultiLeggedRobotSceneCfg 안에\n"
        "contact_forces = ContactSensorCfg(...)\n"
        "가 정의되어 있는지 확인하세요."
    )

contact_sensor = scene["contact_forces"]

print("Contact sensor found.")
print("body_names:", contact_sensor.body_names)
print("num_bodies:", contact_sensor.num_bodies)
print("num_instances:", contact_sensor.num_instances)
print("device:", contact_sensor.device)
print()

if len(contact_sensor.body_names) == 0:
    raise RuntimeError(
        "Contact sensor가 body를 하나도 잡지 못했습니다.\n"
        "ContactSensorCfg의 prim_path가 실제 USD body 이름과 맞는지 확인하세요.\n"
        "예: {ENV_REGEX_NS}/Robot/.*_feet"
    )


# --------------------------------------------------
# Step and print sensor values
# --------------------------------------------------
for step in range(args_cli.steps):
    # 현재 env의 action dimension에 맞는 zero action 생성
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    obs, reward, terminated, truncated, info = env.step(actions)

    # 중요: .data에 접근해야 sensor buffer가 갱신됩니다.
    data = contact_sensor.data

    net_forces_w = data.net_forces_w
    force_norm = torch.norm(net_forces_w, dim=-1)

    current_contact_time = data.current_contact_time
    current_air_time = data.current_air_time
    last_contact_time = data.last_contact_time
    last_air_time = data.last_air_time

    if step % 20 == 0:
        print(f"\n[step {step}]")
        print("net_forces_w shape:", tuple(net_forces_w.shape))
        print("force_norm shape:", tuple(force_norm.shape))

        print("env0 force_norm:")
        print(force_norm[0].detach().cpu().numpy())

        if current_contact_time is not None:
            print("env0 current_contact_time:")
            print(current_contact_time[0].detach().cpu().numpy())
        else:
            print("current_contact_time is None. track_air_time=True인지 확인하세요.")

        if current_air_time is not None:
            print("env0 current_air_time:")
            print(current_air_time[0].detach().cpu().numpy())
        else:
            print("current_air_time is None. track_air_time=True인지 확인하세요.")

        if last_contact_time is not None:
            print("env0 last_contact_time:")
            print(last_contact_time[0].detach().cpu().numpy())

        if last_air_time is not None:
            print("env0 last_air_time:")
            print(last_air_time[0].detach().cpu().numpy())

        contact_bool = force_norm > 1.0
        contact_ratio = contact_bool.float().mean().item()
        max_force = force_norm.max().item()
        mean_force = force_norm.mean().item()

        print("contact ratio over all feet/envs:", contact_ratio)
        print("max force norm:", max_force)
        print("mean force norm:", mean_force)


env.close()
simulation_app.close()