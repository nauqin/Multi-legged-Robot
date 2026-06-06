# scripts/check_imu_sensor.py

from __future__ import annotations

import argparse
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check Hugo IMU sensor values.")
parser.add_argument("--task", type=str, default="Hugo-Hexapod-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Task registration import.
try:
    import Multi_Legged_Robot.tasks  # noqa: F401
except Exception:
    try:
        import multi_legged_robot.tasks  # noqa: F401
    except Exception as e:
        print("[WARN] Could not import task package automatically.")
        print("       If parse_env_cfg fails, check the import path used in train.py.")
        print(e)


from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

env_cfg.viewer.eye = (5.0, 5.0, 4.0)
env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env = env.unwrapped

obs, info = env.reset()
scene = env.scene

print("\n==============================")
print("Available scene keys:")
print(list(scene.keys()))
print("==============================\n")

if "imu" not in scene.keys():
    raise RuntimeError(
        "scene에 'imu'가 없습니다. "
        "MultiLeggedRobotSceneCfg 안에 imu = ImuCfg(...)가 있는지 확인하세요."
    )

imu = scene["imu"]

print("IMU sensor found.")
print("num_instances:", imu.num_instances)
print("device:", imu.device)
print("is_initialized:", imu.is_initialized)
print()


def print_available_data_fields(data):
    fields = []
    for name in dir(data):
        if name.startswith("_"):
            continue
        try:
            value = getattr(data, name)
        except Exception:
            continue
        if torch.is_tensor(value):
            fields.append((name, tuple(value.shape)))
    return fields


for step in range(args_cli.steps):
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    obs, reward, terminated, truncated, info = env.step(actions)

    # Important: accessing .data triggers lazy update.
    data = imu.data

    if step == 0:
        print("[IMU tensor fields]")
        fields = print_available_data_fields(data)
        for name, shape in fields:
            print(f"{name}: {shape}")
        print()

    if step % 20 == 0:
        print(f"\n[step {step}]")

        # Robustly print common IMU fields.
        common_fields = [
            "pos_w",
            "quat_w",
            "lin_vel_b",
            "ang_vel_b",
            "lin_acc_b",
            "ang_acc_b",
            "lin_vel_w",
            "ang_vel_w",
            "lin_acc_w",
            "ang_acc_w",
        ]

        for name in common_fields:
            if hasattr(data, name):
                value = getattr(data, name)
                if torch.is_tensor(value):
                    print(f"{name} shape:", tuple(value.shape))
                    print(f"env0 {name}:", value[0].detach().cpu().numpy())

        # A quick sanity check if available.
        if hasattr(data, "lin_acc_b") and torch.is_tensor(data.lin_acc_b):
            acc_norm = torch.norm(data.lin_acc_b, dim=-1)
            print("env0 lin_acc_b norm:", acc_norm[0].detach().cpu().item())

        if hasattr(data, "ang_vel_b") and torch.is_tensor(data.ang_vel_b):
            gyro_norm = torch.norm(data.ang_vel_b, dim=-1)
            print("env0 ang_vel_b norm:", gyro_norm[0].detach().cpu().item())

env.close()
simulation_app.close()