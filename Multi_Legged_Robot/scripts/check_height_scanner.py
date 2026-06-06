# scripts/check_height_scanner.py

from __future__ import annotations

import argparse
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check Hugo height scanner sensor.")
parser.add_argument("--task", type=str, default="Hugo-Hexapod-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--debug_vis", action="store_true", help="Enable height scanner debug visualization.")
parser.add_argument("--nominal_height", type=float, default=1.75)
parser.add_argument("--scale", type=float, default=1.0)
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


def get_root_z(robot):
    """Get robot root z position robustly."""
    if hasattr(robot.data, "root_pos_w"):
        return robot.data.root_pos_w[:, 2]
    return robot.data.root_state_w[:, 2]


def compute_relative_height_scan(sensor, robot, nominal_height: float, scale: float):
    """Same conversion used in env_cfg height_scan observation."""
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    base_z = get_root_z(robot).unsqueeze(1)

    relative_height = ray_hits_z - base_z + nominal_height

    relative_height = torch.nan_to_num(
        relative_height,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    relative_height = torch.clamp(relative_height / scale, min=-1.0, max=1.0)

    return relative_height


env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

env_cfg.viewer.eye = (5.0, 5.0, 4.0)
env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

if hasattr(env_cfg.scene, "height_scanner") and env_cfg.scene.height_scanner is not None:
    env_cfg.scene.height_scanner.debug_vis = args_cli.debug_vis

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env = env.unwrapped

obs, info = env.reset()
scene = env.scene

print("\n==============================")
print("Available scene keys:")
print(list(scene.keys()))
print("==============================\n")

if "height_scanner" not in scene.keys():
    raise RuntimeError(
        "scene에 'height_scanner'가 없습니다. "
        "MultiLeggedRobotSceneCfg 안에 height_scanner = RayCasterCfg(...)가 있는지 확인하세요."
    )

height_scanner = scene["height_scanner"]
robot = scene["robot"]

print("Height scanner found.")
print("num_instances:", height_scanner.num_instances)
print("device:", height_scanner.device)
print("is_initialized:", height_scanner.is_initialized)
print("has_debug_vis_implementation:", height_scanner.has_debug_vis_implementation)
print()

# Print cfg info if available.
try:
    print("[Height scanner cfg]")
    print("prim_path:", height_scanner.cfg.prim_path)
    print("mesh_prim_paths:", height_scanner.cfg.mesh_prim_paths)
    print("ray_alignment:", getattr(height_scanner.cfg, "ray_alignment", None))
    print("max_distance:", height_scanner.cfg.max_distance)
    print("update_period:", height_scanner.cfg.update_period)
    print("pattern_cfg:", height_scanner.cfg.pattern_cfg)
    print()
except Exception as e:
    print("[WARN] Could not print full height scanner cfg:", e)
    print()

for step in range(args_cli.steps):
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    obs, reward, terminated, truncated, info = env.step(actions)

    # Accessing .data triggers lazy sensor update.
    data = height_scanner.data

    if step == 0:
        print("[Height scanner data attributes]")
        for name in dir(data):
            if name.startswith("_"):
                continue
            try:
                value = getattr(data, name)
            except Exception:
                continue

            if torch.is_tensor(value):
                print(f"{name}: tensor shape={tuple(value.shape)}, dtype={value.dtype}")
        print()

    if step % 20 == 0:
        print(f"\n[step {step}]")

        ray_hits_w = data.ray_hits_w
        pos_w = data.pos_w

        print("pos_w shape:", tuple(pos_w.shape), "dtype:", pos_w.dtype)
        print("ray_hits_w shape:", tuple(ray_hits_w.shape), "dtype:", ray_hits_w.dtype)

        finite_mask = torch.isfinite(ray_hits_w)
        finite_ratio = finite_mask.float().mean().item()
        print("ray_hits_w finite ratio:", finite_ratio)

        ray_hits_z = ray_hits_w[..., 2]
        z_finite = torch.isfinite(ray_hits_z)

        if z_finite.any():
            z_valid = ray_hits_z[z_finite]
            print(
                "ray_hits_z finite min/max/mean:",
                z_valid.min().item(),
                z_valid.max().item(),
                z_valid.mean().item(),
            )
        else:
            print("ray_hits_z has no finite values.")

        rel_scan = compute_relative_height_scan(
            height_scanner,
            robot,
            nominal_height=args_cli.nominal_height,
            scale=args_cli.scale,
        )

        print("relative height scan shape:", tuple(rel_scan.shape))
        print(
            "env0 rel_scan min/max/mean:",
            rel_scan[0].min().item(),
            rel_scan[0].max().item(),
            rel_scan[0].mean().item(),
        )

        num_show = min(20, rel_scan.shape[1])
        print("env0 rel_scan first values:")
        print(rel_scan[0, :num_show].detach().cpu().numpy())

        if isinstance(obs, dict):
            print("obs keys:", obs.keys())
            if "policy" in obs:
                policy_obs = obs["policy"]
                if torch.is_tensor(policy_obs):
                    print("policy obs shape:", tuple(policy_obs.shape))
        elif torch.is_tensor(obs):
            print("obs shape:", tuple(obs.shape))

env.close()
simulation_app.close()
