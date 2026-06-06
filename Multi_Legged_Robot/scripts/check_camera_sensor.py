# scripts/check_camera_sensor.py

from __future__ import annotations

import argparse
import os
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check Hugo front RGB-D camera sensor.")
parser.add_argument("--task", type=str, default="Hugo-Hexapod-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="/tmp/hugo_camera_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera sensors require camera rendering to be enabled.
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

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


def tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def save_rgb_depth(rgb_tensor, depth_tensor, save_dir, step):
    """Save env0 rgb/depth images for quick visual inspection."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        print("[WARN] PIL/numpy image save unavailable:", e)
        return

    if rgb_tensor is not None:
        rgb = tensor_to_numpy(rgb_tensor[0])

        # Expected shape: H x W x 3 or H x W x 4.
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]

            if rgb.dtype != np.uint8:
                # Handle float image in 0~1 or 0~255.
                rgb = rgb.astype(np.float32)
                if rgb.max() <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            rgb_path = os.path.join(save_dir, f"rgb_step_{step:04d}.png")
            Image.fromarray(rgb).save(rgb_path)
            print("saved:", rgb_path)

    if depth_tensor is not None:
        depth = tensor_to_numpy(depth_tensor[0])

        # Expected shape: H x W or H x W x 1.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        valid = depth[depth > 0.0]
        if valid.size > 0:
            d_min = valid.min()
            d_max = valid.max()
            depth_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
        else:
            depth_norm = depth

        depth_img = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)

        depth_path = os.path.join(save_dir, f"depth_step_{step:04d}.png")
        Image.fromarray(depth_img).save(depth_path)
        print("saved:", depth_path)


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

if "front_camera" not in scene.keys():
    raise RuntimeError(
        "scene에 'front_camera'가 없습니다. "
        "MultiLeggedRobotSceneCfg 안에 front_camera = CameraCfg(...)가 있는지 확인하세요."
    )

camera = scene["front_camera"]

print("Front camera sensor found.")
print("num_instances:", camera.num_instances)
print("device:", camera.device)
print("is_initialized:", camera.is_initialized)
print("image_shape:", camera.image_shape)
print("render_product_paths sample:")
for path in camera.render_product_paths[: min(3, len(camera.render_product_paths))]:
    print(" ", path)
print()

for step in range(args_cli.steps):
    actions = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    obs, reward, terminated, truncated, info = env.step(actions)

    # Accessing .data triggers lazy sensor update.
    data = camera.data

    if step == 0:
        print("[Camera data attributes]")
        for name in dir(data):
            if name.startswith("_"):
                continue
            try:
                value = getattr(data, name)
            except Exception:
                continue

            if torch.is_tensor(value):
                print(f"{name}: tensor shape={tuple(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"{name}: dict keys={list(value.keys())}")
        print()

    if step % 20 == 0:
        print(f"\n[step {step}]")

        output = data.output
        print("output keys:", list(output.keys()))

        rgb = output.get("rgb", None)
        depth = output.get("depth", None)

        # Some IsaacLab versions may store depth-like data under distance_to_image_plane.
        if depth is None:
            depth = output.get("distance_to_image_plane", None)

        if rgb is not None:
            print("rgb shape:", tuple(rgb.shape), "dtype:", rgb.dtype)
            print("rgb env0 min/max:", rgb[0].min().item(), rgb[0].max().item())

        if depth is not None:
            print("depth shape:", tuple(depth.shape), "dtype:", depth.dtype)

            depth0 = depth[0]
            finite_mask = torch.isfinite(depth0)
            if finite_mask.any():
                finite_depth = depth0[finite_mask]
                print(
                    "depth env0 finite min/max/mean:",
                    finite_depth.min().item(),
                    finite_depth.max().item(),
                    finite_depth.mean().item(),
                )
            else:
                print("depth env0 has no finite values.")

        if hasattr(data, "pos_w") and torch.is_tensor(data.pos_w):
            print("camera env0 pos_w:", data.pos_w[0].detach().cpu().numpy())

        if hasattr(data, "quat_w_ros") and torch.is_tensor(data.quat_w_ros):
            print("camera env0 quat_w_ros:", data.quat_w_ros[0].detach().cpu().numpy())

        if hasattr(data, "quat_w_world") and torch.is_tensor(data.quat_w_world):
            print("camera env0 quat_w_world:", data.quat_w_world[0].detach().cpu().numpy())

        if hasattr(data, "quat_w_opengl") and torch.is_tensor(data.quat_w_opengl):
            print("camera env0 quat_w_opengl:", data.quat_w_opengl[0].detach().cpu().numpy())

        if step in [20, 40, 80]:
            save_rgb_depth(rgb, depth, args_cli.save_dir, step)

env.close()
simulation_app.close()