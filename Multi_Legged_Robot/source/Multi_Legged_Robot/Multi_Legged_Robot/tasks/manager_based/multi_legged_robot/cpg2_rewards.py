"""CPG2 (CPG + 수치 IK + residual) 전용 보상 / 종료 함수.

기존 multi_legged_robot_env_cfg.py 의 함수는 그대로 재사용하고,
여기에는 새로 필요한 것만 둔다.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _terrain_z_under_base(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """높이 스캐너 레이 히트 z 의 중앙값 (몸통 아래 지면 높이) (N,)."""
    sensor = env.scene[sensor_cfg.name]
    z = sensor.data.ray_hits_w.torch[..., 2]
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, float("nan")))
    med = torch.nanmedian(z, dim=1).values
    return torch.nan_to_num(med, nan=0.0)


def base_height_terrain_l2(env, target_height: float, sensor_cfg: SceneEntityCfg,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                           max_err: float = 0.20) -> torch.Tensor:
    """지면 기준 몸통 높이 오차^2.  계단 / 경사에서도 지면을 따라간다."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w.torch[:, 2] - _terrain_z_under_base(env, sensor_cfg)
    err = (h - target_height).clamp(-max_err, max_err)
    return err * err


def base_height_terrain_below(env, minimum_height: float, sensor_cfg: SceneEntityCfg,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """종료 조건: 지면 기준 몸통 높이가 minimum 보다 낮다.

    기존 root_height_below_minimum 은 월드 z 기준이라 내려가는 계단 / 경사
    (inverted pyramid) 에서 멀쩡히 걷는데도 종료된다.
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w.torch[:, 2] - _terrain_z_under_base(env, sensor_cfg)
    return h < minimum_height


def knee_extension_l2(env, asset_cfg: SceneEntityCfg, min_bend: float = 0.35,
                      knee_sign: float = -1.0) -> torch.Tensor:
    """무릎이 min_bend 보다 펴지면 벌점.  bend = knee_sign * q (굽힐수록 +).

    IK 는 무릎을 0.2 rad 이상 굽힌 범위에서만 풀지만 residual 이 더 펼 수 있다.
    기준 자세 굽힘은 약 0.87 rad 이고 발 들기 때 더 굽으므로, 정상 보행에서는
    이 항이 0 이다. 무릎을 펴서 버티는 해만 막는다.
    """
    asset = env.scene[asset_cfg.name]
    bend = knee_sign * asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    deficit = torch.clamp(min_bend - bend, min=0.0)
    return torch.mean(deficit * deficit, dim=1)


def feet_slip_l2(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                 threshold: float = 1.0) -> torch.Tensor:
    """디딘 발의 수평 속도^2 합.  CPG 입각기는 등속 궤적이라 0 이 이상적이다."""
    contact = env.scene[sensor_cfg.name]
    forces = contact.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = (forces > threshold).float()
    asset = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.sum(v * v, dim=-1) * in_contact, dim=1)


def feet_stumble(env, sensor_cfg: SceneEntityCfg, ratio: float = 3.0) -> torch.Tensor:
    """발이 수직보다 수평으로 크게 밀리면 (턱에 걸림) 1."""
    contact = env.scene[sensor_cfg.name]
    f = contact.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    fxy = torch.norm(f[..., :2], dim=-1)
    fz = torch.abs(f[..., 2])
    return torch.any((fxy > ratio * fz) & (fxy > 20.0), dim=1).float()


def joint_vel_norm_l2(env, asset_cfg: SceneEntityCfg, vel_limit: float) -> torch.Tensor:
    """속도 한계로 정규화한 관절 속도^2 평균.  (prismatic 볼스크류 0.25 m/s 용)"""
    asset = env.scene[asset_cfg.name]
    v = asset.data.joint_vel.torch[:, asset_cfg.joint_ids] / vel_limit
    return torch.mean(v * v, dim=1)
