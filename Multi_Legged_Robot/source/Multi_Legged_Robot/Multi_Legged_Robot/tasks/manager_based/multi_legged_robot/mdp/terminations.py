# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions specific to the Hugo hexapod environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["root_height_below_minimum_rel"]


def root_height_below_minimum_rel(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the base falls below ``minimum_height`` *above the terrain*.

    ``isaaclab.envs.mdp.root_height_below_minimum`` compares the root z against a
    world-frame constant. On generated rough terrain that misfires in both
    directions: a healthy robot standing on top of a pyramid is never "too low",
    and a healthy robot at the bottom of a slope is terminated immediately.

    This variant subtracts the local ground height sampled by the height
    scanner, so the threshold means "body height above the ground right under
    me" regardless of where the terrain sits in world coordinates.

    Rays that miss the mesh return ``inf`` / ``-inf``; those are excluded from
    the mean rather than poisoning it. If every ray misses, the ground is
    assumed to be at z = 0.

    Note:
        A plain mean over the whole scan patch is used, matching IsaacLab's
        ``base_height_l2``. On stair edges the patch straddles two step levels;
        if that causes spurious terminations, swap the mean for
        ``torch.median`` or restrict the rays to those near the base.

    Args:
        env: The RL environment instance.
        minimum_height: Terminate below this height, measured from the local ground.
        asset_cfg: The robot asset. Defaults to ``SceneEntityCfg("robot")``.
        sensor_cfg: The ray caster used to sample the ground. Defaults to
            ``SceneEntityCfg("height_scanner")``.

    Returns:
        Boolean tensor of shape ``(num_envs,)``, True where the episode should end.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits_z)
    hits_z = torch.where(valid, hits_z, torch.zeros_like(hits_z))
    num_valid = valid.sum(dim=1).clamp(min=1)
    ground_z = hits_z.sum(dim=1) / num_valid

    relative_height = asset.data.root_pos_w[:, 2] - ground_z
    return relative_height < minimum_height