"""CPG + 수치 IK + RL residual 액션 항 (v2).

구조
    상위 규칙 (GaitSelector, 50 Hz): 명령 속도 / 지형 거칠기 -> wave / ripple / tripod
    CPG (GaitCPG, 200 Hz)        : 위상형 Hopf 6개, gait 블렌딩, 전방향 보폭 -> 발끝 목표
    수치 IK (DLSLegIK, 200 Hz)    : 발끝 목표 -> 5 관절 (무릎 굽힘 유지, prismatic 은 기준값)
    + 정책 residual (50 Hz, 30 개) -> set_joint_position_target

v1 (cpg_action.py) 과의 차이는 cpg_core.py 머리말 참고. v1 은 비교용으로 그대로 둔다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .cpg_core import (
    GAIT_NAMES,
    GaitCPG,
    GaitParams,
    GaitSelector,
    HugoPostureCfg,
    IKWeights,
    SelectorParams,
    build_hugo,
    gait_speed_caps,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CPGIKAction(ActionTerm):
    cfg: "CPGIKActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "CPGIKActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        device = self.device
        self._n_legs = len(cfg.leg_prefixes)

        # ---------------- 관절 인덱스 (다리별 체인 순서) -------------------
        names = self._asset.data.joint_names

        def idx(n):
            if n not in names:
                raise ValueError(f"joint '{n}' not found. available: {names}")
            return names.index(n)

        suffix = (cfg.roll_suffix, cfg.hip_pitch_suffix, cfg.prismatic1_suffix,
                  cfg.knee_pitch_suffix, cfg.prismatic2_suffix)
        chain_ids = [[idx(f"{pre}_{s}") for s in suffix] for pre in cfg.leg_prefixes]
        self._chain_ids = torch.tensor(chain_ids, dtype=torch.long, device=device)   # (L, 5)

        # residual 순서는 v1 과 같게: roll 6, hip 6, knee 6, pr1 6, pr2 6
        group_cols = (0, 1, 3, 2, 4)
        self._residual_ids = torch.cat([self._chain_ids[:, c] for c in group_cols])
        self._action_dim = int(self._residual_ids.numel())
        sr, sp = cfg.residual_scale_revolute, cfg.residual_scale_prismatic
        self._residual_scale = torch.tensor(
            [sr] * (3 * self._n_legs) + [sp] * (2 * self._n_legs), device=device)

        # ---------------- 기구학 / IK / CPG --------------------------------
        self._chain, self._ik, nominal, q_nom = build_hugo(
            cfg.posture, device=device, leg_prefixes=cfg.leg_prefixes, weights=cfg.ik_weights)
        self._q_nom = q_nom
        self._cpg = GaitCPG(self.num_envs, nominal, cfg.gait, device=device,
                            initial_gait=cfg.initial_gait)
        caps = gait_speed_caps(cfg.gait, device=device)
        self._selector = GaitSelector(self.num_envs, cfg.selector, caps, device=device)
        self._fixed_gait = None if cfg.gait_mode == "auto" else GAIT_NAMES.index(cfg.gait_mode)

        n = self.num_envs
        self._q_ik = q_nom.unsqueeze(0).repeat(n, 1, 1)                      # (N, L, 5)
        self._ik_err = torch.zeros(n, self._n_legs, device=device)
        self._raw_actions = torch.zeros(n, self._action_dim, device=device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._first = torch.ones(n, dtype=torch.bool, device=device)
        self._rough = torch.zeros(n, device=device)
        self._cmd = torch.zeros(n, 3, device=device)
        self._cmd_sel_f = torch.zeros(n, 3, device=device)   # 걸음새 선택용 (줄이기 전 명령, 저역통과)

        self._physics_dt = float(getattr(env, "physics_dt", env.step_dt / 4.0))
        self._targets = self._asset.data.default_joint_pos.torch.clone()

        print(f"[CPG-IK] q_nom (roll, hip, pr1, knee, pr2) = "
              f"{[round(v, 3) for v in q_nom[0].tolist()]}, caps {caps.tolist()}, "
              f"gait_mode={cfg.gait_mode}")

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def nominal_joint_pos(self) -> torch.Tensor:
        """(L, 5) 기준 자세. env cfg 의 기본 관절각과 같아야 한다."""
        return self._q_nom

    def process_actions(self, actions: torch.Tensor):
        """정책 주기 (50 Hz)."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self._residual_scale

        self._cmd[:] = self._read_command()
        # 걸음새 선택은 줄이기 전 명령으로 한다. 줄인 명령은 지금 걸음새의 최대 속도를
        # 넘지 못하므로, 그걸로 고르면 더 빠른 걸음새로 올라갈 근거가 약해진다.
        raw = self._read_unclipped_command()
        a = self._env.step_dt / (self.cfg.gait.cmd_tau + self._env.step_dt)
        self._cmd_sel_f += a * (raw - self._cmd_sel_f)
        self._cmd_sel_f[self._first] = raw[self._first]      # 리셋 직후는 필터 없이
        self._rough[:] = self._read_roughness()

        # ---- 걸음새 결정 ---------------------------------------------------
        cur = self._cpg.gait_cur
        if self._fixed_gait is not None:
            g = torch.full_like(cur, self._fixed_gait)
            imm = self._first.clone()
        else:
            # 리셋 직후 첫 결정은 필터 전 명령으로 바로 정한다 (_cmd_sel_f 를 위에서 맞춰 둠)
            speed = torch.norm(self._cpg.leg_demand(self._cmd_sel_f), dim=-1).amax(dim=1)
            g, imm = self._selector.step(self._env.step_dt, speed, self._rough, cur, self._first)
        override = self._read_gait_override()
        if override is not None:
            forced = override >= 0
            # 강제 gait 도 인접 gait 를 거쳐 간다 (tripod <-> wave 직행 금지)
            busy = self._cpg.blend < 1.0
            nxt = torch.where(busy, cur, cur + torch.sign(override - cur))
            g = torch.where(forced, torch.where(self._first, override, nxt), g)
        self._cpg.request_gait(g, imm)
        self._first[:] = False

    def apply_actions(self):
        """물리 스텝 (200 Hz)."""
        targets = self._targets
        targets[:] = self._asset.data.default_joint_pos.torch
        if self.cfg.enable_cpg:
            feet = self._cpg.step(self._physics_dt, self._cmd)
            self._q_ik, self._ik_err = self._ik.solve(feet, self._q_ik, iters=self.cfg.ik_iters)
            flat_ids = self._chain_ids.reshape(-1)
            targets[:, flat_ids] = self._q_ik.reshape(self.num_envs, -1)
        if self.cfg.enable_residual:
            targets[:, self._residual_ids] += self._processed_actions
        self._asset.set_joint_position_target(targets)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._cpg.reset(env_ids)
        self._selector.reset(env_ids)
        self._q_ik[ids] = self._q_nom
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._first[ids] = True
        self._cmd_sel_f[ids] = 0.0

    # ------------------------------------------------------------------
    def _read_command(self) -> torch.Tensor:
        try:
            cmd = self._env.command_manager.get_command(self.cfg.command_name)[:, :3]
        except Exception:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return cmd

    def _read_unclipped_command(self) -> torch.Tensor:
        """걸음새 속도로 줄이기 전 명령. 명령 항이 따로 두지 않으면 명령 그대로."""
        try:
            term = self._env.command_manager.get_term(self.cfg.command_name)
        except Exception:  # noqa: BLE001
            return self._cmd
        raw = getattr(term, "unclipped_command", None)
        return self._cmd if raw is None else raw[:, :3]

    def command_scale(self, cmd: torch.Tensor) -> torch.Tensor:
        """명령 항이 부른다: 지금 걸음새가 낼 수 있는 속도 안으로 줄이는 배율 (N,)."""
        return self._cpg.command_scale(cmd[:, :3])

    def _read_gait_override(self):
        if not self.cfg.allow_command_gait_override:
            return None
        try:
            term = self._env.command_manager.get_term(self.cfg.command_name)
        except Exception:
            return None
        return getattr(term, "gait_override", None)

    def _read_roughness(self) -> torch.Tensor:
        """높이 스캔 표준편차 [m]. 센서가 없으면 0."""
        if not self.cfg.roughness_sensor_name:
            return torch.zeros(self.num_envs, device=self.device)
        try:
            sensor = self._env.scene[self.cfg.roughness_sensor_name]
        except KeyError:
            return torch.zeros(self.num_envs, device=self.device)
        hits = sensor.data.ray_hits_w.torch
        valid = torch.isfinite(hits).all(dim=-1)                          # (N, R)
        vf = valid.float()
        cnt = vf.sum(dim=1, keepdim=True).clamp_min(1.0)
        z_raw = torch.where(valid, hits[..., 2], torch.zeros_like(hits[..., 2]))
        z_mean = (z_raw * vf).sum(dim=1, keepdim=True) / cnt
        z = torch.where(valid, z_raw, z_mean)            # 빗나간 레이는 평균으로 (편차 0)
        # 경사는 거칠기가 아니다: 평면을 빼고 남은 편차만 본다
        if self.cfg.roughness_detrend:
            xy_raw = torch.where(valid.unsqueeze(-1), hits[..., :2], torch.zeros_like(hits[..., :2]))
            xy_mean = (xy_raw * vf.unsqueeze(-1)).sum(dim=1, keepdim=True) / cnt.unsqueeze(-1)
            xy = torch.where(valid.unsqueeze(-1), xy_raw - xy_mean, torch.zeros_like(xy_raw))
            A = torch.cat([xy, vf.unsqueeze(-1)], dim=-1)                  # (N, R, 3)
            zc = ((z - z_mean) * vf).unsqueeze(-1)
            AtA = A.transpose(1, 2) @ A + 1e-6 * torch.eye(3, device=A.device)
            sol = torch.linalg.solve(AtA, A.transpose(1, 2) @ zc)
            z = (zc - A @ sol).squeeze(-1)
        return torch.std(z, dim=1)

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    @property
    def oscillator_state(self) -> torch.Tensor:
        return self._cpg.observation()

    @property
    def gait_state(self) -> torch.Tensor:
        return self._cpg.gait_observation()

    @property
    def ik_targets(self) -> torch.Tensor:
        """(N, L, 5) IK 출력 (residual 더하기 전)."""
        return self._q_ik

    @property
    def chain_joint_ids(self) -> torch.Tensor:
        return self._chain_ids


@configclass
class CPGIKActionCfg(ActionTermCfg):
    class_type: type = CPGIKAction
    asset_name: str = "robot"

    leg_prefixes: tuple = ("R1", "R2", "R3", "L1", "L2", "L3")
    roll_suffix: str = "joint1_roll"
    hip_pitch_suffix: str = "joint2_pitch"
    knee_pitch_suffix: str = "joint3_pitch"
    prismatic1_suffix: str = "prismatic1"
    prismatic2_suffix: str = "prismatic2"

    posture: HugoPostureCfg = HugoPostureCfg()
    ik_weights: IKWeights = IKWeights()
    ik_iters: int = 2
    gait: GaitParams = GaitParams()
    selector: SelectorParams = SelectorParams()

    # "auto" = 상위 규칙, 또는 "wave" / "ripple" / "tripod" 고정
    gait_mode: str = "auto"
    initial_gait: str = "ripple"
    # 명령 항이 gait_override (N,) long 을 내면 그걸 따른다 (데모용)
    allow_command_gait_override: bool = True

    command_name: str = "base_velocity"
    roughness_sensor_name: str = "height_scanner"
    roughness_detrend: bool = True

    enable_cpg: bool = True
    enable_residual: bool = True
    residual_scale_revolute: float = 0.10   # [rad]
    residual_scale_prismatic: float = 0.04  # [m]


# =====================================================================
# 관측 항
# =====================================================================

def cpg_ik_oscillator_state(env, action_term_name: str = "cpg") -> torch.Tensor:
    """(N, 12) 재매핑 위상 cos/sin x 진폭. sin > 0 이면 유각기."""
    return env.action_manager.get_term(action_term_name).oscillator_state


def cpg_ik_gait_state(env, action_term_name: str = "cpg") -> torch.Tensor:
    """(N, 6) 목표 gait one-hot, 전환 진행도, duty, 주파수."""
    return env.action_manager.get_term(action_term_name).gait_state
