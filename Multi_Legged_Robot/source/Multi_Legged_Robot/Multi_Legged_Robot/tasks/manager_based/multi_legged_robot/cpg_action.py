"""CPG + RL residual 액션 항 (Isaac Lab).

놓을 위치:
    source/Multi_Legged_Robot/Multi_Legged_Robot/tasks/manager_based/
        multi_legged_robot/cpg_action.py

구조
    Hopf 발진기 6개 (다리당 1개, 물리 스텝 200 Hz 로 적분)
        -> 매핑 함수로 발끝 목표 위치
        -> 해석적 IK (무릎 = 0 가정)
        -> roll / hip pitch / prismatic1 / prismatic2 목표값
        -> + 정책 residual (무릎 포함 30개 관절 전부)
        -> set_joint_position_target

정책은 리듬을 만들 필요가 없다. CPG 가 만든 기본 보행 위에
지형 보정만 얹으면 된다. 다리를 영구히 들고 있는 해는 구조적으로 불가능하다.

3a(numpy) 에서 확인된 제약:
  - 다리 길이는 0.66 m 보다 짧아질 수 없다 (prismatic lower=0)
    -> h - k1 >= 0.66 이어야 한다. h=0.76, k1=0.06 이면 여유 40 mm.
  - prismatic 속도 한계 0.25 m/s 가 가장 빡빡하다 (현재 78% 사용)
    -> omega 를 올리면 정비례로 늘어난다. 1.0 Hz 가 상한에 가깝다.
  - 리셋 시 발진기를 극한 순환 위 목표 위상으로 놓아야 한다.
    0 이나 무작위로 놓으면 과도구간에서 prismatic 속도가 한계의 387% 까지 튄다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =====================================================================
# 걸음새 정의 (단위: 주기).  다리 순서는 cfg.leg_prefixes 와 일치해야 한다.
# =====================================================================

GAIT_PHASES = {
    "tripod": (0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    "ripple": (0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6),
    "wave":   (2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6),
}

GAIT_DUTY = {"tripod": 0.5, "ripple": 2 / 3, "wave": 5 / 6}


class CPGAction(ActionTerm):
    """CPG 가 기본 보행을, 정책이 보정을 담당하는 액션 항."""

    cfg: CPGActionCfg
    _asset: Articulation

    def __init__(self, cfg: CPGActionCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self._n_legs = len(cfg.leg_prefixes)
        device = self.device

        # ---------------- 관절 인덱스 --------------------------------
        # 다리별로 5개씩, 이름으로 정확히 찾는다. 정규식 순서에 의존하면
        # Isaac 이 알파벳 순으로 정렬해 버려 다리 순서가 어긋난다.
        names = self._asset.data.joint_names

        def idx_of(full_name: str) -> int:
            if full_name not in names:
                raise ValueError(
                    f"joint '{full_name}' not found. available: {names}")
            return names.index(full_name)

        roll_ids, hip_ids, knee_ids, pr1_ids, pr2_ids = [], [], [], [], []
        for pre in cfg.leg_prefixes:
            roll_ids.append(idx_of(f"{pre}_{cfg.roll_suffix}"))
            hip_ids.append(idx_of(f"{pre}_{cfg.hip_pitch_suffix}"))
            knee_ids.append(idx_of(f"{pre}_{cfg.knee_pitch_suffix}"))
            pr1_ids.append(idx_of(f"{pre}_{cfg.prismatic1_suffix}"))
            pr2_ids.append(idx_of(f"{pre}_{cfg.prismatic2_suffix}"))

        self._roll_ids = torch.tensor(roll_ids, dtype=torch.long, device=device)
        self._hip_ids = torch.tensor(hip_ids, dtype=torch.long, device=device)
        self._knee_ids = torch.tensor(knee_ids, dtype=torch.long, device=device)
        self._pr1_ids = torch.tensor(pr1_ids, dtype=torch.long, device=device)
        self._pr2_ids = torch.tensor(pr2_ids, dtype=torch.long, device=device)

        # residual 은 위 5그룹 전체(30개)에 같은 순서로 대응시킨다
        self._residual_ids = torch.cat([
            self._roll_ids, self._hip_ids, self._knee_ids,
            self._pr1_ids, self._pr2_ids,
        ])
        self._action_dim = int(self._residual_ids.numel())

        sr, sp = cfg.residual_scale_revolute, cfg.residual_scale_prismatic
        self._residual_scale = torch.tensor(
            [sr] * (3 * self._n_legs) + [sp] * (2 * self._n_legs),
            device=device,
        )

        # ---------------- roll 부호 (좌 +, 우 -) ----------------------
        sign = [1.0 if pre.startswith("L") else -1.0 for pre in cfg.leg_prefixes]
        self._roll_sign = torch.tensor(sign, device=device)

        # ---------------- 위상차 행렬 ---------------------------------
        if cfg.gait not in GAIT_PHASES:
            raise ValueError(f"unknown gait '{cfg.gait}'")
        phase = torch.tensor(GAIT_PHASES[cfg.gait], device=device) * 2.0 * math.pi
        theta = phase.unsqueeze(1) - phase.unsqueeze(0)          # (6, 6)
        self._cos_theta = torch.cos(theta)
        self._sin_theta = torch.sin(theta)
        self._cos_theta.fill_diagonal_(0.0)
        self._sin_theta.fill_diagonal_(0.0)
        self._init_phase = phase                                  # 리셋용

        # ---------------- 듀티 (입각기 비율) --------------------------
        # omega 를 상수로 두면 위/아래 반 바퀴가 같은 속도라 duty 가 0.5 로
        # 고정된다. 그러면 ripple/wave 는 위상만 흩어진 채 항상 3개가 떠 있게
        # 되고, wave 는 주기의 34% 동안 무게중심이 지지다각형 밖으로 나간다.
        # 따라서 유각기/입각기 각속도를 분리해야 한다.
        #     t_stance = pi / omega_st = duty * T
        #     t_swing  = pi / omega_sw = (1 - duty) * T
        self._duty = (cfg.duty_override if cfg.duty_override > 0.0
                      else GAIT_DUTY[cfg.gait])
        period = 2.0 * math.pi / cfg.omega
        self._omega_st = math.pi / (self._duty * period)
        self._omega_sw = math.pi / ((1.0 - self._duty) * period)

        # ---------------- 버퍼 ----------------------------------------
        n = self.num_envs
        self._x = torch.zeros(n, self._n_legs, device=device)
        self._y = torch.zeros(n, self._n_legs, device=device)
        self._raw_actions = torch.zeros(n, self._action_dim, device=device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # 커맨드에서 역산한 보폭을 저역통과시킨 값
        self._d_step_smooth = torch.zeros(n, 1, device=device)

        self._default_joint_pos = self._asset.data.default_joint_pos.clone()

        # 물리 스텝 dt (정책 주기가 아니라 시뮬레이션 스텝)
        self._physics_dt = float(getattr(env, "physics_dt", env.step_dt / 4.0))

        self._reset_oscillators(None)

        # ---------------- 안전 점검 -----------------------------------
        margin = cfg.stance_height - cfg.swing_lift - cfg.leg_length_min
        if margin < 0.0:
            raise ValueError(
                f"stance_height - swing_lift = "
                f"{cfg.stance_height - cfg.swing_lift:.3f} m < L0 "
                f"{cfg.leg_length_min:.3f} m. prismatic 은 수축만 되므로 "
                f"이 조합은 도달 불가능하다. h 를 올리거나 k1 을 줄일 것.")
        stab = 2.0 * cfg.alpha * cfg.mu ** 2 * self._physics_dt
        print(f"[CPG] dt={self._physics_dt:.5f}s  "
              f"f={cfg.omega / (2 * math.pi):.3f}Hz  gait={cfg.gait}  "
              f"duty={self._duty:.3f}  L0 margin={margin * 1000:.0f}mm  "
              f"2*a*mu^2*dt={stab:.2f}")

        # 유각기가 짧아질수록 발을 같은 높이만큼 더 빨리 올려야 한다.
        # prismatic 속도 한계 0.25 m/s 를 넘기 쉬운 지점.
        swing_ratio = 0.5 / (1.0 - self._duty)
        if swing_ratio > 1.5:
            print(f"[CPG] 경고: duty {self._duty:.3f} 이면 유각기 발 속도가 "
                  f"tripod 대비 {swing_ratio:.1f}배다. "
                  f"omega 를 {cfg.omega / swing_ratio:.2f} 이하로 낮출 것.")

    # ------------------------------------------------------------------
    # ActionTerm 인터페이스
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

    def process_actions(self, actions: torch.Tensor):
        """정책 주기(50 Hz)마다 한 번 호출된다. residual 만 저장한다."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self._residual_scale

    def apply_actions(self):
        """물리 스텝(200 Hz)마다 호출된다. 발진기 적분은 여기서 한다.

        정책 주기에 발진기를 묶으면 리듬이 거칠어지므로 분리하는 것이 중요하다.
        """
        if self.cfg.enable_cpg:
            self._integrate()
            targets = self._cpg_targets()
        else:
            targets = self._default_joint_pos.clone()

        if self.cfg.enable_residual:
            targets[:, self._residual_ids] += self._processed_actions

        self._asset.set_joint_position_target(targets)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._reset_oscillators(env_ids)
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._d_step_smooth.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._d_step_smooth[env_ids] = 0.0

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------

    def _reset_oscillators(self, env_ids):
        """극한 순환 위, 목표 위상으로 놓는다.

        0 이나 무작위로 놓으면 수렴 과도구간에서 다리 길이가 급격히 변해
        prismatic 속도 한계(0.25 m/s)를 크게 넘는다. 3a 에서 확인된 항목.
        """
        x0 = self.cfg.mu * torch.cos(self._init_phase)
        y0 = self.cfg.mu * torch.sin(self._init_phase)
        if env_ids is None:
            self._x[:] = x0
            self._y[:] = y0
        else:
            self._x[env_ids] = x0
            self._y[env_ids] = y0

    def _integrate(self):
        """결합된 Hopf 발진기 한 스텝 (오일러)."""
        x, y = self._x, self._y
        cfg = self.cfg

        r_sq = x * x + y * y
        radial = cfg.alpha * (cfg.mu * cfg.mu - r_sq)

        # y > 0 이 유각기, y < 0 이 입각기. 시그모이드로 부드럽게 전환한다.
        # 계단함수를 쓰면 omega 가 순간적으로 튀어 궤적이 꺾인다.
        blend = torch.sigmoid(cfg.duty_sharpness * y)
        omega = self._omega_sw * blend + self._omega_st * (1.0 - blend)

        x_dot = radial * x - omega * y
        y_dot = radial * y + omega * x

        if cfg.coupling != 0.0:
            # sum_j C[i,j] * x[j]  ->  x @ C^T  (배치 차원 유지)
            cx = x @ self._cos_theta.t()
            sx = x @ self._sin_theta.t()
            cy = y @ self._cos_theta.t()
            sy = y @ self._sin_theta.t()
            x_dot = x_dot + cfg.coupling * (cx - sy)
            y_dot = y_dot + cfg.coupling * (sx + cy)

        self._x = x + self._physics_dt * x_dot
        self._y = y + self._physics_dt * y_dot

    def _d_step(self) -> torch.Tensor:
        """커맨드 속도로부터 보폭 진폭을 역산한다.  (num_envs, 1)

            v = 2 * d_step * f / duty   ->   d_step = v * duty / (2f)

        부호가 살아 있으므로 음수 커맨드면 자연스럽게 뒤로 걷는다.
        """
        cfg = self.cfg
        if not cfg.use_command_speed:
            target = torch.full((self.num_envs, 1), cfg.d_step_fixed,
                                device=self.device)
        else:
            try:
                cmd = self._env.command_manager.get_command(cfg.command_name)
                vx = cmd[:, 0:1]
                f = cfg.omega / (2.0 * math.pi)
                target = vx * self._duty / (2.0 * f)
                target = torch.clamp(target, -cfg.d_step_max, cfg.d_step_max)
            except Exception:
                target = torch.full((self.num_envs, 1), cfg.d_step_fixed,
                                    device=self.device)

        # 속도 커맨드는 몇 초마다 계단처럼 바뀐다. 그대로 쓰면 보폭이
        # 한 물리 스텝(5 ms)에 70 mm 점프하고, 발끝 속도로는 14 m/s 가
        # 되어 다리가 채찍처럼 휘두른다. 1차 저역통과로 부드럽게 잇는다.
        a = self._physics_dt / (cfg.d_step_tau + self._physics_dt)
        self._d_step_smooth += a * (target - self._d_step_smooth)
        return self._d_step_smooth

    def _cpg_targets(self) -> torch.Tensor:
        """발진기 상태 -> 매핑 -> IK -> 관절 목표값."""
        cfg = self.cfg
        x, y = self._x, self._y

        # --- 매핑: 다리 평면 안의 2D 궤적 ---------------------------
        d_step = self._d_step()                       # (N, 1)
        px_l = -d_step * x                            # (N, 6) 앞뒤
        pz_l = torch.where(
            y > 0.0,
            -cfg.stance_height + cfg.swing_lift * y,  # 유각기: 발을 든다
            -cfg.stance_height + cfg.stance_push * y,  # 입각기: 살짝 누른다
        )

        # --- 평면을 roll_nominal 만큼 바깥으로 기울인다 -------------
        # 이렇게 하면 IK 결과 roll 이 정확히 roll_nominal 로 나온다.
        roll_nom = cfg.roll_nominal * self._roll_sign  # (6,)
        c = torch.cos(roll_nom).unsqueeze(0)
        s = torch.sin(roll_nom).unsqueeze(0)
        px = px_l
        py = -s * pz_l
        pz = c * pz_l

        # --- IK (무릎 = 0) -------------------------------------------
        L = torch.sqrt(px * px + py * py + pz * pz).clamp_min(1e-6)
        q_roll = torch.atan2(py, -pz)
        q_hip = -torch.asin((px / L).clamp(-1.0, 1.0))
        d = ((L - cfg.leg_length_min) * 0.5).clamp(0.0, cfg.prismatic_stroke)

        # --- 관절 목표값 조립 ----------------------------------------
        targets = self._default_joint_pos.clone()
        targets[:, self._roll_ids] = q_roll
        targets[:, self._hip_ids] = q_hip
        targets[:, self._knee_ids] = 0.0    # CPG 는 무릎을 쓰지 않는다
        targets[:, self._pr1_ids] = d
        targets[:, self._pr2_ids] = d
        return targets

    # ------------------------------------------------------------------
    # 관측용
    # ------------------------------------------------------------------

    @property
    def oscillator_state(self) -> torch.Tensor:
        """(num_envs, 12) = [x1..x6, y1..y6].  관측에 반드시 넣을 것."""
        return torch.cat([self._x, self._y], dim=1)



@configclass
class CPGActionCfg(ActionTermCfg):
    """CPG + residual 액션 설정."""

    class_type: type = CPGAction

    asset_name: str = "robot"

    # --- 다리/관절 이름 ------------------------------------------------
    leg_prefixes: tuple = ("R1", "R2", "R3", "L1", "L2", "L3")
    roll_suffix: str = "joint1_roll"
    hip_pitch_suffix: str = "joint2_pitch"
    knee_pitch_suffix: str = "joint3_pitch"
    prismatic1_suffix: str = "prismatic1"
    prismatic2_suffix: str = "prismatic2"

    # --- 발진기 ---------------------------------------------------------
    gait: str = "tripod"
    mu: float = 1.0
    omega: float = 6.3          # rad/s -> 1.003 Hz (한 걸음 주기)
    alpha: float = 50.0         # 2*alpha*mu^2*dt < 1 (dt=1/200 에서 0.5)
    coupling: float = 0.5       # 크게 하면 진폭이 mu 보다 부풀어 보폭이 커진다

    # 듀티(입각기 비율)를 gait 기본값 대신 직접 지정하고 싶을 때만 사용
    duty_override: float = -1.0
    duty_sharpness: float = 20.0   # 시그모이드 전환의 날카로움

    # --- 매핑 -----------------------------------------------------------
    stance_height: float = 0.76     # h  [m]  hip -> 발 링크 원점
    swing_lift: float = 0.06        # k1 [m]  h - k1 >= L0 필수
    stance_push: float = 0.008      # k2 [m]
    roll_nominal: float = 0.20      # [rad] 좌 +, 우 -

    # --- 기구학 (ver3.1 URDF) -------------------------------------------
    leg_length_min: float = 0.66    # L0
    prismatic_stroke: float = 0.15

    # --- 보폭 -----------------------------------------------------------
    use_command_speed: bool = True  # 커맨드 속도로 d_step 을 역산
    command_name: str = "base_velocity"
    d_step_fixed: float = 0.0873    # use_command_speed=False 일 때
    d_step_max: float = 0.20
    d_step_tau: float = 0.3         # 보폭 저역통과 시정수 [s]

    # --- residual -------------------------------------------------------
    enable_residual: bool = True    # False 로 두면 CPG 단독 (디버깅용)
    enable_cpg: bool = True         # False 로 두면 기본자세 + residual
    residual_scale_revolute: float = 0.15   # [rad]
    residual_scale_prismatic: float = 0.03  # [m]


# =====================================================================
# 관측 항
# =====================================================================

def cpg_oscillator_state(env, action_term_name: str = "cpg") -> torch.Tensor:
    """발진기 상태를 관측으로 준다.

    이게 없으면 정책은 지금이 유각기인지 입각기인지 알 수 없다.
    보정을 언제 넣어야 할지 모르는 채로 학습하게 되므로 반드시 필요하다.
    """
    term = env.action_manager.get_term(action_term_name)
    return term.oscillator_state
