"""CPG + 수치 IK 코어 (Isaac Lab 비의존, 순수 torch).

Isaac 없이 CPU 에서 단독 검증할 수 있도록 Isaac 의존성을 이 파일에서 뺐다.
ActionTerm (cpg_ik_action.py) 과 오프라인 검증 스크립트
(scripts/check_cpg_ik.py) 가 같은 코드를 쓴다.

구성
    1. LegChain        : 범용 직렬 체인 FK / Jacobian (revolute + prismatic)
    2. DLSLegIK        : 가중 감쇠 최소자승 수치 IK (자세 가중치로 여유자유도 분배)
    3. GaitCPG         : 위상형 Hopf 발진기 6개 + gait 블렌딩 + 전방향 보폭
    4. GaitSelector    : 상위 규칙 (명령 속도 / 지형 거칠기 -> wave/ripple/tripod)

이전 버전(cpg_action.py) 대비 바뀐 점
    - 해석 IK(무릎 0 고정) -> 수치 IK. 무릎을 Go2 처럼 굽힌 자세를 기준으로 하고,
      발 들기는 무릎/hip 으로, 체고/레벨링은 prismatic 으로 나눠 맡긴다.
      prismatic 속도 한계(0.25 m/s)가 더 이상 보행 주파수를 묶지 않는다.
    - 직교좌표 Hopf -> 극좌표(위상형) Hopf. 결합이 위상에만 걸려 진폭이
      mu 보다 부풀지 않는다. 결합을 키워 gait 전환을 빠르게 수렴시킬 수 있다.
    - duty 를 발진기 각속도가 아니라 위상 재매핑으로 적용한다. 발진기는 등속이라
      위상차가 정확히 유지되고, duty 를 연속적으로 바꿀 수 있다.
    - 입각기 발 궤적을 등속 직선으로 바꿨다. 동시에 땅을 디딘 다리들의 상대
      속도가 0 이 되어 다리끼리 미는(미끄러지는) 현상이 없어진다.
    - vx, vy, yaw 전방향 보폭. 다리별 보폭 벡터 = (v + w x r_i) * duty / (2f).
    - 진폭 r 을 0 으로 보내면 제자리에 선다. 정지/출발이 미분방정식으로 부드럽다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:  # Isaac Lab 안에서는 configclass (hydra / to_dict 호환), 밖에서는 dataclass
    from isaaclab.utils import configclass as _cfgclass
except Exception:  # noqa: BLE001
    _cfgclass = dataclass


# =====================================================================
# 1. 범용 직렬 체인
# =====================================================================

@dataclass
class JointSpec:
    """URDF joint 하나. origin 은 부모 링크 기준 (rpy=0 가정), axis 는 단위벡터."""

    name: str
    type: str                     # "revolute" | "prismatic"
    origin: tuple                 # (x, y, z)
    axis: tuple                   # (ax, ay, az)
    lower: float
    upper: float


@dataclass
class LegChainSpec:
    """다리 하나: base -> joint 들 -> 발끝(tip) 고정 오프셋.

    tip_offset 은 마지막 관절의 자식 링크 기준. URDF 를 바꿔도
    이 구조만 다시 채우면 IK 는 그대로 쓴다 (형태 비교용).
    """

    joints: list
    tip_offset: tuple


def hugo_v3_leg_chain(hip_xyz: tuple) -> LegChainSpec:
    """Hugo ver3 / ver3.1 다리 (URDF 와 동일 수치).

    roll(x) -> hip pitch(y) -> [0.03] prismatic1(-z) -> [0.30] knee pitch(y)
            -> [0.03] prismatic2(-z) -> [0.30] 발 링크 원점
    """
    return LegChainSpec(
        joints=[
            JointSpec("roll", "revolute", hip_xyz, (1.0, 0.0, 0.0), -2.0, 2.0),
            JointSpec("hip", "revolute", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.0, 2.0),
            JointSpec("pr1", "prismatic", (0.0, 0.0, -0.03), (0.0, 0.0, -1.0), 0.0, 0.15),
            JointSpec("knee", "revolute", (0.0, 0.0, -0.30), (0.0, 1.0, 0.0), -2.0, 2.0),
            JointSpec("pr2", "prismatic", (0.0, 0.0, -0.03), (0.0, 0.0, -1.0), 0.0, 0.15),
        ],
        tip_offset=(0.0, 0.0, -0.30),
    )


# Hugo 다리 순서. 기존 cpg_action.py 와 같다.
HUGO_LEG_PREFIXES = ("R1", "R2", "R3", "L1", "L2", "L3")
HUGO_HIP_XYZ = {
    "R1": (0.45, -0.3, 0.0), "R2": (0.0, -0.3, 0.0), "R3": (-0.45, -0.3, 0.0),
    "L1": (0.45, 0.3, 0.0), "L2": (0.0, 0.3, 0.0), "L3": (-0.45, 0.3, 0.0),
}


def _skew(a: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(a[..., 0])
    return torch.stack([
        torch.stack([z, -a[..., 2], a[..., 1]], -1),
        torch.stack([a[..., 2], z, -a[..., 0]], -1),
        torch.stack([-a[..., 1], a[..., 0], z], -1),
    ], -2)


def _axis_angle_to_R(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (3,), q (B,) -> (B, 3, 3)."""
    K = _skew(axis).expand(q.shape[0], 3, 3)
    eye = torch.eye(3, device=q.device, dtype=q.dtype).expand_as(K)
    s = torch.sin(q)[:, None, None]
    c = torch.cos(q)[:, None, None]
    return eye + s * K + (1.0 - c) * (K @ K)


class LegChain:
    """같은 구조의 다리 L 개를 배치로 계산한다 (hip 위치만 다름)."""

    def __init__(self, specs: list, device="cpu", dtype=torch.float32):
        self.n_legs = len(specs)
        self.n_joints = len(specs[0].joints)
        self.device, self.dtype = device, dtype
        t = lambda v: torch.tensor(v, device=device, dtype=dtype)  # noqa: E731

        self.types = [j.type for j in specs[0].joints]
        for s in specs:
            assert [j.type for j in s.joints] == self.types, "모든 다리 구조가 같아야 한다"

        # (L, J, 3)
        self.origins = t([[j.origin for j in s.joints] for s in specs])
        self.axes = t([[j.axis for j in s.joints] for s in specs])
        self.tip = t([s.tip_offset for s in specs])                       # (L, 3)
        self.lower = t([[j.lower for j in s.joints] for s in specs])       # (L, J)
        self.upper = t([[j.upper for j in s.joints] for s in specs])
        self.is_prismatic = torch.tensor(
            [t_ == "prismatic" for t_ in self.types], device=device)

    def fk(self, q: torch.Tensor, with_jacobian: bool = True):
        """q (N, L, J) -> tip (N, L, 3) [base 기준], J (N, L, 3, J)."""
        N, L, J = q.shape
        B = N * L
        qf = q.reshape(B, J)
        R = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3).clone()
        p = torch.zeros(B, 3, device=q.device, dtype=q.dtype)

        origins = self.origins.unsqueeze(0).expand(N, L, J, 3).reshape(B, J, 3)
        axes = self.axes.unsqueeze(0).expand(N, L, J, 3).reshape(B, J, 3)

        joint_p, joint_a = [], []
        for k in range(J):
            p = p + (R @ origins[:, k, :, None]).squeeze(-1)
            a_w = (R @ axes[:, k, :, None]).squeeze(-1)
            joint_p.append(p)
            joint_a.append(a_w)
            if self.types[k] == "revolute":
                # 모든 다리의 k 번째 축이 같은 경우가 대부분이지만, 일반성을 위해
                # 다리별 축으로 계산한다.
                Rk = self._rot_batch(axes[:, k, :], qf[:, k])
                R = R @ Rk
            else:
                p = p + a_w * qf[:, k:k + 1]
        tip_off = self.tip.unsqueeze(0).expand(N, L, 3).reshape(B, 3)
        tip = p + (R @ tip_off[:, :, None]).squeeze(-1)

        if not with_jacobian:
            return tip.reshape(N, L, 3), None

        cols = []
        for k in range(J):
            if self.types[k] == "revolute":
                cols.append(torch.cross(joint_a[k], tip - joint_p[k], dim=-1))
            else:
                cols.append(joint_a[k])
        jac = torch.stack(cols, dim=-1)                                   # (B, 3, J)
        return tip.reshape(N, L, 3), jac.reshape(N, L, 3, J)

    @staticmethod
    def _rot_batch(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """axis (B,3), q (B,) -> (B,3,3)."""
        K = _skew(axis)
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand_as(K)
        s = torch.sin(q)[:, None, None]
        c = torch.cos(q)[:, None, None]
        return eye + s * K + (1.0 - c) * (K @ K)


# =====================================================================
# 2. 수치 IK
# =====================================================================

@_cfgclass
class IKWeights:
    """자세 가중치 [1/rad^2 또는 1/m^2], 과제 오차 가중치 1 [1/m^2] 기준.

    다리 하나에 관절 5개, 과제 3개 -> 여유자유도 2.
    다리 길이를 바꾸는 방법이 셋(prismatic1, prismatic2, 무릎)이므로
    가중치 비로 누가 맡을지 정한다.
        prismatic 자세 가중치 >> 무릎  ->  발 들기/보폭은 무릎+hip 이 맡고,
                                           prismatic 은 기준값에 머문다.
    prismatic 은 볼스크류라 느리고(0.25 m/s), 무릎은 빠르다(5 rad/s).
    체고/레벨링에 쓸 prismatic 은 RL residual 이 직접 움직인다.

    값을 고를 때의 기준: 과제 오차 1 cm = 1e-4.
        무릎 0.4 rad 이동 비용   = w_knee * 0.16 = 1.6e-4  (무릎 1e-3)
        같은 8 cm 를 prismatic 으로 = w_pr * 2 * 0.04^2 = 1.6e-3 (pr 0.5)
        -> 무릎이 10 배 싸다.
    """

    roll: float = 1.0e-3
    hip: float = 1.0e-4
    pr1: float = 0.5
    knee: float = 1.0e-3
    pr2: float = 0.5
    damping: float = 1.0e-4       # Levenberg-Marquardt 감쇠 (특이점 보호)
    max_step: float = 0.25        # 반복 1회 관절 변화 상한 [rad 또는 m]


class DLSLegIK:
    """min ||p* - FK(q)||^2 + (q - q_nom)^T W (q - q_nom).

    가우스-뉴턴 + 감쇠:
        (J^T J + W + lambda I) dq = J^T e + W (q_nom - q)
    이전 물리 스텝의 해로 웜스타트하므로 스텝당 2~3 회 반복이면 수렴한다.
    해석 해에 의존하지 않으므로 URDF 만 바뀌면 다른 형태에도 그대로 쓴다.
    """

    def __init__(self, chain: LegChain, weights: IKWeights, q_nom: torch.Tensor,
                 lower: torch.Tensor | None = None, upper: torch.Tensor | None = None):
        self.chain = chain
        names = [j for j in ("roll", "hip", "pr1", "knee", "pr2")]
        assert chain.n_joints == len(names)
        w = torch.tensor([getattr(weights, n) for n in names],
                         device=chain.device, dtype=chain.dtype)
        self.W = torch.diag_embed(w.expand(chain.n_legs, -1))            # (L, J, J)
        self.w_vec = w
        self.damping = weights.damping
        self.max_step = weights.max_step
        self.q_nom = q_nom                                               # (L, J)
        self.lower = chain.lower if lower is None else lower
        self.upper = chain.upper if upper is None else upper

    def solve(self, target: torch.Tensor, q: torch.Tensor, iters: int = 3,
              q_nom: torch.Tensor | None = None):
        """target (N,L,3), q 초기값 (N,L,J) -> q (N,L,J), 잔여오차 (N,L)."""
        q_nom = self.q_nom if q_nom is None else q_nom
        J_n = self.chain.n_joints
        eye = torch.eye(J_n, device=q.device, dtype=q.dtype)
        for _ in range(iters):
            tip, jac = self.chain.fk(q)
            e = target - tip                                             # (N,L,3)
            jt = jac.transpose(-1, -2)                                   # (N,L,J,3)
            A = jt @ jac + self.W + self.damping * eye
            b = (jt @ e.unsqueeze(-1)).squeeze(-1) + self.w_vec * (q_nom - q)
            dq = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
            dq = dq.clamp(-self.max_step, self.max_step)
            q = torch.max(torch.min(q + dq, self.upper), self.lower)
        tip, _ = self.chain.fk(q, with_jacobian=False)
        return q, torch.norm(target - tip, dim=-1)


# =====================================================================
# 3. 걸음새
# =====================================================================

# 위상 오프셋 (주기 단위). 다리 순서 R1 R2 R3 L1 L2 L3. 기존 값과 동일.
GAIT_NAMES = ("wave", "ripple", "tripod")          # 느리고 안정 -> 빠름
GAIT_PHASES = {
    "tripod": (0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    "ripple": (0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6),
    "wave":   (2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6),
}
GAIT_DUTY = {"tripod": 0.5, "ripple": 2 / 3, "wave": 5 / 6}


@_cfgclass
class GaitParams:
    # 걸음새별 보행 주파수 [Hz].  최대 속도 = 2 * stride_max * f / duty
    #   wave   0.40 Hz -> 0.144 m/s,  ripple 0.55 Hz -> 0.248 m/s,
    #   tripod 0.65 Hz -> 0.390 m/s   (stride_max 0.15 m)
    # wave 는 유각기가 주기의 1/6 뿐이라 발을 같은 높이로 드는 무릎 속도가 가장
    # 크다. 0.45 Hz 에서 무릎 3.4 rad/s (한계 5) 였으므로 0.40 으로 낮춘다.
    freq_wave: float = 0.40
    freq_ripple: float = 0.55
    freq_tripod: float = 0.65
    stride_max: float = 0.15       # 발끝 보폭 반진폭 상한 [m]
    swing_lift: float = 0.08       # 유각기 발 들림 [m]
    # gait 전환 블렌딩 시간 [s]. 전환 중에는 위상 오프셋이 움직이므로 일부 다리의
    # 위상 속도가 빨라진다. 1.0 s 에서 무릎 4.8 rad/s (한계 96%) 까지 튀었다.
    transition_time: float = 2.0
    coupling: float = 2.0          # 위상 결합 세기 [rad/s]. 6k ~ 수렴 속도
    amp_alpha: float = 5.0         # 진폭 수렴 속도 [1/s] (정지/출발 시정수 0.2 s)
    cmd_tau: float = 0.3           # 명령 저역통과 시정수 [s]
    stand_speed: float = 0.03      # 다리별 요구 속도가 이보다 작으면 선다 [m/s]


def _wrap01(x: torch.Tensor) -> torch.Tensor:
    """주기 단위 차이를 (-0.5, 0.5] 로."""
    return x - torch.round(x)


def _smoothstep(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


class GaitCPG:
    """위상형 Hopf 발진기 + gait 블렌딩 + 전방향 보폭 + 발끝 궤적.

    발진기 (다리 i):
        phi_i' = 2 pi f + k * sum_j sin(phi_j - phi_i - (off_j - off_i))
        r_i'   = alpha * (mu - r_i)        mu = 1 (걷기) / 0 (서기)
    직교좌표 Hopf 와 극한 순환이 같다 (r = mu 로 수렴하는 원).
    결합이 위상에만 걸리므로 진폭이 부풀지 않는다.

    duty 는 위상 재매핑으로 넣는다 (u = phi / 2pi):
        u in [0, 1-beta)  -> 유각기, s_sw = u / (1-beta)
        u in [1-beta, 1)  -> 입각기, s_st = (u - (1-beta)) / beta
    """

    def __init__(self, num_envs: int, nominal_feet: torch.Tensor, params: GaitParams,
                 device="cpu", dtype=torch.float32, initial_gait: str = "tripod"):
        self.N = num_envs
        self.L = nominal_feet.shape[0]
        self.p = params
        self.device, self.dtype = device, dtype
        self.nominal = nominal_feet.to(device, dtype)                    # (L, 3) base 기준

        t = lambda v: torch.tensor(v, device=device, dtype=dtype)  # noqa: E731
        self.gait_off = torch.stack([t(GAIT_PHASES[g]) for g in GAIT_NAMES])   # (G, L)
        self.gait_duty = t([GAIT_DUTY[g] for g in GAIT_NAMES])                  # (G,)
        self.gait_freq = t([getattr(params, f"freq_{g}") for g in GAIT_NAMES])               # (G,)
        # gait 별 최대 속도 [m/s] = 2 * stride_max * f / duty  (보폭 상한에서 나온다)
        self.gait_cap = 2.0 * params.stride_max * self.gait_freq / self.gait_duty             # (G,)
        g0 = GAIT_NAMES.index(initial_gait)

        N, L = self.N, self.L
        self.phi = torch.zeros(N, L, device=device, dtype=dtype)          # [0, 1) 주기 단위
        self.r = torch.zeros(N, L, device=device, dtype=dtype)
        self.gait_cur = torch.full((N,), g0, device=device, dtype=torch.long)   # 목표(도착) gait
        self.off_from = self.gait_off[self.gait_cur].clone()
        self.duty_from = self.gait_duty[self.gait_cur].clone()
        self.freq_from = self.gait_freq[self.gait_cur].clone()
        self.blend = torch.ones(N, device=device, dtype=dtype)            # 1 = 전환 완료
        self.cmd_f = torch.zeros(N, 3, device=device, dtype=dtype)        # 저역통과 명령
        self.stride = torch.zeros(N, L, 2, device=device, dtype=dtype)
        self.reset(None)

    # ------------------------------------------------------------------
    def current(self):
        """블렌딩 중인 (위상 오프셋 (N,L), duty (N,), freq (N,))."""
        s = _smoothstep(self.blend)
        off_to = self.gait_off[self.gait_cur]
        off = self.off_from + s[:, None] * _wrap01(off_to - self.off_from)
        duty = self.duty_from + s * (self.gait_duty[self.gait_cur] - self.duty_from)
        freq = self.freq_from + s * (self.gait_freq[self.gait_cur] - self.freq_from)
        return off, duty, freq

    def request_gait(self, gait_idx: torch.Tensor, immediate: torch.Tensor | None = None):
        """gait_idx (N,) long. 바뀐 env 만 현재 블렌딩 상태에서 새 전환을 시작한다.

        immediate=True 인 env 는 블렌딩 없이 바로 바꾼다 (리셋 직후 첫 결정).
        """
        change = gait_idx != self.gait_cur
        if immediate is not None:
            imm = immediate & change
        else:
            imm = torch.zeros_like(change)
        smooth = change & ~imm
        if smooth.any():
            off, duty, freq = self.current()
            self.off_from[smooth] = off[smooth]
            self.duty_from[smooth] = duty[smooth]
            self.freq_from[smooth] = freq[smooth]
            self.blend[smooth] = 0.0
            self.gait_cur[smooth] = gait_idx[smooth]
        if imm.any():
            g = gait_idx[imm]
            self.gait_cur[imm] = g
            self.off_from[imm] = self.gait_off[g]
            self.duty_from[imm] = self.gait_duty[g]
            self.freq_from[imm] = self.gait_freq[g]
            self.blend[imm] = 1.0
            # 위상도 목표 패턴 위로 바로 옮긴다 (극한 순환 위 초기화)
            base = self.phi[imm, 0:1] - self.gait_off[g][:, 0:1]
            self.phi[imm] = torch.remainder(base + self.gait_off[g], 1.0)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        off = self.gait_off[self.gait_cur[ids]]
        self.phi[ids] = off                # 목표 위상 위에서 시작 (극한 순환 위)
        self.r[ids] = 0.0                  # 선 자세에서 출발 -> 리셋 과도구간 없음
        self.off_from[ids] = off
        self.duty_from[ids] = self.gait_duty[self.gait_cur[ids]]
        self.freq_from[ids] = self.gait_freq[self.gait_cur[ids]]
        self.blend[ids] = 1.0
        self.cmd_f[ids] = 0.0
        self.stride[ids] = 0.0

    # ------------------------------------------------------------------
    def leg_demand(self, cmd: torch.Tensor) -> torch.Tensor:
        """다리별 요구 발 속도 (N, L, 2) = v + w x r_i  (base 평면)."""
        x = self.nominal[:, 0].unsqueeze(0)
        y = self.nominal[:, 1].unsqueeze(0)
        vx = cmd[:, 0:1] - cmd[:, 2:3] * y
        vy = cmd[:, 1:2] + cmd[:, 2:3] * x
        return torch.stack([vx, vy], dim=-1)

    def speed_cap(self) -> torch.Tensor:
        """지금 낼 수 있는 다리별 최대 발 속도 (N,) [m/s].

        전환 중에는 출발 gait 와 도착 gait 중 작은 값을 쓴다. 전환이 끝나기 전에는
        느린 쪽 gait 의 속도까지만 확실히 낼 수 있기 때문이다.
        """
        cap_to = self.gait_cap[self.gait_cur]
        cap_from = 2.0 * self.p.stride_max * self.freq_from / self.duty_from
        return torch.where(self.blend < 1.0, torch.minimum(cap_from, cap_to), cap_to)

    def command_scale(self, cmd: torch.Tensor) -> torch.Tensor:
        """명령 (N,3) 을 지금 gait 가 낼 수 있는 속도 안으로 줄이는 배율 (N,), 1 이하.

        기준은 몸통 속도가 아니라 가장 빠른 다리의 요구 속도 (v + w x r) 다. 회전 명령이
        섞이면 바깥쪽 다리가 더 빨리 움직여야 하므로 몸통 속도만 보면 부족하다.
        vx, vy, wz 를 같은 배율로 줄여 진행 방향과 곡률은 유지한다.
        """
        speed = torch.norm(self.leg_demand(cmd), dim=-1).amax(dim=1)
        return torch.clamp(self.speed_cap() / speed.clamp_min(1e-9), max=1.0)

    def step(self, dt: float, cmd: torch.Tensor):
        """한 물리 스텝. cmd (N,3) = [vx, vy, wz] base 기준.

        반환: 발끝 목표 (N, L, 3) base 기준.
        """
        p = self.p
        # --- 명령 저역통과 (재샘플링 계단 -> 부드러운 보폭 변화) --------
        a = dt / (p.cmd_tau + dt)
        self.cmd_f += a * (cmd - self.cmd_f)

        # --- gait 블렌딩 진행 --------------------------------------------
        self.blend = (self.blend + dt / max(p.transition_time, 1e-6)).clamp(max=1.0)
        off, duty, freq = self.current()

        # --- 발진기 적분 ---------------------------------------------------
        dphi = self.phi.unsqueeze(1) - self.phi.unsqueeze(2)      # [i, j] = phi_j - phi_i
        doff = off.unsqueeze(1) - off.unsqueeze(2)                # off_j - off_i
        coup = torch.sin(2.0 * math.pi * (dphi - doff)).sum(dim=2)  # (N, L)
        phi_dot = freq.unsqueeze(1) + p.coupling / (2.0 * math.pi) * coup
        self.phi = torch.remainder(self.phi + dt * phi_dot, 1.0)

        demand = self.leg_demand(self.cmd_f)                       # (N, L, 2)
        speed = torch.norm(demand, dim=-1).amax(dim=1)             # (N,)
        mu = (speed > p.stand_speed).to(self.dtype).unsqueeze(1)
        self.r = self.r + dt * p.amp_alpha * (mu - self.r)

        # --- 보폭: v = 2 a f / duty  ->  a = v duty / (2 f) ---------------
        stride = demand * (duty / (2.0 * freq))[:, None, None]     # (N, L, 2)
        mag = torch.norm(stride, dim=-1).amax(dim=1).clamp_min(1e-9)
        scale = torch.clamp(p.stride_max / mag, max=1.0)           # 곡률 유지하며 같이 줄임
        self.stride = stride * scale[:, None, None]

        return self.foot_targets(duty)

    def foot_targets(self, duty: torch.Tensor) -> torch.Tensor:
        u = self.phi
        b = duty.unsqueeze(1)
        swing = u < (1.0 - b)
        s_sw = (u / (1.0 - b)).clamp(0.0, 1.0)
        s_st = ((u - (1.0 - b)) / b).clamp(0.0, 1.0)

        # 진행 방향 정규화 좌표 xi: +1 = 뒤 끝(입각 종료), -1 = 앞 끝(착지)
        #   유각기: 코사인으로 +1 -> -1 (양 끝 속도 0)
        #   입각기: 등속 직선 -1 -> +1  (몸통과 같은 속도로 발이 뒤로 감)
        xi = torch.where(swing, torch.cos(math.pi * s_sw), -1.0 + 2.0 * s_st)
        lift = torch.where(swing, torch.sin(math.pi * s_sw), torch.zeros_like(u))

        r = self.r
        tgt = self.nominal.unsqueeze(0).expand(self.N, -1, -1).clone()
        tgt[..., 0:2] = tgt[..., 0:2] - (r * xi).unsqueeze(-1) * self.stride
        tgt[..., 2] = tgt[..., 2] + r * self.p.swing_lift * lift
        self._swing = swing
        return tgt

    # ------------------------------------------------------------------
    def observation(self) -> torch.Tensor:
        """(N, 2L): 재매핑 위상의 cos/sin x 진폭. sin > 0 이면 유각기."""
        _, duty, _ = self.current()
        u = self.phi
        b = duty.unsqueeze(1)
        psi = torch.where(u < 1.0 - b, math.pi * u / (1.0 - b),
                          math.pi + math.pi * (u - (1.0 - b)) / b)
        return torch.cat([self.r * torch.cos(psi), self.r * torch.sin(psi)], dim=1)

    def gait_observation(self) -> torch.Tensor:
        """(N, 6): 목표 gait one-hot 3 + 블렌드 진행 + duty + freq."""
        _, duty, freq = self.current()
        onehot = torch.nn.functional.one_hot(self.gait_cur, len(GAIT_NAMES)).to(self.dtype)
        return torch.cat([onehot, self.blend[:, None], duty[:, None], freq[:, None]], dim=1)


# =====================================================================
# 4. 상위 규칙: 걸음새 선택
# =====================================================================

@_cfgclass
class SelectorParams:
    # 다리별 최대 요구 속도 기준 [m/s]. 각 gait 최대 속도보다 충분히 낮게.
    up_ripple: float = 0.11      # wave -> ripple
    up_tripod: float = 0.19      # ripple -> tripod
    hysteresis: float = 0.02
    rough_threshold: float = 0.025  # 평면 제거 후 높이 스캔 표준편차 [m] (격자 ~6 cm 이상)
    # 거칠기 값은 격자/계단 위에서 기준 근처 (중앙값 0.025~0.035) 를 오르내린다. 그대로 쓰면
    # 걸음새가 2 초마다 오갔다 (전환의 30 % 가 3 초 안에 되돌아감). 그래서
    #   1) 지수이동평균으로 고르고 (시정수 rough_tau)
    #   2) 험지 판정에 히스테리시스를 둔다 (기준 + h 를 넘어야 험지, 기준 - h 밑이어야 평지)
    rough_tau: float = 1.0          # [s]
    rough_hysteresis: float = 0.005  # [m]
    decide_period: float = 0.5      # 결정 주기 [s]
    # 한 gait 최소 유지 시간 [s]. 전환 블렌딩이 2.0 s 걸리므로 그보다 길어야
    # 전환이 끝난 걸음새로 실제로 걷는 구간이 생긴다 (1.5 -> 3.0).
    min_dwell: float = 3.0
    threshold_jitter: float = 0.0   # 학습 시 env 별 임계값 흔들기 [m/s]


class GaitSelector:
    """명령 속도 + 지형 거칠기로 wave(0)/ripple(1)/tripod(2) 를 고른다.

    - 빠를수록 duty 가 작은 gait (속도 상한 때문)
    - 거칠면 한 단계 안정한 쪽 (속도가 허락할 때만)
    - 전환은 인접 gait 로만 (tripod <-> wave 직행 금지)
    - 최소 유지 시간으로 떨림(chattering) 방지
    """

    def __init__(self, num_envs: int, params: SelectorParams, cap: torch.Tensor,
                 device="cpu", dtype=torch.float32):
        self.p = params
        self.cap = cap                                        # (G,) gait 별 최대 속도
        self.timer = torch.zeros(num_envs, device=device, dtype=dtype)
        self.dwell = torch.full((num_envs,), 1e3, device=device, dtype=dtype)
        self.jitter = torch.zeros(num_envs, device=device, dtype=dtype)
        self.rough_f = torch.zeros(num_envs, device=device, dtype=dtype)      # 평활화한 거칠기
        self.is_rough = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.timer[ids] = 0.0
        self.dwell[ids] = 1e3
        self.rough_f[ids] = 0.0
        self.is_rough[ids] = False
        if self.p.threshold_jitter > 0.0:
            n = self.jitter[ids].numel()
            self.jitter[ids] = (torch.rand(n, device=self.jitter.device) * 2 - 1) * self.p.threshold_jitter

    def update_roughness(self, dt: float, rough: torch.Tensor, first=None) -> torch.Tensor:
        """거칠기를 고르고 히스테리시스를 걸어 험지 여부 (N,) bool 을 갱신한다."""
        p = self.p
        a = dt / (p.rough_tau + dt)
        self.rough_f = self.rough_f + a * (rough - self.rough_f)
        if first is not None:
            # 리셋 직후에는 지금 값으로 바로 정한다 (0 에서 서서히 올라오지 않게)
            self.rough_f = torch.where(first, rough, self.rough_f)
            self.is_rough = torch.where(first, rough > p.rough_threshold, self.is_rough)
        self.is_rough = torch.where(self.is_rough,
                                    self.rough_f > p.rough_threshold - p.rough_hysteresis,
                                    self.rough_f > p.rough_threshold + p.rough_hysteresis)
        return self.is_rough

    def desired(self, speed: torch.Tensor, is_rough: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        p = self.p
        j = self.jitter
        # 히스테리시스: 현재보다 위로 가려면 +h, 아래로 가려면 -h
        up_r = p.up_ripple + j + torch.where(cur >= 1, -p.hysteresis, p.hysteresis)
        up_t = p.up_tripod + j + torch.where(cur >= 2, -p.hysteresis, p.hysteresis)
        g = (speed > up_r).long() + (speed > up_t).long()
        # 거친 지형이면 한 단계 낮춘다 (그 gait 상한의 90% 안일 때만).
        # 이미 낮은 gait 에 있으면 90% 까지 머물고, 새로 낮추려면 h 만큼 더 느려야 한다.
        # 이 여유가 없으면 낮추는 기준과 올리는 기준이 같은 속도가 되어 그 근처에서 오간다.
        # (여유를 2h 로 하면 낮출 수 있는 속도 구간이 거의 사라져 험지 규칙이 죽는다.)
        #   험지에서: wave 는 0.11 미만에서 들어가 0.13 까지, ripple 은 0.20 미만에서 들어가 0.22 까지
        lower = (g - 1).clamp(min=0)
        margin = torch.where(cur <= lower, torch.zeros_like(speed), torch.full_like(speed, p.hysteresis))
        can_lower = speed < 0.9 * self.cap[lower] - margin
        g = torch.where(is_rough & can_lower & (g > 0), lower, g)
        return g

    def step(self, dt: float, speed, rough, cur, first=None):
        """반환: 요청할 gait (N,), 즉시 적용 마스크 (N,)."""
        self.timer += dt
        self.dwell += dt
        want = self.desired(speed, self.update_roughness(dt, rough, first), cur)
        decide = self.timer >= self.p.decide_period
        self.timer = torch.where(decide, torch.zeros_like(self.timer), self.timer)
        nxt = cur + torch.sign(want - cur)                     # 인접 gait 로만
        ok = decide & (self.dwell >= self.p.min_dwell) & (nxt != cur)
        imm = torch.zeros_like(ok) if first is None else first
        out = torch.where(ok, nxt, cur)
        out = torch.where(imm, want, out)
        self.dwell = torch.where(ok | imm, torch.zeros_like(self.dwell), self.dwell)
        return out, imm


def gait_speed_caps(params: GaitParams, device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(
        [2.0 * params.stride_max * getattr(params, f"freq_{g}") / GAIT_DUTY[g] for g in GAIT_NAMES],
        device=device, dtype=dtype)


# =====================================================================
# 5. Hugo 기본 설정 헬퍼
# =====================================================================

@_cfgclass
class HugoPostureCfg:
    """기준 자세.

    stance_height 는 hip -> 발 링크 원점의 수직 거리. 발바닥 두께 0.02 를 더하면
    지면 기준 몸통(hip) 높이가 된다. 이전 0.84 (무릎 0, 거의 편 다리) 에서
    0.72 로 낮춰 무게중심을 내리고 무릎을 굽힌다.

    prismatic 기준값은 행정 중앙(0.075). 위아래 어느 쪽으로도 ±0.07 m 여유가
    있어 RL residual 이 체고/레벨링에 쓸 수 있다.

    knee_sign = -1: 무릎이 hip-발 선보다 뒤로 꺾이는 Go2 식 (hip +, knee -).
    """

    stance_height: float = 0.72
    roll_splay: float = 0.20
    prismatic_nominal: float = 0.075
    knee_sign: float = -1.0
    knee_limit_bent: float = 0.20     # 무릎이 이보다 펴지지 않게 IK 한계 [rad]
    foot_x_offset: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    prismatic_margin: float = 0.005


def build_hugo(posture: HugoPostureCfg, device="cpu", dtype=torch.float32,
               leg_prefixes=HUGO_LEG_PREFIXES, weights: IKWeights | None = None):
    """체인, IK, 기준 발 위치, 기준 관절각을 만든다.

    반환: chain, ik, nominal_feet (L,3), q_nom (L,J)
    """
    specs = [hugo_v3_leg_chain(HUGO_HIP_XYZ[pre]) for pre in leg_prefixes]
    chain = LegChain(specs, device=device, dtype=dtype)
    L, J = chain.n_legs, chain.n_joints
    hip = chain.origins[:, 0, :]                                       # (L, 3)

    side = torch.tensor([1.0 if pre.startswith("L") else -1.0 for pre in leg_prefixes],
                        device=device, dtype=dtype)
    h = posture.stance_height
    nominal = hip.clone()
    nominal[:, 0] += torch.tensor(posture.foot_x_offset, device=device, dtype=dtype)
    nominal[:, 1] += side * h * math.tan(posture.roll_splay)
    nominal[:, 2] = -h

    # 관절 한계: 무릎은 굽힌 쪽으로만, prismatic 은 끝단 여유
    lower = chain.lower.clone()
    upper = chain.upper.clone()
    m = posture.prismatic_margin
    lower[:, 2] = m
    upper[:, 2] = 0.15 - m
    lower[:, 4] = m
    upper[:, 4] = 0.15 - m
    if posture.knee_sign < 0:
        upper[:, 3] = -posture.knee_limit_bent
    else:
        lower[:, 3] = posture.knee_limit_bent

    # 초기 자세: 무릎 굽힘을 해석적으로 대략 잡고 IK 로 정밀화
    l1 = 0.33 + posture.prismatic_nominal
    l2 = 0.33 + posture.prismatic_nominal
    reach = torch.norm(nominal - hip, dim=-1)
    cos_k = ((reach ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)).clamp(-1, 1)
    knee0 = posture.knee_sign * torch.acos(cos_k)
    q0 = torch.zeros(L, J, device=device, dtype=dtype)
    q0[:, 0] = side * posture.roll_splay
    q0[:, 1] = -knee0 / 2.0
    q0[:, 2] = posture.prismatic_nominal
    q0[:, 3] = knee0
    q0[:, 4] = posture.prismatic_nominal

    w = weights or IKWeights()
    ik = DLSLegIK(chain, w, q_nom=q0, lower=lower, upper=upper)
    q, err = ik.solve(nominal.unsqueeze(0), q0.unsqueeze(0), iters=50)
    q_nom = q[0]
    ik.q_nom = q_nom
    if err.max() > 1e-4:
        raise RuntimeError(f"기준 자세 IK 수렴 실패: {err.max().item():.2e} m")
    return chain, ik, nominal, q_nom
