"""
CPG 3a단계 - 매핑 함수 + 역기구학 검증 (numpy, Isaac 불필요)

실행:  python cpg_step3a_mapping_ik.py
필요:  numpy, matplotlib

2단계에서 발진기 6개가 걸음새를 만드는 것까지 확인했다.
이제 (x, y)를 발끝 위치로 바꾸고, 그것을 관절값으로 푼다.
Isaac에 붙이기 전에 ver3.1 하드웨어 한계를 전부 통과하는지 확인한다.

검사 항목
  1. IK -> FK 왕복 오차 (부호 규약이 맞는가)
  2. 다리 길이 L 이 [0.66, 0.96] m 안인가
  3. prismatic 변위 d1, d2 가 [0, 0.15] m 안인가
  4. prismatic 속도가 0.25 m/s 를 넘지 않는가   <- 가장 빡빡함
  5. hip pitch 각도가 action scale 0.3 rad 안인가
  6. hip pitch 각속도가 6.0 rad/s 안인가
  7. 유각기 발 들림이 충분한가
  8. 이론 보행 속도가 커맨드 범위와 맞는가
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# ver3.1 하드웨어 상수 (URDF / env_cfg 에서 그대로 가져옴)
# =====================================================================

L0 = 0.66            # prismatic 완전 수축 시 hip -> 발 링크 원점 [m]
STROKE = 0.15        # prismatic 관절당 스트로크 [m]
FOOT_OFFSET = 0.02   # 발 실린더 아랫면까지 [m]

PRISMATIC_VEL_LIMIT = 0.25   # [m/s]  LEY63 카탈로그
HIP_VEL_LIMIT = 6.0          # [rad/s]
JOINT_LIMIT = 2.0            # [rad]  URDF
HIP_ACTION_SCALE = 0.30      # [rad]  정책 도달 범위
ROLL_NOMINAL = 0.20          # [rad]  좌 +, 우 -  (발 간격 0.9 m 확보)

# 다리 순서와 hip 위치 (base_link 기준)
LEG_NAMES = ["R1", "R2", "R3", "L1", "L2", "L3"]
HIP_XY = np.array([
    [0.45, -0.30], [0.00, -0.30], [-0.45, -0.30],
    [0.45,  0.30], [0.00,  0.30], [-0.45,  0.30],
])
ROLL_SIGN = np.array([-1.0, -1.0, -1.0, +1.0, +1.0, +1.0])
N_LEGS = 6


# =====================================================================
# CPG 파라미터 (선택지 A)
# =====================================================================

MU = 1.0
OMEGA = 3.5          # rad/s -> 1.003 Hz. 다리 26.7 kg 의 스윙 관성 고려해 낮춤
ALPHA = 50.0
K_COUPLE = 0.5
DT = 1.0 / 200.0     # SIM_DT

H_STANCE = 0.80      # hip -> 발 링크 원점, 입각기 기준 [m]
K1_LIFT = 0.11       # 유각기 최대 발 들림 [m]   (h - k1 = 0.70 >= L0)
K2_PUSH = 0.008      # 입각기 지면 압입 [m]

DUTY = 0.5           # tripod

GAITS = {
    "tripod": np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
    "ripple": np.array([0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6]),
    "wave":   np.array([2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6]),
}


def d_step_from_speed(v_cmd, duty=DUTY, omega=OMEGA):
    """커맨드 속도로부터 보폭 진폭을 역산한다.

    입각기 동안 발은 몸통 기준 +d 에서 -d 로, 즉 2d 만큼 쓸린다.
    그 사이 시간이 duty*T 이므로 몸통 속도는

        v = 2 * d_step / (duty * T) = 2 * d_step * f / duty

    따라서  d_step = v * duty / (2 * f).

    주의: duty 가 나눗셈에 들어간다. tripod(0.5)는 두 조가 번갈아
    반 주기씩 몸을 밀어내므로 한 주기에 4*d_step 을 간다.
    """
    f = omega / (2.0 * np.pi)
    return v_cmd * duty / (2.0 * f)


# =====================================================================
# 발진기 (2단계와 동일)
# =====================================================================

def phase_matrix(gait_vector):
    phi = 2.0 * np.pi * np.asarray(gait_vector, float)
    return phi[:, None] - phi[None, :]


def network_derivative(x, y, theta, mu, omega, alpha, k):
    r_sq = x * x + y * y
    radial = alpha * (mu * mu - r_sq)
    x_dot = radial * x - omega * y
    y_dot = radial * y + omega * x
    if k != 0.0:
        C, S = np.cos(theta), np.sin(theta)
        np.fill_diagonal(C, 0.0)
        np.fill_diagonal(S, 0.0)
        x_dot += k * (C @ x - S @ y)
        y_dot += k * (S @ x + C @ y)
    return x_dot, y_dot


def run_cpg(theta, duration, gait_vector=None, seed=0):
    """gait_vector 를 주면 극한 순환 위에서 목표 위상으로 바로 시작한다.

    무작위 초기값으로 시작하면 수렴 과정에서 다리 길이가 급격히 변해
    prismatic 속도가 한계를 크게 넘는다. Isaac 에서 매 리셋마다 그런
    과도구간이 생기면 발이 끌리거나 액추에이터가 포화된다.
    그래서 리셋 시에는 반드시 이 방식으로 초기화해야 한다.
    """
    if gait_vector is not None:
        ang = 2.0 * np.pi * np.asarray(gait_vector, float)
        x, y = MU * np.cos(ang), MU * np.sin(ang)
    else:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, N_LEGS)
        y = rng.uniform(-1, 1, N_LEGS)

    n = int(duration / DT)
    t = np.zeros(n)
    X = np.zeros((n, N_LEGS))
    Y = np.zeros((n, N_LEGS))
    X[0], Y[0] = x, y

    for i in range(n - 1):
        xd, yd = network_derivative(X[i], Y[i], theta, MU, OMEGA, ALPHA, K_COUPLE)
        X[i + 1] = X[i] + DT * xd
        Y[i + 1] = Y[i] + DT * yd
        t[i + 1] = t[i] + DT
    return t, X, Y


# =====================================================================
# 매핑 함수:  (x, y)  ->  발끝 위치 (hip 좌표계)
# =====================================================================

def map_to_foot(X, Y, d_step, h=H_STANCE, k1=K1_LIFT, k2=K2_PUSH,
                roll_nominal=ROLL_NOMINAL):
    """발진기 상태를 hip 좌표계의 발끝 위치로 바꾼다.

    먼저 '다리 평면' 안에서 2D 궤적을 만든다.
        px_l = -d_step * x          앞뒤
        pz_l = -h + k1*y  (y>0)     유각기: 발을 든다
             = -h + k2*y  (y<=0)    입각기: 살짝 눌러 접지를 유지
    x 가 +1 -> 0 -> -1 로 가는 위쪽 반 바퀴에서 발이 뒤에서 앞으로 이동하고,
    아래쪽 반 바퀴에서 앞에서 뒤로 쓸리며 몸통을 민다.

    그 다음 평면 전체를 roll_nominal 만큼 바깥으로 기울인다.
        p = Rx(roll_nom) * (px_l, 0, pz_l)
    이렇게 하면 IK 를 풀었을 때 roll 이 정확히 roll_nom 으로 나온다.
    기본 자세의 벌림(좌 +0.2 / 우 -0.2, 발 간격 0.9 m)이 그대로 유지된다.
    """
    px_l = -d_step * X
    pz_l = np.where(Y > 0.0, -h + k1 * Y, -h + k2 * Y)

    roll_nom = roll_nominal * ROLL_SIGN[None, :]
    c, s = np.cos(roll_nom), np.sin(roll_nom)

    px = px_l
    py = -s * pz_l
    pz = c * pz_l
    return np.stack([px, py, pz], axis=-1)


# =====================================================================
# 역기구학  (무릎 = 0 고정)
# =====================================================================

def inverse_kinematics(p, L0=L0, stroke=STROKE):
    """발끝 위치 -> (roll, hip pitch, d1, d2).

    무릎을 0 으로 두면 다리가 직선 막대가 되고, hip 의 roll/pitch 는
    방향만, prismatic 은 길이만 담당한다. 즉 구면좌표 변환이다.

        L      = |p|
        roll   = atan2(py, -pz)
        pitch  = -asin(px / L)
        d1=d2  = (L - L0) / 2

    삼각형을 풀 일도, 해가 둘로 갈릴 일도 없다.
    """
    L = np.linalg.norm(p, axis=-1)
    roll = np.arctan2(p[..., 1], -p[..., 2])
    pitch = -np.arcsin(np.clip(p[..., 0] / L, -1.0, 1.0))
    d = (L - L0) / 2.0
    d_clipped = np.clip(d, 0.0, stroke)
    return roll, pitch, d_clipped, d_clipped, L, d


def forward_kinematics(roll, pitch, d1, d2, L0=L0):
    """검산용 순기구학.  p = Rx(roll) * Ry(pitch) * (0, 0, -L)"""
    L = L0 + d1 + d2
    px = -L * np.sin(pitch)
    py = L * np.sin(roll) * np.cos(pitch)
    pz = -L * np.cos(roll) * np.cos(pitch)
    return np.stack([px, py, pz], axis=-1)


# =====================================================================
# 검사
# =====================================================================

def check(t, X, Y, d_step, label):
    p = map_to_foot(X, Y, d_step)
    roll, pitch, d1, d2, L, d_raw = inverse_kinematics(p)

    print("=" * 74)
    print(f"{label}   (d_step = {d_step:.4f} m)")
    print("=" * 74)

    ok_all = True

    def verdict(name, value, limit, unit, mode="max"):
        nonlocal ok_all
        if mode == "max":
            ok = value <= limit
            rel = value / limit * 100.0
        else:
            ok = value >= limit
            rel = limit / max(value, 1e-9) * 100.0
        ok_all &= ok
        print(f"  {name:<34}{value:9.4f} {unit:<7}"
              f"한계 {limit:7.3f}  ({rel:5.1f}%)  {'OK' if ok else 'NG'}")

    # 1. IK -> FK 왕복
    p_back = forward_kinematics(roll, pitch, d1, d2)
    rt_err = np.abs(p_back - p).max()
    print(f"\n  [1] IK->FK 왕복 최대 오차           {rt_err:.3e} m  "
          f"{'OK' if rt_err < 1e-9 else 'NG'}")
    ok_all &= rt_err < 1e-9

    # 2. 다리 길이
    print()
    print(f"  [2] 다리 길이 L      최소 {L.min():.4f}  최대 {L.max():.4f} m"
          f"   (허용 {L0:.2f} ~ {L0+2*STROKE:.2f})")
    l_ok = (L.min() >= L0) and (L.max() <= L0 + 2 * STROKE)
    ok_all &= l_ok
    print(f"      {'OK' if l_ok else 'NG  <- 매핑이 도달 불가능한 위치를 요구함'}")

    # 3. prismatic 변위 (clip 전 원시값으로 판정)
    print()
    print(f"  [3] prismatic 변위   최소 {d_raw.min():.4f}  최대 {d_raw.max():.4f} m"
          f"   (허용 0.000 ~ {STROKE:.3f})")
    d_ok = (d_raw.min() >= 0.0) and (d_raw.max() <= STROKE)
    ok_all &= d_ok
    clipped = np.mean((d_raw < 0.0) | (d_raw > STROKE)) * 100.0
    print(f"      clip 발생 비율 {clipped:.2f}%   {'OK' if d_ok else 'NG'}")

    # 4~6. 속도 계열
    print()
    d_vel = np.abs(np.gradient(d1, DT, axis=0)).max()
    verdict("[4] prismatic 최대 속도", d_vel, PRISMATIC_VEL_LIMIT, "m/s")

    pitch_vel = np.abs(np.gradient(pitch, DT, axis=0)).max()
    verdict("[6] hip pitch 최대 각속도", pitch_vel, HIP_VEL_LIMIT, "rad/s")

    verdict("[5] hip pitch 최대 각도", np.abs(pitch).max(),
            HIP_ACTION_SCALE, "rad")
    verdict("    roll 최대 각도", np.abs(roll).max(), JOINT_LIMIT, "rad")

    # 7. 발 들림 (입각기 최저점 대비)
    print()
    foot_z = p[..., 2]
    lift = foot_z.max() - foot_z.min()
    print(f"  [7] 발끝 높이 변화량                {lift:9.4f} m")
    print(f"      입각기 최저 z {foot_z.min():.4f}   유각기 최고 z {foot_z.max():.4f}")

    # 8. 이론 속도
    f = OMEGA / (2.0 * np.pi)
    v = 2.0 * d_step * f / DUTY
    print()
    print(f"  [8] 이론 보행 속도                  {v:9.4f} m/s   "
          f"(주파수 {f:.3f} Hz, duty {DUTY})")

    # 부가 정보
    print()
    print(f"      roll 실측 (좌/우)  {roll[:, 3].mean():+.4f} / {roll[:, 0].mean():+.4f} rad")
    print(f"      발 간격 (좌우)     {(p[:,3,1].mean()+0.30) - (p[:,0,1].mean()-0.30):.4f} m")
    print(f"      prismatic 평균     {d1.mean():.4f} m   "
          f"표준편차 {d1.std():.4f} m  (스트로크의 {d1.std()/STROKE*100:.1f}%)")

    print()
    print(f"  >>> 종합: {'전 항목 통과' if ok_all else '한계 위반 있음'}")
    print()
    return p, roll, pitch, d1, L, ok_all


# =====================================================================
# 그림
# =====================================================================

def draw(t, p, roll, pitch, d1, L, d_step):
    period = 2.0 * np.pi / OMEGA
    mask = t > t[-1] - 2.0 * period
    tt = t[mask]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    # (a) 다리 평면 안의 발끝 궤적 - R1
    a = ax[0, 0]
    px = p[mask, 0, 0]
    pz_l = p[mask, 0, 2] / np.cos(ROLL_NOMINAL * ROLL_SIGN[0])
    a.plot(px, pz_l, linewidth=1.4)
    a.plot(px[0], pz_l[0], "o", color="black", markersize=5)
    a.axhline(-H_STANCE, color="gray", linestyle=":", linewidth=0.9)
    a.axhline(-L0, color="red", linestyle="--", linewidth=1.0)
    a.text(px.min(), -L0 + 0.004, "L0 = 0.66 m hard floor", fontsize=7, color="red")
    a.set_title("(a) foot path in leg plane, R1")
    a.set_xlabel("forward  px [m]")
    a.set_ylabel("down  pz [m]")
    a.set_aspect("equal")
    a.grid(alpha=0.3)

    # (b) 6개 다리 발끝 높이
    a = ax[0, 1]
    for j in range(N_LEGS):
        a.plot(tt, p[mask, j, 2], linewidth=0.9, label=LEG_NAMES[j])
    a.set_title("(b) foot height, all legs")
    a.set_xlabel("time [s]")
    a.set_ylabel("pz [m]")
    a.grid(alpha=0.3)
    a.legend(fontsize=7, ncol=3)

    # (c) 다리 길이
    a = ax[0, 2]
    for j in range(N_LEGS):
        a.plot(tt, L[mask, j], linewidth=0.9)
    a.axhline(L0, color="red", linestyle="--", linewidth=1.0)
    a.axhline(L0 + 2 * STROKE, color="red", linestyle="--", linewidth=1.0)
    a.set_title("(c) leg length L  (red = hardware limits)")
    a.set_xlabel("time [s]")
    a.set_ylabel("L [m]")
    a.grid(alpha=0.3)

    # (d) prismatic 변위
    a = ax[1, 0]
    for j in range(N_LEGS):
        a.plot(tt, d1[mask, j], linewidth=0.9)
    a.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    a.axhline(STROKE, color="red", linestyle="--", linewidth=1.0)
    a.set_ylim(-0.01, STROKE + 0.01)
    a.set_title("(d) prismatic displacement (each of 2)")
    a.set_xlabel("time [s]")
    a.set_ylabel("d [m]")
    a.grid(alpha=0.3)

    # (e) prismatic 속도
    a = ax[1, 1]
    dv = np.gradient(d1, DT, axis=0)
    for j in range(N_LEGS):
        a.plot(tt, dv[mask, j], linewidth=0.9)
    a.axhline(PRISMATIC_VEL_LIMIT, color="red", linestyle="--", linewidth=1.0)
    a.axhline(-PRISMATIC_VEL_LIMIT, color="red", linestyle="--", linewidth=1.0)
    a.set_title("(e) prismatic velocity  (red = 0.25 m/s limit)")
    a.set_xlabel("time [s]")
    a.set_ylabel("d_dot [m/s]")
    a.grid(alpha=0.3)

    # (f) hip pitch
    a = ax[1, 2]
    for j in range(N_LEGS):
        a.plot(tt, pitch[mask, j], linewidth=0.9)
    a.axhline(HIP_ACTION_SCALE, color="red", linestyle="--", linewidth=1.0)
    a.axhline(-HIP_ACTION_SCALE, color="red", linestyle="--", linewidth=1.0)
    a.set_title("(f) hip pitch  (red = action scale 0.3)")
    a.set_xlabel("time [s]")
    a.set_ylabel("q2 [rad]")
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("cpg_step3a_trajectory.png", dpi=130)
    return fig


# =====================================================================

def reset_transient(theta, d_step):
    """리셋 초기화 방식에 따른 과도구간 비교.

    Isaac 에서는 에피소드가 끝날 때마다 발진기를 다시 놓아야 한다.
    어떻게 놓느냐에 따라 첫 수십 ms 의 prismatic 속도가 완전히 달라진다.
    """
    print("=" * 74)
    print("리셋 초기화 방식 비교 - 과도구간의 prismatic 속도")
    print("=" * 74)
    print(f"{'초기화':<22}{'최대 d_vel':>12}{'한계 대비':>11}{'판정':>7}")

    cases = [
        ("극한 순환 + 목표 위상", dict(gait_vector=GAITS["tripod"])),
        ("무작위 (-1, 1)", dict(gait_vector=None, seed=0)),
        ("무작위 (-1, 1) seed 3", dict(gait_vector=None, seed=3)),
    ]
    for label, kw in cases:
        t, X, Y = run_cpg(theta, duration=1.0, **kw)
        p = map_to_foot(X, Y, d_step)
        _, _, d1, _, _, _ = inverse_kinematics(p)
        dv = np.abs(np.gradient(d1, DT, axis=0)).max()
        print(f"{label:<22}{dv:12.4f}{dv/PRISMATIC_VEL_LIMIT*100:10.1f}%"
              f"{'OK' if dv <= PRISMATIC_VEL_LIMIT else 'NG':>7}")
    print()
    print("  -> Isaac 의 reset(env_ids) 에서 반드시 극한 순환 위 목표 위상으로")
    print("     초기화할 것. 0 이나 무작위로 놓으면 매 리셋마다 발이 튄다.")
    print()


def speed_sweep():
    """커맨드 속도 범위 전체에서 한계를 넘는 지점이 있는지 훑는다."""
    print("=" * 74)
    print("속도 스윕 - 커맨드 범위 전체에서 한계 확인")
    print("=" * 74)
    print(f"{'v_cmd':>7}{'d_step':>9}{'L_min':>9}{'d_max':>8}"
          f"{'d_vel':>9}{'pitch':>9}{'판정':>8}")

    theta = phase_matrix(GAITS["tripod"])
    t, X, Y = run_cpg(theta, duration=6.0, gait_vector=GAITS["tripod"])
    tail = t > 3.0

    for v in [0.10, 0.20, 0.30, 0.45, 0.60, 0.80]:
        ds = d_step_from_speed(v)
        p = map_to_foot(X[tail], Y[tail], ds)
        roll, pitch, d1, d2, L, d_raw = inverse_kinematics(p)
        dv = np.abs(np.gradient(d1, DT, axis=0)).max()

        ok = (L.min() >= L0 and d_raw.max() <= STROKE
              and dv <= PRISMATIC_VEL_LIMIT
              and np.abs(pitch).max() <= HIP_ACTION_SCALE)
        print(f"{v:7.2f}{ds:9.4f}{L.min():9.4f}{d_raw.max():8.4f}"
              f"{dv:9.4f}{np.abs(pitch).max():9.4f}{'OK' if ok else 'NG':>8}")
    print()


def main():
    print("=" * 74)
    print("ver3.1 상수와 CPG 파라미터 (선택지 A)")
    print("=" * 74)
    f = OMEGA / (2 * np.pi)
    print(f"  L0 {L0} m,  stroke {STROKE} m,  prismatic vel limit "
          f"{PRISMATIC_VEL_LIMIT} m/s")
    print(f"  h {H_STANCE} m,  k1 {K1_LIFT} m,  k2 {K2_PUSH} m")
    print(f"  omega {OMEGA} rad/s -> {f:.3f} Hz,  alpha {ALPHA},  k {K_COUPLE}")
    print(f"  roll nominal +-{ROLL_NOMINAL} rad")
    print()
    print(f"  유각기 최상단 다리 길이 = h - k1 = {H_STANCE - K1_LIFT:.3f} m"
          f"   vs L0 {L0}  -> 여유 {(H_STANCE-K1_LIFT-L0)*1000:.0f} mm")
    print()

    v_cmd = 0.35
    d_step = d_step_from_speed(v_cmd)

    theta = phase_matrix(GAITS["tripod"])
    t, X, Y = run_cpg(theta, duration=6.0, gait_vector=GAITS["tripod"])

    p, roll, pitch, d1, L, ok = check(t, X, Y, d_step,
                                      f"tripod, 커맨드 속도 {v_cmd} m/s "
                                      f"(극한 순환 위에서 시작)")
    draw(t, p, roll, pitch, d1, L, d_step)
    reset_transient(theta, d_step)
    speed_sweep()

    print("=" * 74)
    print("저장됨: cpg_step3a_trajectory.png")
    print("=" * 74)
    plt.show()


if __name__ == "__main__":
    main()
