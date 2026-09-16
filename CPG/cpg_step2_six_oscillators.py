"""
CPG 2단계 - 발진기 6개를 묶어 걸음새 만들기

실행:  python cpg_step2_six_oscillators.py
필요:  numpy, matplotlib

1단계에서 발진기 하나가 원을 그리는 것을 확인했다.
이제 6개를 결합해서 다리 사이의 위상 관계, 즉 걸음새를 만든다.

실험 1. 결합이 없으면 제멋대로, 있으면 지정한 위상차로 수렴하는가
실험 2. 위상차 벡터가 tripod / ripple / wave와 대응하는가
실험 3. 걸음새를 도중에 바꿔도 점프가 없는가
실험 4. 결합 강도 k는 얼마가 적당한가
실험 5. 듀티(입각기 비율)를 바꿔 발자국 차트를 재현할 수 있는가
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 다리 번호와 걸음새 정의
# =====================================================================
# 다리 순서:  0=R1(우앞)  1=R2(우중)  2=R3(우뒤)
#            3=L1(좌앞)  4=L2(좌중)  5=L3(좌뒤)

LEG_NAMES = ["R1", "R2", "R3", "L1", "L2", "L3"]
N_LEGS = 6

# 걸음새 = 숫자 6개. 단위는 '주기' (0 = 기준, 0.5 = 반 주기 뒤).
GAITS = {
    "tripod": np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
    "ripple": np.array([0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6]),
    "wave":   np.array([2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6]),
}

# 걸음새별 듀티 = 입각기가 한 주기에서 차지하는 비율
DUTY = {"tripod": 0.5, "ripple": 2 / 3, "wave": 5 / 6}


def phase_matrix(gait_vector):
    """위상차 행렬을 만든다.  theta[i, j] = (i가 j보다 앞서야 하는 각도) [rad]

    숫자 6개에서 36칸이 전부 유도된다. 자유도는 여전히 6개다.
    """
    phi = 2.0 * np.pi * np.asarray(gait_vector, dtype=float)
    return phi[:, None] - phi[None, :]


# =====================================================================
# 결합된 Hopf 네트워크
# =====================================================================

def network_derivative(x, y, theta, mu, omega, alpha, k):
    """발진기 6개의 변화율을 한 번에 계산한다.

    개별 항 (1단계와 동일):
        반지름 항  alpha * (mu^2 - r^2) * (x, y)
        회전 항    (-omega*y, +omega*x)

    결합 항 (2단계에서 추가):
        j번 발진기의 점을 theta[i,j] 만큼 회전시킨 자리로 i를 끌어당긴다.
        회전 공식이 그대로 들어간다.
            x_rot = x*cos(t) - y*sin(t)
            y_rot = x*sin(t) + y*cos(t)
        sum_j C[i,j]*x[j] 는 행렬곱 (C @ x)[i] 와 같으므로
        루프 없이 한 줄로 쓴다. 자기 자신은 대각을 0으로 만들어 제외한다.
    """
    r_sq = x * x + y * y
    radial = alpha * (mu * mu - r_sq)

    x_dot = radial * x - omega * y
    y_dot = radial * y + omega * x

    if k != 0.0:
        C = np.cos(theta)
        S = np.sin(theta)
        np.fill_diagonal(C, 0.0)
        np.fill_diagonal(S, 0.0)
        x_dot += k * (C @ x - S @ y)
        y_dot += k * (S @ x + C @ y)

    return x_dot, y_dot


def integrate_network(x0, y0, theta, mu, omega, alpha, k, dt, duration,
                      theta_switch=None, t_switch=None):
    """오일러 적분. theta_switch를 주면 t_switch에서 걸음새를 바꾼다."""
    n_steps = int(duration / dt)
    t = np.zeros(n_steps)
    X = np.zeros((n_steps, N_LEGS))
    Y = np.zeros((n_steps, N_LEGS))
    X[0], Y[0] = np.array(x0, float), np.array(y0, float)

    switch_idx = int(t_switch / dt) if t_switch is not None else None
    theta_now = theta.copy()

    for i in range(n_steps - 1):
        if switch_idx is not None and i == switch_idx:
            theta_now = theta_switch.copy()

        x_dot, y_dot = network_derivative(X[i], Y[i], theta_now,
                                          mu, omega, alpha, k)
        X[i + 1] = X[i] + dt * x_dot
        Y[i + 1] = Y[i] + dt * y_dot
        t[i + 1] = t[i] + dt

    return t, X, Y


def relative_phase(X, Y, reference=0):
    """각 다리가 기준 다리보다 얼마나 앞서 있는지를 '주기' 단위로 반환한다.

    atan2(y, x)가 현재 위상이고, 기준 다리와의 차를 2*pi로 나눠
    0~1 범위로 접는다. 0.5면 반 주기 차이다.
    """
    angle = np.arctan2(Y, X)
    rel = (angle - angle[:, [reference]]) / (2.0 * np.pi)
    return np.mod(rel, 1.0)


def circular_error(measured, target):
    """주기 단위 위상차의 오차. 0.99와 0.01은 0.02 차이로 본다."""
    d = np.abs(measured - target)
    return np.minimum(d, 1.0 - d)


# =====================================================================
# 기본 파라미터
# =====================================================================

MU = 1.0
OMEGA = 10.5      # T = 0.598 s, 약 1.67 Hz
ALPHA = 50.0
K = 0.5           # 결합 강도 (2단계에서 새로 등장)
DT = 1.0 / 200.0

rng = np.random.default_rng(0)


def random_start():
    """무작위 초기 상태. 원 위가 아니라 아무 데서나 출발시킨다."""
    return rng.uniform(-1.0, 1.0, N_LEGS), rng.uniform(-1.0, 1.0, N_LEGS)


def print_setup():
    print("=" * 66)
    print("파라미터")
    print("=" * 66)
    print(f"  mu={MU}  omega={OMEGA}  alpha={ALPHA}  k={K}  dt={DT:.5f}")
    print(f"  주기 T = 2*pi/omega = {2*np.pi/OMEGA:.4f} s")
    print()
    print("  걸음새 정의 (단위: 주기)")
    header = "        " + "".join(f"{n:>8}" for n in LEG_NAMES)
    print(header)
    for name, vec in GAITS.items():
        row = "".join(f"{v:8.3f}" for v in vec)
        print(f"  {name:<6}{row}")
    print()


# =====================================================================
# 실험 1. 결합이 있어야 동기화된다
# =====================================================================

def experiment_1(ax_no, ax_yes):
    print("=" * 66)
    print("실험 1. 결합 유무 비교 (tripod 목표, 무작위 초기값)")
    print("=" * 66)

    theta = phase_matrix(GAITS["tripod"])
    x0, y0 = random_start()
    target = GAITS["tripod"] - GAITS["tripod"][0]

    for k, ax, label in [(0.0, ax_no, "k = 0 (결합 없음)"),
                         (K, ax_yes, f"k = {K} (결합 있음)")]:
        t, X, Y = integrate_network(x0, y0, theta, MU, OMEGA, ALPHA, k,
                                    DT, duration=4.0)
        rel = relative_phase(X, Y)
        err = circular_error(rel, target[None, :]).max(axis=1)

        for j in range(1, N_LEGS):
            ax.plot(t, rel[:, j], linewidth=0.9, label=LEG_NAMES[j])

        settled = np.where(err < 0.02)[0]
        t_lock = t[settled[0]] if len(settled) else float("nan")

        print(f"\n  {label}")
        print(f"    최종 오차(최대) : {err[-1]:.4f} 주기")
        print(f"    0.02 이내 진입   : {t_lock:.3f} s"
              if np.isfinite(t_lock) else "    0.02 이내 진입   : 없음")
        print("    최종 위상차 (목표 / 실측)")
        for j in range(1, N_LEGS):
            print(f"      {LEG_NAMES[j]}  {target[j]:6.3f} / {rel[-1, j]:6.3f}")

        for j in range(1, N_LEGS):
            ax.axhline(target[j], color="gray", linestyle=":", linewidth=0.6)
        ax.set_title(label.replace("결합 없음", "no coupling")
                          .replace("결합 있음", "with coupling"))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("phase lead over R1 [cycles]")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    print()


# =====================================================================
# 실험 2. 위상차 벡터가 곧 걸음새다
# =====================================================================

def experiment_2(axes):
    print("=" * 66)
    print("실험 2. 걸음새별 위상 수렴 확인")
    print("=" * 66)

    for ax, (name, vec) in zip(axes, GAITS.items()):
        theta = phase_matrix(vec)
        x0, y0 = random_start()
        t, X, Y = integrate_network(x0, y0, theta, MU, OMEGA, ALPHA, K,
                                    DT, duration=5.0)
        rel = relative_phase(X, Y)
        target = np.mod(vec - vec[0], 1.0)
        err = circular_error(rel[-1], target)

        print(f"\n  {name}  (최대 오차 {err.max():.4f} 주기)")
        # 0.999 와 0.001 은 같은 위상이므로 표시할 때 0 쪽으로 접는다
        shown = np.where(rel[-1] > 0.995, rel[-1] - 1.0, rel[-1])
        print("       " + "".join(f"{n:>8}" for n in LEG_NAMES))
        print("  목표 " + "".join(f"{v:8.3f}" for v in target))
        print("  실측 " + "".join(f"{v:8.3f}" for v in shown))

        # 마지막 두 주기만 그린다 (수렴 후 모습)
        period = 2.0 * np.pi / OMEGA
        mask = t > t[-1] - 2.0 * period
        for j in range(N_LEGS):
            ax.plot(t[mask], X[mask, j] + j * 2.5, linewidth=1.0)
            ax.text(t[mask][0], j * 2.5 + 0.9, LEG_NAMES[j], fontsize=7)

        ax.set_title(f"{name}: x of each oscillator")
        ax.set_xlabel("time [s]")
        ax.set_yticks([])
        ax.grid(alpha=0.3, axis="x")
    print()


# =====================================================================
# 실험 3. 걸음새 전환에 점프가 없다
# =====================================================================

def experiment_3(ax):
    print("=" * 66)
    print("실험 3. tripod -> ripple 전환")
    print("=" * 66)

    t_switch, duration = 3.0, 8.0
    theta_a = phase_matrix(GAITS["tripod"])
    theta_b = phase_matrix(GAITS["ripple"])

    # 주의: 6개를 모두 똑같은 값 (mu, 0) 에서 출발시키면 안 된다.
    # 완전 동상(in-phase) 상태는 결합 항이 대칭이라 힘이 0인 평형점이고,
    # 수치적으로 정확히 그 위에 있으면 영원히 빠져나오지 못한다.
    x0, y0 = random_start()
    t, X, Y = integrate_network(x0, y0,
                                theta_a, MU, OMEGA, ALPHA, K, DT,
                                duration=duration,
                                theta_switch=theta_b, t_switch=t_switch)

    idx = int(t_switch / DT)
    jump = np.abs(X[idx + 1] - X[idx]).max()
    typical = np.abs(np.diff(X[idx - 40:idx], axis=0)).max()

    target_b = np.mod(GAITS["ripple"] - GAITS["ripple"][0], 1.0)
    rel = relative_phase(X, Y)
    err = circular_error(rel, target_b[None, :]).max(axis=1)
    after = np.where(err[idx:] < 0.02)[0]
    t_relock = t[idx + after[0]] - t_switch if len(after) else float("nan")

    print(f"  전환 시각 : {t_switch} s")
    print(f"  전환 순간 x 최대 변화량 : {jump:.5f}")
    print(f"  평소 한 스텝 최대 변화량 : {typical:.5f}")
    print(f"  -> 비율 {jump/typical:.2f}배  (1에 가까우면 점프 없음)")
    print(f"  새 걸음새로 재수렴 : {t_relock:.3f} s")
    print()

    for j in range(N_LEGS):
        ax.plot(t, X[:, j] + j * 2.5, linewidth=0.8)
        ax.text(t[0], j * 2.5 + 0.9, LEG_NAMES[j], fontsize=7)
    ax.axvline(t_switch, color="red", linestyle=":", linewidth=1.2)
    ax.set_title("experiment 3: tripod -> ripple at the red line")
    ax.set_xlabel("time [s]")
    ax.set_yticks([])
    ax.grid(alpha=0.3, axis="x")


# =====================================================================
# 실험 4. 결합 강도 k
# =====================================================================

def experiment_4(ax):
    print("=" * 66)
    print("실험 4. 결합 강도 k 스윕")
    print("=" * 66)
    print(f"{'k':>7}  {'수렴시각[s]':>12}  {'최종오차':>10}  {'진폭 r 범위':>18}")

    theta = phase_matrix(GAITS["tripod"])
    target = GAITS["tripod"] - GAITS["tripod"][0]
    x0, y0 = random_start()

    for k in [0.0, 0.1, 0.5, 2.0, 8.0, 20.0]:
        with np.errstate(over="ignore", invalid="ignore"):
            t, X, Y = integrate_network(x0, y0, theta, MU, OMEGA, ALPHA, k,
                                        DT, duration=6.0)
        rel = relative_phase(X, Y)
        err = circular_error(rel, target[None, :]).max(axis=1)

        settled = np.where(err < 0.02)[0]
        t_lock = t[settled[0]] if len(settled) else float("nan")

        r = np.sqrt(X ** 2 + Y ** 2)[-200:]
        band = ("발산" if not np.all(np.isfinite(r))
                else f"[{r.min():.3f}, {r.max():.3f}]")

        print(f"{k:7.1f}  {t_lock:12.3f}  {err[-1]:10.4f}  {band:>18}")

        if np.all(np.isfinite(err)):
            ax.plot(t, err, linewidth=1.0, label=f"k={k:g}")

    ax.axhline(0.02, color="k", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_title("experiment 4: phase error vs time for different k")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("max phase error [cycles]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    print()
    print("  k가 작으면 수렴이 느리고, 지나치게 크면 위상이 흔들리거나 발산한다.")
    print("  1단계의 alpha와 같은 종류의 트레이드오프다.")
    print()
    print("  주목할 부작용: k가 커질수록 진폭 r 자체가 mu(=1)보다 커진다.")
    print("  결합 항이 에너지를 추가로 밀어넣기 때문이다. 3단계에서 발끝")
    print("  위치로 매핑할 때 보폭이 의도보다 커지므로, r을 정규화하거나")
    print("  k를 작게 유지해야 한다.")
    print()


# =====================================================================
# 실험 5. 듀티 제어 - 발자국 차트 재현
# =====================================================================

def omega_switched(y, omega_st, omega_sw, b=10.0):
    """y > 0 (유각기) 이면 omega_sw, y < 0 (입각기) 이면 omega_st.

    시그모이드 1/(e^(-b*y)+1) 는 y>0에서 1, y<0에서 0 을 뱉는다.
    계단함수 대신 쓰는 이유는 전환 지점에서 궤적이 꺾이지 않게 하려고.
    """
    s = 1.0 / (np.exp(-b * y) + 1.0)      # y>0 -> 1
    return omega_sw * s + omega_st * (1.0 - s)


def integrate_with_duty(theta, duty, period, mu, alpha, k, dt, duration):
    """입각기가 period*duty 초, 유각기가 period*(1-duty) 초 걸리도록 만든다.

    반 바퀴 도는 데 걸리는 시간이 pi/omega 이므로
        omega_st = pi / (duty * period)
        omega_sw = pi / ((1-duty) * period)
    duty=0.5 이면 둘이 같아져 1단계와 동일해진다.
    """
    omega_st = np.pi / (duty * period)
    omega_sw = np.pi / ((1.0 - duty) * period)

    n_steps = int(duration / dt)
    t = np.zeros(n_steps)
    X = np.zeros((n_steps, N_LEGS))
    Y = np.zeros((n_steps, N_LEGS))
    X[0] = mu * np.cos(2 * np.pi * np.arange(N_LEGS) * 0.0)
    Y[0] = 0.0
    X[0], Y[0] = rng.uniform(-1, 1, N_LEGS), rng.uniform(-1, 1, N_LEGS)

    for i in range(n_steps - 1):
        omega = omega_switched(Y[i], omega_st, omega_sw)
        x_dot, y_dot = network_derivative(X[i], Y[i], theta,
                                         mu, omega, alpha, k)
        X[i + 1] = X[i] + dt * x_dot
        Y[i + 1] = Y[i] + dt * y_dot
        t[i + 1] = t[i] + dt

    return t, X, Y, omega_st, omega_sw


def experiment_5(axes):
    print("=" * 66)
    print("실험 5. 듀티 제어와 발자국 차트")
    print("=" * 66)
    print("  1단계 식만 쓰면 위/아래 반 바퀴가 같은 속도라 듀티가 0.5로")
    print("  고정된다. omega를 y 부호에 따라 바꿔야 ripple/wave가 나온다.")
    print()
    print(f"{'걸음새':>8}  {'목표듀티':>8}  {'실측듀티':>8}  {'동시 유각 다리수':>16}")

    period = 2.0 * np.pi / OMEGA

    for ax, (name, vec) in zip(axes, GAITS.items()):
        theta = phase_matrix(vec)
        duty = DUTY[name]
        t, X, Y, w_st, w_sw = integrate_with_duty(
            theta, duty, period, MU, ALPHA, K, DT, duration=8.0)

        # 뒤쪽 3주기만 사용 (수렴 후)
        mask = t > t[-1] - 3.0 * period
        tt, YY = t[mask], Y[mask]
        swing = YY > 0.0

        measured_duty = 1.0 - swing.mean()
        n_swing = swing.sum(axis=1)
        print(f"{name:>8}  {duty:8.3f}  {measured_duty:8.3f}"
              f"  {n_swing.mean():16.2f}")

        for j in range(N_LEGS):
            on = swing[:, j]
            edges = np.diff(on.astype(int))
            starts = list(np.where(edges == 1)[0] + 1)
            ends = list(np.where(edges == -1)[0] + 1)
            if on[0]:
                starts = [0] + starts
            if on[-1]:
                ends = ends + [len(on) - 1]
            bars = [(tt[s], tt[e] - tt[s]) for s, e in zip(starts, ends)]
            ax.broken_barh(bars, (j - 0.4, 0.8))

        ax.set_yticks(range(N_LEGS))
        ax.set_yticklabels(LEG_NAMES)
        ax.invert_yaxis()
        ax.set_xlabel("time [s]")
        ax.set_title(f"{name}  (duty {duty:.2f}, bars = swing)")
        ax.grid(alpha=0.3, axis="x")
    print()
    print("  동시 유각 다리수가 tripod 3, ripple 2, wave 1 에 가까우면")
    print("  앞에서 본 발자국 차트가 코드로 재현된 것이다.")
    print()


# =====================================================================

def main():
    print_setup()

    fig1, ax1 = plt.subplots(1, 2, figsize=(13, 4.5))
    experiment_1(ax1[0], ax1[1])
    fig1.tight_layout()
    fig1.savefig("cpg_step2_coupling.png", dpi=130)

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
    experiment_2(ax2)
    fig2.tight_layout()
    fig2.savefig("cpg_step2_gaits.png", dpi=130)

    fig3, ax3 = plt.subplots(1, 2, figsize=(14, 4.5))
    experiment_3(ax3[0])
    experiment_4(ax3[1])
    fig3.tight_layout()
    fig3.savefig("cpg_step2_switch_and_k.png", dpi=130)

    fig4, ax4 = plt.subplots(1, 3, figsize=(15, 3.8))
    experiment_5(ax4)
    fig4.tight_layout()
    fig4.savefig("cpg_step2_footfall.png", dpi=130)

    print("=" * 66)
    print("저장됨:")
    print("  cpg_step2_coupling.png       결합 유무 비교")
    print("  cpg_step2_gaits.png          걸음새별 6개 신호")
    print("  cpg_step2_switch_and_k.png   걸음새 전환, k 스윕")
    print("  cpg_step2_footfall.png       발자국 차트")
    print("=" * 66)

    plt.show()


if __name__ == "__main__":
    main()
