"""
CPG 1단계 - Hopf 발진기 하나 이해하기

실행:  python cpg_step1_single_hopf.py
필요:  numpy, matplotlib

실험 1. 어디서 출발하든 같은 원으로 수렴하는가 (극한 순환)
실험 2. omega가 주기를 결정하는가 (T = 2*pi/omega)
실험 3. alpha가 수렴 속도를 결정하는가 + 적분 안정성 한계
실험 4. 외란을 받아도 스스로 복구하는가
실험 5. 파라미터를 도중에 바꿔도 점프가 없는가 (vs sin 함수)
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 핵심 함수 - 이 두 개가 CPG의 전부다
# =====================================================================

def hopf_derivative(x, y, mu, omega, alpha):
    """Hopf 발진기의 변화율을 계산한다.

    반지름 항:  alpha * (mu^2 - r^2) * (x, y)
        r < mu 이면 양수  -> 바깥으로 민다
        r > mu 이면 음수  -> 안으로 당긴다
        r = mu 이면 0     -> 아무것도 안 한다

    회전 항:   (-omega*y, +omega*x)
        항상 진행 방향과 직각으로 밀어서 원을 그리게 한다
        부호가 비대칭인 것이 '반시계 방향'이라는 정보다
    """
    r_sq = x * x + y * y
    radial = alpha * (mu * mu - r_sq)

    x_dot = radial * x - omega * y
    y_dot = radial * y + omega * x
    return x_dot, y_dot


def integrate(x0, y0, mu, omega, alpha, dt, duration):
    """오일러 적분. 미분방정식을 푸는 게 아니라 그냥 한 스텝씩 더해나간다.

        x <- x + dt * x_dot

    이 한 줄이 전부다. 적분 간격 dt는 계산의 정밀도를 정할 뿐,
    리듬의 주기와는 무관하다. 주기는 omega 혼자 결정한다.
    """
    n_steps = int(duration / dt)
    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    x[0], y[0] = x0, y0

    for i in range(n_steps - 1):
        x_dot, y_dot = hopf_derivative(x[i], y[i], mu, omega, alpha)
        x[i + 1] = x[i] + dt * x_dot
        y[i + 1] = y[i] + dt * y_dot
        t[i + 1] = t[i] + dt

    return t, x, y


def measure_period(t, x):
    """x가 음수에서 양수로 바뀌는 순간들의 간격으로 주기를 측정한다."""
    crossings = []
    for i in range(len(x) - 1):
        if x[i] <= 0.0 < x[i + 1]:
            # 선형 보간으로 정확한 교차 시각을 찾는다
            frac = -x[i] / (x[i + 1] - x[i])
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))

    if len(crossings) < 3:
        return None
    # 앞쪽은 아직 수렴 중이므로 뒤쪽 교차만 사용
    return float(np.mean(np.diff(crossings[1:])))


# =====================================================================
# 기본 파라미터 - 우리 로봇 기준 출발값
# =====================================================================

MU = 1.0          # 원의 반지름 (진폭)
OMEGA = 10.5      # 각속도 [rad/s]  ->  T = 2*pi/10.5 = 0.60 s, 약 1.67 Hz
ALPHA = 50.0      # 수렴율
DT = 1.0 / 200.0  # Isaac Lab의 sim.dt와 동일


def print_setup():
    period = 2.0 * np.pi / OMEGA
    margin = 2.0 * ALPHA * MU * MU * DT

    print("=" * 62)
    print("파라미터")
    print("=" * 62)
    print(f"  mu    = {MU}        원의 반지름")
    print(f"  omega = {OMEGA}       각속도 [rad/s]")
    print(f"  alpha = {ALPHA}      수렴율")
    print(f"  dt    = {DT:.5f}   적분 간격 [s]")
    print()
    print(f"  이론 주기 T = 2*pi/omega = {period:.4f} s  ({1.0/period:.2f} Hz)")
    print()
    print("적분 안정성 조건:  2 * alpha * mu^2 * dt < 1")
    print(f"  2 * {ALPHA} * {MU}^2 * {DT:.5f} = {margin:.3f}", end="  ")
    print("-> 안전" if margin < 1.0 else "-> 위험! 발산할 수 있음")
    print()


# =====================================================================
# 실험 1. 극한 순환 - 어디서 출발하든 같은 원으로
# =====================================================================

def experiment_1(ax_phase, ax_radius):
    starts = [
        (0.10, 0.00),   # 원보다 한참 안쪽
        (0.30, 0.30),
        (-0.20, 0.15),
        (1.80, 0.00),   # 원보다 한참 바깥
        (-1.50, 1.20),
        (0.02, -0.02),  # 원점 근처
    ]

    print("=" * 62)
    print("실험 1. 극한 순환")
    print("=" * 62)
    print(f"{'시작점':>18}  {'시작 반지름':>10}  {'최종 반지름':>10}")

    for x0, y0 in starts:
        t, x, y = integrate(x0, y0, MU, OMEGA, ALPHA, DT, duration=2.0)
        r = np.sqrt(x * x + y * y)

        ax_phase.plot(x, y, linewidth=0.8, alpha=0.75)
        ax_phase.plot(x0, y0, "o", markersize=4, color="black", zorder=5)
        ax_radius.plot(t, r, linewidth=0.9, alpha=0.75)

        print(f"  ({x0:5.2f}, {y0:5.2f})  {np.hypot(x0, y0):10.3f}  {r[-1]:10.3f}")

    # 목표 원
    theta = np.linspace(0, 2 * np.pi, 400)
    ax_phase.plot(MU * np.cos(theta), MU * np.sin(theta),
                  "k--", linewidth=1.2, label=f"target r = {MU}")

    ax_phase.set_title("experiment 1: all trajectories converge to one circle")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_aspect("equal")
    ax_phase.grid(alpha=0.3)
    ax_phase.legend(fontsize=8)

    ax_radius.axhline(MU, color="k", linestyle="--", linewidth=1.2)
    ax_radius.set_title("radius vs time (black dashed = mu)")
    ax_radius.set_xlabel("time [s]")
    ax_radius.set_ylabel("r = sqrt(x^2 + y^2)")
    ax_radius.grid(alpha=0.3)
    print()


# =====================================================================
# 실험 2. omega가 주기를 결정한다
# =====================================================================

def experiment_2(ax):
    print("=" * 62)
    print("실험 2. omega와 주기")
    print("=" * 62)
    print(f"{'omega':>8}  {'이론 T':>10}  {'측정 T':>10}  {'오차':>8}")

    for omega in [5.0, 10.5, 20.0]:
        t, x, y = integrate(1.0, 0.0, MU, omega, ALPHA, DT, duration=6.0)
        measured = measure_period(t, x)
        theory = 2.0 * np.pi / omega

        ax.plot(t, x, linewidth=1.0,
                label=f"omega={omega}  (T={theory:.3f}s)")

        if measured is None:
            print(f"{omega:8.1f}  {theory:10.4f}  {'측정불가':>10}")
        else:
            error = abs(measured - theory) / theory * 100.0
            print(f"{omega:8.1f}  {theory:10.4f}  {measured:10.4f}  {error:7.2f}%")

    ax.set_title("experiment 2: omega sets the period, nothing else does")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    print()


# =====================================================================
# 실험 3. alpha가 수렴 속도를 결정한다
# =====================================================================

def experiment_3(ax):
    print("=" * 62)
    print("실험 3. alpha와 수렴 속도, 그리고 적분 한계")
    print("=" * 62)
    print("  같은 시작점(0.1, 0). r이 0.95에 도달하는 시각과,")
    print("  충분히 시간이 지난 뒤 r이 흔들리는 폭을 함께 본다.")
    print()
    print(f"{'alpha':>7}  {'2*a*mu^2*dt':>12}  {'도달시각':>9}  {'정상상태 r 범위':>18}")

    for alpha in [5.0, 20.0, 50.0, 150.0, 300.0, 420.0]:
        with np.errstate(over="ignore", invalid="ignore"):
            t, x, y = integrate(0.1, 0.0, MU, OMEGA, alpha, DT, duration=3.0)
        r = np.sqrt(x * x + y * y)

        reached = np.where(r > 0.95 * MU)[0]
        t_reach = t[reached[0]] if len(reached) > 0 else float("nan")
        margin = 2.0 * alpha * MU * MU * DT

        tail = r[-400:]
        if not np.all(np.isfinite(tail)):
            band = "발산 (inf/nan)"
        else:
            band = f"[{np.min(tail):.3f}, {np.max(tail):.3f}]"

        ax.plot(t, r, linewidth=1.0, label=f"alpha={alpha:g}")

        print(f"{alpha:7.0f}  {margin:12.2f}  {t_reach:9.4f}  {band:>18}")

    ax.axhline(MU, color="k", linestyle="--", linewidth=1.0)
    ax.set_title("experiment 3: alpha sets how fast r reaches mu")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("r")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1e3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    print()
    print("  2*alpha*mu^2*dt < 1 은 여유를 둔 실무 기준이지,")
    print("  이 값을 넘는 순간 발산하는 경계선이 아니다.")
    print("  dt=1/200 에서 실제로는 alpha~150 부터 r에 리플이 보이고")
    print("  alpha~400 부근에서 발산한다. 기준값을 지키면 리플이 없다.")
    print()


# =====================================================================
# 실험 4. 외란 복구 - sin 함수는 못 하는 일
# =====================================================================

def experiment_4(ax_time, ax_phase):
    print("=" * 62)
    print("실험 4. 외란 복구")
    print("=" * 62)

    duration, t_kick = 2.5, 1.0
    n_steps = int(duration / DT)
    kick_idx = int(t_kick / DT)

    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = MU, 0.0

    for i in range(n_steps - 1):
        x_dot, y_dot = hopf_derivative(x[i], y[i], MU, OMEGA, ALPHA)
        x[i + 1] = x[i] + DT * x_dot
        y[i + 1] = y[i] + DT * y_dot
        t[i + 1] = t[i] + DT

        # 발이 돌부리에 걸린 상황을 흉내낸다
        if i == kick_idx:
            x[i + 1] += 1.1
            y[i + 1] -= 0.7

    r = np.sqrt(x * x + y * y)
    r_after = r[kick_idx + 1]

    start = kick_idx + 1
    recovered = np.where(np.abs(r[start:] - MU) < 0.02 * MU)[0]
    t_recover = t[start + recovered[0]] - t[start] if len(recovered) else float("nan")

    print(f"  외란 직전 반지름 : {r[kick_idx]:.3f}")
    print(f"  외란 직후 반지름 : {r_after:.3f}")
    print(f"  원으로 복귀 소요 : {t_recover:.4f} s")
    print()
    print("  sin(omega*t) 방식이면 외부에서 값을 밀어넣을 자리 자체가 없다.")
    print("  Hopf는 x, y가 변수이므로 밀 수 있고, 반지름 항이 되돌린다.")
    print()

    ax_time.plot(t, x, linewidth=1.0, label="x")
    ax_time.plot(t, r, linewidth=1.0, label="r")
    ax_time.axvline(t_kick, color="red", linestyle=":", linewidth=1.2)
    ax_time.axhline(MU, color="k", linestyle="--", linewidth=0.8)
    ax_time.set_title("experiment 4: recovery after a kick (red line)")
    ax_time.set_xlabel("time [s]")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)

    ax_phase.plot(x[:kick_idx], y[:kick_idx], linewidth=1.0, label="before")
    ax_phase.plot(x[kick_idx:], y[kick_idx:], linewidth=1.0, label="after")
    theta = np.linspace(0, 2 * np.pi, 400)
    ax_phase.plot(MU * np.cos(theta), MU * np.sin(theta),
                  "k--", linewidth=1.0)
    ax_phase.set_title("same kick, phase plane")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_aspect("equal")
    ax_phase.grid(alpha=0.3)
    ax_phase.legend(fontsize=8)


# =====================================================================
# 실험 5. 파라미터를 도중에 바꿔도 점프가 없다
# =====================================================================

def experiment_5(ax):
    print("=" * 62)
    print("실험 5. 주파수를 도중에 바꾸면")
    print("=" * 62)

    duration, t_switch = 3.4, 1.7
    omega_before, omega_after = 6.28, 12.56

    n_steps = int(duration / DT)
    switch_idx = int(t_switch / DT)

    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = MU, 0.0

    for i in range(n_steps - 1):
        omega = omega_before if i < switch_idx else omega_after
        x_dot, y_dot = hopf_derivative(x[i], y[i], MU, omega, ALPHA)
        x[i + 1] = x[i] + DT * x_dot
        y[i + 1] = y[i] + DT * y_dot
        t[i + 1] = t[i] + DT

    # 비교 대상: 시계만 보는 방식
    omega_naive = np.where(t < t_switch, omega_before, omega_after)
    x_naive = MU * np.cos(omega_naive * t)

    jump_cpg = abs(x[switch_idx + 1] - x[switch_idx])
    jump_naive = abs(x_naive[switch_idx + 1] - x_naive[switch_idx])

    print(f"  t = {t_switch} s 에서 omega를 {omega_before} -> {omega_after} 로 바꿈")
    print()
    print(f"  Hopf 발진기   전환 순간 x 변화량 : {jump_cpg:.5f}")
    print(f"  cos(omega*t)  전환 순간 x 변화량 : {jump_naive:.5f}")
    print()
    print("  cos 방식은 omega가 t와 곱해져 있으므로 omega를 바꾸는 순간")
    print("  값 자체가 튄다. Hopf는 omega가 변화율 계산에만 쓰이고")
    print("  x, y는 누적된 상태이므로 연속이다.")
    print()

    ax.plot(t, x, linewidth=1.2, label="Hopf oscillator")
    ax.plot(t, x_naive, linewidth=1.0, alpha=0.8, label="cos(omega*t)")
    ax.axvline(t_switch, color="red", linestyle=":", linewidth=1.2)
    ax.set_title("experiment 5: changing omega mid-run (red line)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# =====================================================================

def main():
    print_setup()

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    experiment_1(axes[0, 0], axes[0, 1])
    experiment_2(axes[1, 0])
    experiment_3(axes[1, 1])
    experiment_4(axes[2, 0], axes[2, 1])

    fig.tight_layout()
    fig.savefig("cpg_step1_experiments.png", dpi=130)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    experiment_5(ax2)
    fig2.tight_layout()
    fig2.savefig("cpg_step1_omega_switch.png", dpi=130)

    print("=" * 62)
    print("저장됨: cpg_step1_experiments.png, cpg_step1_omega_switch.png")
    print("=" * 62)

    plt.show()


if __name__ == "__main__":
    main()
