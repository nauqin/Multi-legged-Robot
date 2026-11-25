"""CPG + 수치 IK 오프라인 검증 (Isaac 불필요, CPU).

    python scripts/check_cpg_ik.py [--urdf "../usd files/hugo_hexapod_ver3_1.urdf"] [--out cpg_ik_check]

검증 항목
    1. FK 가 URDF 와 일치하는가 (URDF 를 numpy 로 독립 파싱해 대조)
    2. 기준 자세: 무릎 굽힘, prismatic 행정 중앙, 몸통 높이
    3. 시나리오 (정지 -> wave -> ripple -> tripod -> 제자리 회전 -> 곡선 -> 옆걸음 -> 후진 -> 정지)
       - IK 잔여오차, 관절 속도 / 한계 사용률, 무릎이 펴지는지
       - gait 전환 중 동시에 뜬 다리 수 (정적 안정성: 3개 이상 지지)
       - 입각 다리들의 발끝 속도가 몸통 속도와 같은지 (미끄러짐 없는 궤적)
    4. 그림: 관절각, 위상/gait, 발끝 궤적, 지지 다리 수
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "source", "Multi_Legged_Robot", "Multi_Legged_Robot", "tasks",
                    "manager_based", "multi_legged_robot", "cpg_core.py")
spec = importlib.util.spec_from_file_location("cpg_core", CORE)
core = importlib.util.module_from_spec(spec)
import sys  # noqa: E402
sys.modules["cpg_core"] = core
spec.loader.exec_module(core)


# ---------------------------------------------------------------------------
def urdf_fk(urdf_path, leg, q5):
    """URDF 를 직접 따라가며 발 링크 원점 위치를 계산 (rpy=0 가정 확인 포함)."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    names = {f"{leg}_joint1_roll": q5[0], f"{leg}_joint2_pitch": q5[1], f"{leg}_prismatic1": q5[2],
             f"{leg}_joint3_pitch": q5[3], f"{leg}_prismatic2": q5[4]}
    chain = []
    link = f"{leg}_feet"
    while link in joints:
        j = joints[link]
        chain.append(j)
        link = j.find("parent").get("link")
    T = np.eye(4)
    for j in reversed(chain):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
        rpy = o.get("rpy") if o is not None else "0 0 0"
        assert all(abs(float(v)) < 1e-9 for v in rpy.split()), "rpy != 0 은 미지원"
        To = np.eye(4)
        To[:3, 3] = xyz
        T = T @ To
        if j.get("type") in ("revolute", "prismatic"):
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            qv = names[j.get("name")]
            M = np.eye(4)
            if j.get("type") == "revolute":
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                M[:3, :3] = np.eye(3) + math.sin(qv) * K + (1 - math.cos(qv)) * K @ K
            else:
                M[:3, 3] = ax * qv
            T = T @ M
    return T[:3, 3]


# ---------------------------------------------------------------------------
SCENARIO = [
    # (시작 t, vx, vy, wz, 설명)
    (0.0, 0.00, 0.00, 0.00, "stand"),
    (2.0, 0.08, 0.00, 0.00, "slow fwd"),
    (10.0, 0.16, 0.00, 0.00, "mid fwd"),
    (18.0, 0.30, 0.00, 0.00, "fast fwd"),
    (26.0, 0.00, 0.00, 0.30, "turn in place"),
    (32.0, 0.18, 0.00, 0.20, "arc"),
    (40.0, 0.00, 0.12, 0.00, "sideways"),
    (46.0, -0.15, 0.00, 0.00, "backward"),
    (54.0, 0.00, 0.00, 0.00, "stop"),
    (58.0, None, None, None, "end"),
]


def scenario_cmd(t):
    for k in range(len(SCENARIO) - 1):
        if SCENARIO[k][0] <= t < SCENARIO[k + 1][0]:
            return SCENARIO[k][1:4]
    return (0.0, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(HERE, "..", "..", "usd files", "hugo_hexapod_ver3_1.urdf"))
    ap.add_argument("--out", default="cpg_ik_check")
    ap.add_argument("--dt", type=float, default=1.0 / 200.0)
    ap.add_argument("--rough", type=float, default=0.0, help="일정한 거칠기 입력 (gait 규칙 확인용)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    posture = core.HugoPostureCfg()
    chain, ik, nominal, q_nom = core.build_hugo(posture, dtype=torch.float64)
    legs = core.HUGO_LEG_PREFIXES

    # ---- 1. FK vs URDF ----------------------------------------------------
    if os.path.exists(args.urdf):
        worst = 0.0
        for _ in range(200):
            q = (torch.rand(1, 6, 5, dtype=torch.float64) * 2 - 1) * 1.2
            q[..., 2] = q[..., 2].abs() * 0.12
            q[..., 4] = q[..., 4].abs() * 0.12
            tip, _ = chain.fk(q, with_jacobian=False)
            for li, leg in enumerate(legs):
                ref = urdf_fk(args.urdf, leg, q[0, li].numpy())
                worst = max(worst, float(np.abs(ref - tip[0, li].numpy()).max()))
        print(f"[1] FK vs URDF  최대 오차 {worst:.2e} m  ({'OK' if worst < 1e-9 else 'FAIL'})")
        # Jacobian 수치 미분 대조
        q = q_nom.unsqueeze(0).clone()
        _, J = chain.fk(q)
        eps = 1e-6
        Jn = torch.zeros_like(J)
        for k in range(5):
            dq = torch.zeros_like(q)
            dq[..., k] = eps
            tp, _ = chain.fk(q + dq, False)
            tm, _ = chain.fk(q - dq, False)
            Jn[..., k] = (tp - tm) / (2 * eps)
        print(f"    Jacobian vs 수치미분 최대 오차 {float((J - Jn).abs().max()):.2e}")
    else:
        print(f"[1] URDF 없음, 건너뜀: {args.urdf}")

    # ---- 2. 기준 자세 -------------------------------------------------------
    print("[2] 기준 자세 (roll, hip, pr1, knee, pr2)")
    for li, leg in enumerate(legs):
        print(f"    {leg}: " + "  ".join(f"{v:+.3f}" for v in q_nom[li].tolist()))
    print(f"    hip -> 발 링크 수직 {posture.stance_height:.3f} m, 지면 기준 hip 높이 "
          f"{posture.stance_height + 0.02:.3f} m, 무릎 굽힘 {abs(q_nom[0, 3].item()):.3f} rad "
          f"({math.degrees(abs(q_nom[0, 3].item())):.1f} deg)")

    # ---- 3. 시나리오 --------------------------------------------------------
    gp = core.GaitParams()
    caps = core.gait_speed_caps(gp, dtype=torch.float64)
    print("[3] gait 최대 속도: " + ", ".join(f"{g} {c:.3f}" for g, c in zip(core.GAIT_NAMES, caps.tolist())))
    cpg = core.GaitCPG(1, nominal, gp, dtype=torch.float64, initial_gait="wave")
    sel = core.GaitSelector(1, core.SelectorParams(), caps, dtype=torch.float64)
    q = q_nom.unsqueeze(0).clone()
    T = SCENARIO[-1][0]
    n = int(T / args.dt)
    log = {k: [] for k in ("t", "q", "err", "gait", "blend", "duty", "swing", "tgt", "cmd", "r")}
    first = torch.tensor([True])
    q_prev = q.clone()
    vel_max = torch.zeros(5, dtype=torch.float64)
    slip_err = []
    for i in range(n):
        t = i * args.dt
        cmd = torch.tensor([scenario_cmd(t)], dtype=torch.float64)
        demand = cpg.leg_demand(cpg.cmd_f)
        speed = torch.norm(demand, dim=-1).amax(dim=1)
        g, imm = sel.step(args.dt, speed, torch.tensor([args.rough], dtype=torch.float64), cpg.gait_cur, first)
        first = torch.tensor([False])
        cpg.request_gait(g, imm)
        tgt = cpg.step(args.dt, cmd)
        q, err = ik.solve(tgt, q, iters=2)
        qd = (q - q_prev) / args.dt
        vel_max = torch.max(vel_max, qd.abs().amax(dim=(0, 1)))
        q_prev = q.clone()
        _, duty, _ = cpg.current()
        log["t"].append(t)
        log["q"].append(q[0].numpy().copy())
        log["err"].append(err[0].numpy().copy())
        log["gait"].append(int(cpg.gait_cur[0]))
        log["blend"].append(float(cpg.blend[0]))
        log["duty"].append(float(duty[0]))
        log["swing"].append(cpg._swing[0].numpy().copy())
        log["tgt"].append(tgt[0].numpy().copy())
        log["cmd"].append(cmd[0].numpy().copy())
        log["r"].append(float(cpg.r[0, 0]))
    for k in log:
        log[k] = np.array(log[k])

    # 입각 다리 발끝 속도 vs 요구 속도 (r = 1, 전환 아님 구간)
    tg = log["tgt"]
    vfoot = np.gradient(tg[:, :, :2], args.dt, axis=0)
    st = ~log["swing"].astype(bool)
    st_inner = st & np.roll(st, 1, axis=0) & np.roll(st, -1, axis=0)
    dem = []
    for i in range(n):
        c = torch.tensor(log["cmd"][i:i + 1])
        dem.append(cpg.leg_demand(c)[0].numpy())
    dem = np.array(dem)

    steady = (log["r"] > 0.999) & (log["blend"] >= 1.0)
    # 명령 필터가 수렴한 구간만 (명령 변경 후 1.5 s 제외)
    tchg = np.array([s[0] for s in SCENARIO])
    since = np.array([t - tchg[tchg <= t].max() for t in log["t"]])
    steady &= since > 1.5
    m = st_inner & steady[:, None]
    slip = np.linalg.norm(vfoot + dem, axis=-1)[m]

    jn = ("roll", "hip", "pr1", "knee", "pr2")
    vlim = np.array([5.0, 5.0, 0.25, 5.0, 0.25])
    print(f"    IK 잔여오차  최대 {log['err'].max() * 1000:.2f} mm, 95% {np.percentile(log['err'], 95) * 1000:.2f} mm")
    print("    관절 속도 최대 / 한계: " + ", ".join(
        f"{a} {v:.3f}/{lim:g} ({100 * v / lim:.0f}%)" for a, v, lim in zip(jn, vel_max.tolist(), vlim)))
    knee = log["q"][:, :, 3]
    print(f"    무릎 범위 {knee.min():+.3f} ~ {knee.max():+.3f} rad (0 에 가까울수록 펴짐)")
    pr = log["q"][:, :, [2, 4]]
    print(f"    prismatic 범위 {pr.min():.4f} ~ {pr.max():.4f} m")
    nsw = log["swing"].sum(axis=1) * (log["r"] > 0.05)
    print(f"    동시 유각 다리 수 최대 {int(nsw.max())} (지지 다리 최소 {6 - int(nsw.max())})")
    if slip.size:
        print(f"    정상구간 입각 발끝 속도 오차 (발 vs -요구속도) 중앙값 {np.median(slip) * 1000:.2f} mm/s, "
              f"최대 {slip.max() * 1000:.1f} mm/s")
    trans = np.flatnonzero(np.diff(log["gait"]) != 0)
    print("    gait 전환: " + ", ".join(
        f"t={log['t'][k + 1]:.1f}s {core.GAIT_NAMES[log['gait'][k]]}->{core.GAIT_NAMES[log['gait'][k + 1]]}"
        for k in trans))
    for k in trans:
        w = slice(k, min(k + int(gp.transition_time / args.dt) + 1, n))
        print(f"      전환 중 동시 유각 최대 {int(nsw[w].max())}")

    np.savez(os.path.join(args.out, "log.npz"), **log)

    # ---- 4. 그림 ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음, 그림 생략")
        return
    t = log["t"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(t, log["cmd"][:, 0], label="vx")
    ax[0].plot(t, log["cmd"][:, 1], label="vy")
    ax[0].plot(t, log["cmd"][:, 2], label="wz")
    ax[0].set_ylabel("command")
    ax[0].legend(loc="upper right")
    for s in SCENARIO[:-1]:
        ax[0].text(s[0] + 0.2, 0.31, s[4], fontsize=7)
    ax[1].plot(t, log["gait"], "k", label="gait (0 wave, 1 ripple, 2 tripod)")
    ax[1].plot(t, log["duty"], label="duty")
    ax[1].plot(t, log["r"], label="amplitude r")
    ax[1].legend(loc="upper right")
    for li, leg in enumerate(legs):
        ax[2].fill_between(t, li + 0.1, li + 0.9, where=~log["swing"][:, li].astype(bool) & (log["r"] > 0.05),
                           step="mid", color="k", lw=0)
    ax[2].set_yticks(np.arange(6) + 0.5)
    ax[2].set_yticklabels(legs)
    ax[2].set_ylabel("stance (black)")
    ax[3].plot(t, log["q"][:, 0, 1], label="R1 hip")
    ax[3].plot(t, log["q"][:, 0, 3], label="R1 knee")
    ax[3].plot(t, log["q"][:, 0, 0], label="R1 roll")
    ax[3].set_ylabel("rad")
    ax[3].legend(loc="upper right")
    ax[4].plot(t, log["q"][:, 0, 2], label="R1 pr1")
    ax[4].plot(t, log["q"][:, 0, 4], label="R1 pr2")
    ax[4].set_ylabel("m")
    ax[4].set_xlabel("time [s]")
    ax[4].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "timeline.png"), dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sel_t = (t > 22) & (t < 24)
    for li, leg in enumerate(legs):
        ax[0].plot(tg[sel_t, li, 0], tg[sel_t, li, 2], label=leg)
    ax[0].set_title("foot target x-z (tripod, 0.30 m/s)")
    ax[0].set_aspect("equal")
    ax[0].legend()
    sel_t = (t > 28) & (t < 31)
    for li, leg in enumerate(legs):
        ax[1].plot(tg[sel_t, li, 0], tg[sel_t, li, 1], label=leg)
    ax[1].set_title("foot target x-y (turn in place)")
    ax[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "foot_paths.png"), dpi=110)

    # 측면 스틱 그림: 기준 자세 (이전 무릎 0 vs 지금)
    fig, ax = plt.subplots(figsize=(5, 6))
    for qq, lab, col in ((q_nom, "numerical IK (now)", "C0"), (None, "knee 0 (before)", "C3")):
        if qq is None:
            qq = q_nom.clone()
            qq[:, 3] = 0.0
            qq[:, 1] = 0.0
            qq[:, 2] = qq[:, 4] = (0.84 - 0.66) / 2
        pts = [np.zeros(3)]
        R = np.eye(3)
        p = np.zeros(3)
        for k, js in enumerate(core.hugo_v3_leg_chain((0, 0, 0)).joints):
            p = p + R @ np.array(js.origin)
            if js.type == "revolute":
                a = np.array(js.axis)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                v = float(qq[0, k]) if k != 0 else 0.0
                R = R @ (np.eye(3) + math.sin(v) * K + (1 - math.cos(v)) * K @ K)
            else:
                p = p + R @ np.array(js.axis) * float(qq[0, k])
            pts.append(p.copy())
        pts.append(p + R @ np.array([0, 0, -0.30]))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 2], "o-", color=col, label=lab)
    ax.axhline(-0.74, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title("leg side view (x forward)")
    ax.legend()
    fig.savefig(os.path.join(args.out, "posture.png"), dpi=110)
    print(f"[4] 그림 저장: {args.out}/timeline.png, foot_paths.png, posture.png")


if __name__ == "__main__":
    main()
