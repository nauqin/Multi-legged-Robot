"""보폭(stride_max) 사전 검증: 물리엔진 없이 CPG + IK 만 돌린다.

학습과 같은 방식(무작위 명령 4~8초마다 재샘플, 걸음새 자동 전환, 200 Hz, IK 2회 반복)으로
여러 env 를 병렬로 돌리고 관절 속도/범위, IK 오차, 이웃 다리 간격을 잰다.
정책 residual, 토크, 몸통-다리 간섭은 보지 않는다.

사용:
    python stride_check.py                      # 전체 (1024 env x 60 s, 5~10 분)
    SC_N=256 SC_T=30 SC_CASES=base,A,G python stride_check.py   # 빠르게 (경우당 약 35 초)

경우(SC_CASES): base 0.15 / A 0.18 / B 0.18(전환 기준 비율 조정) / C 0.20 / D 0.22 / E 0.165
               F 0.18 + 앞뒤 발 4 cm 벌림 / G 0.18 + 6 cm 벌림 / H 0.15 + 6 cm 벌림
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "source", "Multi_Legged_Robot", "Multi_Legged_Robot", "tasks",
    "manager_based", "multi_legged_robot", "cpg_core.py")
spec = importlib.util.spec_from_file_location("cpg_core", CORE)
core = importlib.util.module_from_spec(spec)
sys.modules["cpg_core"] = core
spec.loader.exec_module(core)

dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = 1.0 / 200.0
POLICY_DT = 0.02
N = int(os.environ.get("SC_N", "1024"))
T_END = float(os.environ.get("SC_T", "60"))
BASE = dict(vx=(-0.25, 0.35), vy=(-0.15, 0.15), wz=(-0.35, 0.35))
VLIM = torch.tensor([5.0, 5.0, 0.25, 5.0, 0.25], device=dev, dtype=torch.float64)
ADJ = [(0, 1), (1, 2), (3, 4), (4, 5)]          # 같은 쪽 이웃 다리 (R1-R2, R2-R3, L1-L2, L2-L3)
JN = ("roll", "hip", "pr1", "knee", "pr2")


def rot_x(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)


def rot_y(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


def leg_points(chain, q):
    """q (N,L,5) -> hip, knee, foot (N,L,3) base 기준."""
    hip = chain.origins[:, 0, :].unsqueeze(0).expand(q.shape[0], -1, -1)
    R = rot_x(q[..., 0]) @ rot_y(q[..., 1])
    down = torch.zeros(*q.shape[:2], 3, device=q.device, dtype=q.dtype)
    down[..., 2] = -(0.03 + q[..., 2] + 0.30)
    knee = hip + (R @ down.unsqueeze(-1)).squeeze(-1)
    foot, _ = chain.fk(q, with_jacobian=False)
    return hip, knee, foot


def seg_min_dist(a0, a1, b0, b1, k=21):
    """선분 a, b (.., 3) 사이 최소 거리 (표본점 근사)."""
    s = torch.linspace(0, 1, k, device=a0.device, dtype=a0.dtype)
    pa = a0.unsqueeze(-2) + (a1 - a0).unsqueeze(-2) * s[:, None]          # (.., k, 3)
    pb = b0.unsqueeze(-2) + (b1 - b0).unsqueeze(-2) * s[:, None]
    d = torch.cdist(pa, pb)                                               # (.., k, k)
    return d.flatten(-2).amin(-1)


def run(stride_max, scale_thr, label, xoff=0.0):
    k = stride_max / 0.15
    torch.manual_seed(0)
    posture = core.HugoPostureCfg()
    posture.foot_x_offset = (xoff, 0.0, -xoff, xoff, 0.0, -xoff)   # R1 R2 R3 L1 L2 L3: 앞다리는 앞으로, 뒷다리는 뒤로
    chain, ik, nominal, q_nom = core.build_hugo(posture, device=dev, dtype=torch.float64)
    gp = core.GaitParams()
    gp.stride_max = stride_max
    sp = core.SelectorParams()
    if scale_thr:
        sp.up_ripple *= k
        sp.up_tripod *= k
        sp.hysteresis *= k
    caps = core.gait_speed_caps(gp, device=dev, dtype=torch.float64)
    cpg = core.GaitCPG(N, nominal, gp, device=dev, dtype=torch.float64, initial_gait="ripple")
    sel = core.GaitSelector(N, sp, caps, device=dev, dtype=torch.float64)

    lo = torch.tensor([BASE["vx"][0], BASE["vy"][0], BASE["wz"][0]], device=dev, dtype=torch.float64)
    hi = torch.tensor([BASE["vx"][1], BASE["vy"][1], BASE["wz"][1]], device=dev, dtype=torch.float64)
    lo[:2] *= k
    hi[:2] *= k                                  # 선속도 명령만 보폭 비율로 키운다 (회전은 그대로)

    def sample(n):
        c = lo + (hi - lo) * torch.rand(n, 3, device=dev, dtype=torch.float64)
        c[torch.rand(n, device=dev) < 0.05] = 0.0
        return c

    cmd = sample(N)
    # 가장 가혹한 조합도 일부 env 에 강제로 넣는다 (8초마다 순서를 뒤집어 전환을 유발)
    ext = torch.tensor([[hi[0], 0, 0], [lo[0], 0, 0], [0, hi[1], 0], [hi[0], hi[1], hi[2]], [hi[0], lo[1], lo[2]],
                        [0, 0, hi[2]], [hi[0], 0, hi[2]], [lo[0], hi[1], lo[2]]], device=dev, dtype=torch.float64)
    rep = max(1, min(8, N // (4 * ext.shape[0])))
    n_ext = ext.shape[0] * rep
    next_t = 4.0 + 4.0 * torch.rand(N, device=dev, dtype=torch.float64)
    rough = torch.zeros(N, device=dev, dtype=torch.float64)
    rough[N // 2:] = 0.05                         # 절반은 '험지' 판정 (걸음새 한 단계 낮춤)

    q = q_nom.unsqueeze(0).repeat(N, 1, 1)
    q_prev = q.clone()
    first = torch.ones(N, dtype=torch.bool, device=dev)
    vmax = torch.zeros(5, device=dev, dtype=torch.float64)
    over = torch.zeros(5, device=dev, dtype=torch.float64)
    vsamples, err_samples = [], []
    err_max = 0.0
    qmin = torch.full((5,), 1e9, device=dev, dtype=torch.float64)
    qmax = -qmin.clone()
    knee_at_lim = 0.0
    pr_at_lim = 0.0
    dmin = dict(link=1e9, knee=1e9, foot=1e9)
    nsw_max = 0
    gait_time = torch.zeros(3, device=dev, dtype=torch.float64)
    n_trans = 0
    steps = int(T_END / DT)
    count = 0
    for i in range(steps):
        t = i * DT
        if i % int(POLICY_DT / DT) == 0:          # 50 Hz: 명령 재샘플 + 걸음새 결정
            rs = t >= next_t
            if rs.any():
                cmd[rs] = sample(int(rs.sum()))
                next_t[rs] = t + 4.0 + 4.0 * torch.rand(int(rs.sum()), device=dev, dtype=torch.float64)
            cmd[:n_ext] = (ext if (int(t) // 8) % 2 == 0 else ext.flip(0)).repeat(rep, 1)
            speed = torch.norm(cpg.leg_demand(torch.where(first[:, None], cmd, cpg.cmd_f)), dim=-1).amax(dim=1)
            before = cpg.gait_cur.clone()
            g, imm = sel.step(POLICY_DT, speed, rough, cpg.gait_cur, first)
            cpg.request_gait(g, imm)
            n_trans += int((cpg.gait_cur != before).sum())
            first[:] = False
        tgt = cpg.step(DT, cmd)
        q, err = ik.solve(tgt, q, iters=2)
        qd = ((q - q_prev) / DT).abs()
        q_prev = q.clone()
        if i < 5:
            continue
        count += 1
        vmax = torch.maximum(vmax, qd.amax(dim=(0, 1)))
        over += (qd > VLIM).double().mean(dim=(0, 1))
        err_max = max(err_max, float(err.max()))
        qmin = torch.minimum(qmin, q.amin(dim=(0, 1)))
        qmax = torch.maximum(qmax, q.amax(dim=(0, 1)))
        knee_at_lim += float((q[..., 3] > ik.upper[:, 3] - 1e-4).double().mean())
        pr_at_lim += float(((q[..., [2, 4]] < 0.005 + 1e-5) | (q[..., [2, 4]] > 0.145 - 1e-5)).double().mean())
        gait_time += torch.bincount(cpg.gait_cur, minlength=3).double()
        nsw_max = max(nsw_max, int((cpg._swing & (cpg.r > 0.05)).sum(dim=1).max()))
        if i % 4 == 0:
            vsamples.append(qd.reshape(-1, 5)[::7].clone())
            err_samples.append(err.reshape(-1)[::7].clone())
            hip, knee, foot = leg_points(chain, q)
            for a, b in ADJ:
                d = torch.stack([seg_min_dist(hip[:, a], knee[:, a], hip[:, b], knee[:, b]),
                                 seg_min_dist(knee[:, a], foot[:, a], knee[:, b], foot[:, b]),
                                 seg_min_dist(hip[:, a], knee[:, a], knee[:, b], foot[:, b]),
                                 seg_min_dist(knee[:, a], foot[:, a], hip[:, b], knee[:, b])]).amin()
                dmin["link"] = min(dmin["link"], float(d))
                dmin["knee"] = min(dmin["knee"], float(torch.norm(knee[:, a] - knee[:, b], dim=-1).amin()))
                dmin["foot"] = min(dmin["foot"], float(torch.norm(foot[:, a] - foot[:, b], dim=-1).amin()))
    vs = torch.cat(vsamples)
    es = torch.cat(err_samples)
    p999 = torch.quantile(vs[torch.randperm(vs.shape[0], device=dev)[:2_000_000]], 0.999, dim=0)
    print(f"\n===== {label}: stride_max {stride_max:.3f} m, 앞뒤 발 벌림 {xoff * 100:.0f} cm, "
          f"전환 기준 {'비율 조정' if scale_thr else '그대로'} =====")
    print("  걸음새 최대 속도 [m/s]: " + ", ".join(f"{g} {c:.3f}" for g, c in zip(core.GAIT_NAMES, caps.tolist())))
    print("  기준 자세 (roll, hip, pr1, knee, pr2) 앞/가운데/뒤: "
          + " | ".join(" ".join(f"{v:+.3f}" for v in q_nom[i].tolist()) for i in (0, 1, 2)))
    print(f"  명령 범위: vx {lo[0]:+.2f}~{hi[0]:+.2f}, vy ±{hi[1]:.2f}, wz ±{hi[2]:.2f}")
    print(f"  IK 잔여오차: 최대 {err_max * 1000:.2f} mm, 99.9% {float(torch.quantile(es[:2_000_000], 0.999)) * 1000:.3f} mm")
    print("  관절 속도 최대/한계: " + ", ".join(
        f"{n} {v:.2f}/{lim:g} ({100 * v / lim:.0f}%)" for n, v, lim in zip(JN, vmax.tolist(), VLIM.tolist())))
    print("  관절 속도 99.9%:     " + ", ".join(
        f"{n} {v:.2f} ({100 * v / lim:.0f}%)" for n, v, lim in zip(JN, p999.tolist(), VLIM.tolist())))
    print("  한계 초과 시간 비율: " + ", ".join(f"{n} {100 * o / count:.4f}%" for n, o in zip(JN, over.tolist())))
    print("  관절 범위: " + ", ".join(f"{n} {a:+.3f}~{b:+.3f}" for n, a, b in zip(JN, qmin.tolist(), qmax.tolist())))
    print(f"  무릎이 최소 굽힘 한계에 닿은 시간 {100 * knee_at_lim / count:.3f}%, "
          f"prismatic 끝단에 닿은 시간 {100 * pr_at_lim / count:.3f}%")
    print(f"  이웃 다리 최소 간격(중심선): 링크 {dmin['link'] * 100:.1f} cm (필요 9.0), "
          f"무릎-무릎 {dmin['knee'] * 100:.1f} cm (필요 11.0), 발-발 {dmin['foot'] * 100:.1f} cm (필요 8.0)")
    print("  걸음새 비율: " + ", ".join(f"{g} {100 * x / gait_time.sum():.0f}%" for g, x in zip(core.GAIT_NAMES, gait_time.tolist()))
          + f", 전환 {n_trans}회, 동시에 든 다리 최대 {nsw_max}개")


ALL = {
    "base": (0.15, False, "기준"),
    "A": (0.18, False, "A"),
    "B": (0.18, True, "B"),
    "C": (0.20, True, "C"),
    "D": (0.22, True, "D"),
    "E": (0.165, True, "E"),
    "F": (0.18, True, "F", 0.04),
    "G": (0.18, True, "G", 0.06),
    "H": (0.15, False, "H", 0.06),
}

if __name__ == "__main__":
    for key in os.environ.get("SC_CASES", "base,A,B,C,D").split(","):
        run(*ALL[key])
