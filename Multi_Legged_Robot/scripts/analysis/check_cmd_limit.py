"""명령 속도 제한(걸음새별 최대 속도) 전후 비교. Isaac 불필요, CPG 만 돈다.

    python scripts/analysis/check_cmd_limit.py [--envs 512] [--seconds 60] [--rough 0.0]

학습과 같은 방식으로 명령을 뽑는다 (범위, 4~8 s 마다 재샘플링, 5 % 는 서 있기).
걸음새 선택과 CPG 는 ActionTerm (cpg_ik_action.py) 과 같은 순서로 돈다:
    50 Hz: 명령 갱신 -> (제한) -> 걸음새 선택,   200 Hz: CPG 4 스텝

재는 것
    - 못 낸 시간: 보상이 요구하는 명령의 가장 빠른 다리 속도가, 그 순간 CPG 가 낼 수
      있는 속도 (2 * stride_max * f / duty, 전환 중이면 섞인 값) 를 2 % 넘게 초과한 시간
    - 걸음새별 시간 비율, 전환 중인 시간, 분당 전환 횟수
    - 보상이 요구하는 평균 속도 (제한으로 얼마나 줄었는지)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

torch.set_num_threads(1)  # 작은 텐서 연산이라 스레드를 늘리면 오히려 느려진다

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "..", "source", "Multi_Legged_Robot", "Multi_Legged_Robot", "tasks",
                    "manager_based", "multi_legged_robot", "cpg_core.py")
spec = importlib.util.spec_from_file_location("cpg_core", CORE)
core = importlib.util.module_from_spec(spec)
sys.modules["cpg_core"] = core
spec.loader.exec_module(core)

# multi_legged_robot_cpg2_env_cfg.py 와 같은 값
CMD_X, CMD_Y, CMD_W = (-0.25, 0.35), (-0.15, 0.15), (-0.35, 0.35)
RESAMPLE_S, STANDING = (4.0, 8.0), 0.05
EPISODE_S = 20.0


def run(limit: bool, n: int, seconds: float, rough: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    dt, dec = 1.0 / 200.0, 4
    step_dt = dt * dec
    gp, sp = core.GaitParams(), core.SelectorParams()
    _, _, nominal, _ = core.build_hugo(core.HugoPostureCfg())
    cpg = core.GaitCPG(n, nominal, gp, initial_gait="ripple")
    sel = core.GaitSelector(n, sp, core.gait_speed_caps(gp))

    def uni(lo, hi, k):
        return lo + (hi - lo) * torch.rand(k, generator=g)

    sampled = torch.zeros(n, 3)
    standing = torch.zeros(n, dtype=torch.bool)
    t_left = torch.zeros(n)
    t_ep = torch.zeros(n)
    first = torch.ones(n, dtype=torch.bool)
    cmd_sel_f = torch.zeros(n, 3)
    rough_t = torch.full((n,), rough)

    steps = int(seconds / step_dt)
    miss = torch.zeros(n)
    gait_time = torch.zeros(3)
    blend_time = 0.0
    n_switch = 0
    speed_sum = 0.0
    scale_sum = 0.0
    for _ in range(steps):
        # ---- 에피소드 끝 -> 리셋 (걸음새는 유지, 위상/진폭만 초기화: ActionTerm.reset 과 같다)
        done = t_ep >= EPISODE_S
        if done.any():
            ids = done.nonzero().squeeze(-1)
            cpg.reset(ids)
            sel.reset(ids)
            first[ids] = True
            cmd_sel_f[ids] = 0.0
            t_ep[ids] = 0.0
            t_left[ids] = 0.0
        # ---- 명령 재샘플링
        rs = t_left <= 0.0
        k = int(rs.sum())
        if k:
            sampled[rs] = torch.stack([uni(*CMD_X, k), uni(*CMD_Y, k), uni(*CMD_W, k)], dim=1)
            standing[rs] = torch.rand(k, generator=g) <= STANDING
            t_left[rs] = uni(*RESAMPLE_S, k)
        t_left -= step_dt
        t_ep += step_dt
        raw = torch.where(standing[:, None], torch.zeros_like(sampled), sampled)
        scale = cpg.command_scale(raw) if limit else torch.ones(n)
        cmd = raw * scale[:, None]                     # 관측 / 보상 / CPG 가 쓰는 명령

        # ---- 걸음새 선택 (줄이기 전 명령, 저역통과)
        a = step_dt / (gp.cmd_tau + step_dt)
        cmd_sel_f += a * (raw - cmd_sel_f)
        cmd_sel_f[first] = raw[first]
        speed = torch.norm(cpg.leg_demand(cmd_sel_f), dim=-1).amax(dim=1)
        before = cpg.gait_cur.clone()
        want, imm = sel.step(step_dt, speed, rough_t, cpg.gait_cur, first)
        cpg.request_gait(want, imm)
        n_switch += int(((cpg.gait_cur != before) & ~first).sum())
        first[:] = False

        for _ in range(dec):
            cpg.step(dt, cmd)

        # ---- 집계: 보상이 요구하는 속도 vs 지금 낼 수 있는 속도
        _, duty, freq = cpg.current()
        cap_now = 2.0 * gp.stride_max * freq / duty
        need = torch.norm(cpg.leg_demand(cmd), dim=-1).amax(dim=1)
        miss += (need > 1.02 * cap_now).float()
        gait_time += torch.bincount(cpg.gait_cur, minlength=3).float()
        blend_time += float((cpg.blend < 1.0).sum())
        speed_sum += float(torch.norm(cmd[:, :2], dim=1).sum())
        scale_sum += float(scale.sum())
    tot = steps * n
    return {
        "miss": float(miss.sum()) / tot,
        "gait": (gait_time / tot).tolist(),
        "blend": blend_time / tot,
        "switch_per_min": n_switch / (n * seconds / 60.0),
        "speed": speed_sum / tot,
        "scale": scale_sum / tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rough", type=float, nargs="*", default=[0.0, 0.05],
                    help="거칠기 입력 [m]. 0.025 가 험지 판정 기준이다")
    args = ap.parse_args()
    gp = core.GaitParams()
    caps = core.gait_speed_caps(gp).tolist()
    print("걸음새 최대 속도: " + ", ".join(f"{g} {c:.3f}" for g, c in zip(core.GAIT_NAMES, caps)) + " m/s")
    print(f"{args.envs} 대 x {args.seconds:.0f} s, 명령 범위 vx {CMD_X} vy {CMD_Y} wz {CMD_W}")
    hdr = f"{'거칠기':>6} {'제한':>4} | {'못 낸 시간':>9} | {'wave':>6} {'ripple':>6} {'tripod':>6} | " \
          f"{'전환 중':>7} {'전환/분':>7} | {'평균 명령 속도':>12} {'평균 배율':>8}"
    print(hdr)
    for rough in args.rough:
        for limit in (False, True):
            r = run(limit, args.envs, args.seconds, rough)
            print(f"{rough:6.3f} {'켬' if limit else '끔':>4} | {100 * r['miss']:8.1f}% | "
                  + " ".join(f"{100 * v:5.1f}%" for v in r["gait"])
                  + f" | {100 * r['blend']:6.1f}% {r['switch_per_min']:7.2f} | "
                  f"{r['speed']:10.3f} m/s {r['scale']:8.3f}", flush=True)


if __name__ == "__main__":
    main()
