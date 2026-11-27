"""걸음새 전환 진단: 학습된 정책을 돌리며 거칠기 / 명령 / 선택된 걸음새 / 전환 품질을 기록한다.

Isaac Lab 3.0 (kitless) 용. IsaacLab 폴더에서 실행:

    .venv/bin/python <이 파일> --checkpoint logs/rsl_rl/hugo_cpg2/<run>/model_2700.pt --mode course
    .venv/bin/python <이 파일> --checkpoint ... --mode mix --num_envs 1024 --seconds 20

mode
    course : 16 대 (난이도 4 단계 x 속도 4 종류) 가 평지 -> 격자 -> 계단 -> 평지를 차례로 지난다.
             지형이 바뀔 때 거칠기 값과 걸음새가 어떻게 따라가는지 본다.
    mix    : 학습과 같은 지형 / 무작위 명령. 걸음새별 시간 비율과 지형별 거칠기 분포를 본다.

--limit 0 이면 명령 속도 제한을 끈다 (제한 없이 학습한 이전 체크포인트를 그대로 볼 때).
"""

import argparse
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--mode", choices=("course", "mix"), default="course")
p.add_argument("--num_envs", type=int, default=None)
p.add_argument("--seconds", type=float, default=None)
p.add_argument("--limit", type=int, default=1)
p.add_argument("--out", default=None)
add_launcher_args(p)
a = p.parse_args()
import sys  # noqa: E402

sys.argv = sys.argv[:1]  # 뒤에 불리는 모듈이 sys.argv 를 다시 파싱하지 않도록

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.terrains as terrain_gen  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import Multi_Legged_Robot.tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from Multi_Legged_Robot.tasks.manager_based.multi_legged_robot import multi_legged_robot_cpg2_env_cfg as C  # noqa: E402
from Multi_Legged_Robot.tasks.manager_based.multi_legged_robot.cpg_core import GAIT_NAMES  # noqa: E402

COURSE = a.mode == "course"
SECONDS = a.seconds or (60.0 if COURSE else 20.0)
N = a.num_envs or (16 if COURSE else 1024)
TILE = 4.0
# course 속도 종류 (m/s). 마지막은 시간에 따라 바뀐다 (속도로 인한 전환 확인)
PROFILES = ("0.30", "0.20", "0.12", "varying")


def profile_speed(k: int, t: float) -> float:
    if k == 0:
        return 0.30
    if k == 1:
        return 0.20
    if k == 2:
        return 0.12
    sched = ((0, 0.08), (12, 0.16), (24, 0.30), (36, 0.16), (48, 0.08))
    v = sched[0][1]
    for t0, s in sched:
        if t >= t0:
            v = s
    return v


task = "Hugo-Hexapod-CPG2-Play-v0" if COURSE else "Hugo-Hexapod-CPG2-v0"
env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", play_mode=COURSE)
env_cfg.scene.num_envs = N
env_cfg.episode_length_s = SECONDS + 1.0 if COURSE else 20.0
env_cfg.commands.base_velocity.limit_to_gait = bool(a.limit)
gen = env_cfg.scene.terrain.terrain_generator
if COURSE:
    # 열(y 방향)이 지형 종류, 행(x 방향)이 난이도. 로봇은 +y 를 보고 서서 앞으로 걷는다.
    gen.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.35, grid_height_range=C.GRID_HEIGHT, platform_width=0.5),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=C.STEP_HEIGHT, step_width=0.35,
            platform_width=1.0, border_width=0.25, holes=False),
        "flat_end": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
    }
    gen.size = (TILE, TILE)
    gen.num_rows, gen.num_cols = 4, 4
    gen.curriculum = True
    gen.difficulty_range = (0.1, 0.9)
    gen.use_cache = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    pr = env_cfg.events.reset_base.params["pose_range"]
    pr["x"], pr["y"], pr["yaw"] = (0.0, 0.0), (0.0, 0.0), (math.pi / 2, math.pi / 2)
else:
    env_cfg.scene.terrain.max_init_terrain_level = gen.num_rows - 1   # 모든 난이도에 고르게

# 열 번호 -> 지형 이름 (TerrainGenerator 의 curriculum 배치 규칙과 같다)
props = np.array([s.proportion for s in gen.sub_terrains.values()], dtype=float)
props = np.cumsum(props / props.sum())
names = list(gen.sub_terrains.keys())
COL_NAMES = [names[int(np.min(np.where(j / gen.num_cols + 0.001 < props)[0]))] for j in range(gen.num_cols)]

with launch_simulation(env_cfg, a):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.seed = agent_cfg.seed
    env = gym.make(task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    terrain = u.scene.terrain
    if COURSE:
        rows = torch.arange(N, device=dev) // 4 % gen.num_rows
        terrain.env_origins[:] = terrain.terrain_origins[rows, 0]
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg)
    runner.load(a.checkpoint)
    policy = runner.get_inference_policy(device=dev)

    robot = u.scene["robot"]
    jn = list(robot.joint_names)
    knee_ids = [i for i, n in enumerate(jn) if "joint3_pitch" in n]
    hip_ids = [i for i, n in enumerate(jn) if "joint2_pitch" in n]
    roll_ids = [i for i, n in enumerate(jn) if "joint1_roll" in n]
    pr_ids = [i for i, n in enumerate(jn) if "prismatic" in n]
    term = u.action_manager.get_term("cpg")
    cpg, sel = term._cpg, term._selector
    cmd_term = u.command_manager.get_term("base_velocity")
    col_y = terrain.terrain_origins[0, :, 1]
    row_x = terrain.terrain_origins[:, 0, 0]
    tq_attr = "applied_torque" if hasattr(robot.data, "applied_torque") else "computed_torque"
    print(f"[diag] mode {a.mode}  envs {N}  seconds {SECONDS}  limit {a.limit}  cols {COL_NAMES}  torque '{tq_attr}'")

    obs, _ = env.reset() if COURSE else (env.get_observations(), None)
    if COURSE:
        obs = env.get_observations()
    x0 = robot.data.root_pos_w.torch[:, 0].clone()
    steps = int(SECONDS / u.step_dt)
    keys = ("gait", "blend", "rough", "sel_speed", "cmd", "raw", "vel", "pos", "tilt", "phase_err", "n_swing",
            "knee_vel", "hip_vel", "pr_vel", "knee_tq", "hip_tq", "roll_tq", "knee_tq_mean", "fell", "timeout",
            "col", "row", "cap")
    rec = {k: [] for k in keys}
    for i in range(steps):
        t = i * u.step_dt
        if COURSE:
            # 진행 방향(+y) 유지: 헤딩과 옆 밀림을 약하게 잡아 준다
            yaw = robot.data.heading_w.torch
            wz = (1.0 * (math.pi / 2 - yaw)).clamp(-0.3, 0.3)
            vy = (0.5 * (robot.data.root_pos_w.torch[:, 0] - x0)).clamp(-0.08, 0.08)
            vx = torch.tensor([profile_speed(k % 4, t) for k in range(N)], device=dev)
            cmd_term._sampled[:] = torch.stack([vx, vy, wz], dim=1)
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        off, duty, freq = cpg.current()
        perr = (cpg.phi - cpg.phi[:, :1]) - (off - off[:, :1])
        perr = (perr - torch.round(perr)).abs().amax(dim=1)
        pos = robot.data.root_pos_w.torch
        g = robot.data.projected_gravity_b.torch
        jv = robot.data.joint_vel.torch
        tq = getattr(robot.data, tq_attr).torch
        row = {
            "gait": cpg.gait_cur, "blend": cpg.blend, "rough": term._rough,
            "sel_speed": torch.norm(cpg.leg_demand(term._cmd_sel_f), dim=-1).amax(dim=1),
            "cmd": cmd_term.command[:, :3], "raw": cmd_term.unclipped_command[:, :3],
            "vel": torch.cat([robot.data.root_lin_vel_b.torch[:, :2], robot.data.root_ang_vel_b.torch[:, 2:3]], 1),
            "pos": pos, "tilt": torch.acos((-g[:, 2]).clamp(-1, 1)),
            "phase_err": perr, "n_swing": (cpg._swing & (cpg.r > 0.05)).sum(dim=1),
            "knee_vel": jv[:, knee_ids].abs().amax(1), "hip_vel": jv[:, hip_ids].abs().amax(1),
            "pr_vel": jv[:, pr_ids].abs().amax(1),
            "knee_tq": tq[:, knee_ids].abs().amax(1), "hip_tq": tq[:, hip_ids].abs().amax(1),
            "roll_tq": tq[:, roll_ids].abs().amax(1), "knee_tq_mean": tq[:, knee_ids].abs().mean(1),
            "fell": u.termination_manager.terminated, "timeout": u.termination_manager.time_outs,
            "col": (pos[:, 1:2] - col_y[None, :]).abs().argmin(dim=1),
            "row": (pos[:, 0:1] - row_x[None, :]).abs().argmin(dim=1),
            "cap": 2.0 * cpg.p.stride_max * freq / duty,
        }
        for k in keys:
            rec[k].append(row[k].detach().cpu().numpy().copy())
        if i % 250 == 0:
            print(f"[diag] t {t:5.1f} s  gait {np.bincount(rec['gait'][-1], minlength=3).tolist()}  "
                  f"rough mean {rec['rough'][-1].mean():.3f}  fell so far {int(np.sum(rec['fell']))}")
    R = {k: np.stack(v) for k, v in rec.items()}          # (T, N, ...)
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), f"diag_{a.mode}_limit{a.limit}.npz")
    np.savez_compressed(out, col_names=np.array(COL_NAMES), dt=u.step_dt, profiles=np.array(PROFILES),
                        up_ripple=sel.p.up_ripple, up_tripod=sel.p.up_tripod,
                        rough_threshold=sel.p.rough_threshold, **R)
    print("[diag] saved", out)
    env.close()

# =====================================================================
# 분석
# =====================================================================
dt = float(u.step_dt)
T = R["gait"].shape[0]
gait, blend, rough = R["gait"], R["blend"], R["rough"]
col = R["col"]
print("\n===== 걸음새 시간 비율 =====")
tot = np.bincount(gait.ravel(), minlength=3) / gait.size
print("전체: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in zip(GAIT_NAMES, tot))
      + f"   전환 중 {100 * (blend < 1).mean():.1f}%")
print(f"{'지형':>12} | {'시간':>6} | wave ripple tripod | 거칠기 중앙값 / 90% / 기준(0.025) 초과 | 속도오차 | 못 낸 시간")
need = np.linalg.norm(R["raw"][..., :2], axis=-1)        # 참고용 (회전 제외)
err = np.linalg.norm(R["cmd"][..., :2] - R["vel"][..., :2], axis=-1)
leg_need = R["sel_speed"]
for j, name in enumerate(COL_NAMES):
    m = col == j
    if not m.any():
        continue
    gs = np.bincount(gait[m], minlength=3) / m.sum()
    print(f"{name:>12} | {100 * m.mean():5.1f}% | " + " ".join(f"{100 * v:5.1f}" for v in gs)
          + f" | {np.median(rough[m]):.3f} / {np.percentile(rough[m], 90):.3f} / {100 * (rough[m] > 0.025).mean():5.1f}%"
          + f" | {err[m].mean():.3f} | {100 * (leg_need[m] > 1.02 * R['cap'][m]).mean():5.1f}%")

print("\n===== 전환 =====")
sw = (gait[1:] != gait[:-1]) & ~(R["fell"][:-1] | R["timeout"][:-1])     # (T-1, N), 리셋으로 바뀐 것은 제외
n_sw = int(sw.sum())
minutes = gait.size * dt / 60.0
print(f"전환 {n_sw} 회, 로봇 1 대당 분당 {n_sw / minutes:.2f} 회")
win3, win2 = int(3.0 / dt), int(2.0 / dt)
chatter = fall_after = tilt_after = 0
up = down = 0
for t_i, e in zip(*np.nonzero(sw)):
    before, after = gait[t_i, e], gait[t_i + 1, e]
    up += after > before
    down += after < before
    seg = gait[t_i + 1:t_i + 1 + win3, e]
    chatter += bool((seg == before).any())
    fall_after += bool(R["fell"][t_i + 1:t_i + 1 + win2, e].any())
    tilt_after += bool((R["tilt"][t_i + 1:t_i + 1 + win2, e] > math.radians(20)).any())
if n_sw:
    print(f"  빠른 쪽으로 {up} 회, 느린 쪽으로 {down} 회")
    print(f"  3 초 안에 되돌아간 전환 {chatter} 회 ({100 * chatter / n_sw:.1f}%)")
    print(f"  전환 후 2 초 안에 넘어짐 {fall_after} 회 ({100 * fall_after / n_sw:.1f}%), "
          f"몸통 20 도 넘게 기움 {tilt_after} 회 ({100 * tilt_after / n_sw:.1f}%)")
falls = int(R["fell"].sum())
print(f"  참고: 전체 넘어짐 {falls} 회 = 로봇 1 대당 분당 {falls / minutes:.3f} 회 "
      f"(아무 2 초 구간에 넘어질 확률 {100 * falls / minutes / 30:.2f}%)")
tr, st = blend < 1.0, blend >= 1.0
for lab, m in (("전환 중", tr), ("전환 아님", st)):
    if m.any():
        print(f"  {lab:>6}: 위상 오차 최대 {R['phase_err'][m].max():.3f} 주기, 동시에 든 다리 최대 {int(R['n_swing'][m].max())}, "
              f"무릎 속도 최대 {R['knee_vel'][m].max():.2f} / 99.9% {np.percentile(R['knee_vel'][m], 99.9):.2f} rad/s, "
              f"hip 속도 최대 {R['hip_vel'][m].max():.2f}, 직동 속도 최대 {R['pr_vel'][m].max():.3f} m/s")

print("\n===== 토크 / 속도 (전체) =====")
for lab, k in (("무릎", "knee_tq"), ("hip", "hip_tq"), ("roll", "roll_tq")):
    print(f"  {lab} 토크 최대 {R[k].max():.0f}, 99% {np.percentile(R[k], 99):.0f}, 중앙값 {np.median(R[k]):.0f} N·m")
print(f"  무릎 토크 (6 다리 평균 절댓값) 평균 {R['knee_tq_mean'].mean():.0f} N·m")
print(f"  넘어짐: " + ", ".join(
    f"{n} {int(R['fell'][col == j].sum())}" for j, n in enumerate(COL_NAMES) if (col == j).any()))

if COURSE:
    print("\n===== course: 로봇별 =====")
    for e in range(N):
        s = np.flatnonzero(sw[:, e])
        seq = " ".join(f"{t_i * dt:.0f}s:{GAIT_NAMES[gait[t_i, e]][0]}>{GAIT_NAMES[gait[t_i + 1, e]][0]}" for t_i in s)
        print(f"  난이도행 {e // 4} 속도 {PROFILES[e % 4]:>7}: 이동 {R['pos'][-1, e, 1] - R['pos'][0, e, 1]:5.1f} m, "
              f"넘어짐 {int(R['fell'][:, e].sum())}, 전환 {len(s)} [{seq}]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.arange(T) * dt
        fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=True)
        for e in range(N):
            ax = axes[e // 4, e % 4]
            ax.plot(tt, rough[:, e] * 10, color="C1", lw=0.8, label="roughness x10 [m]")
            ax.axhline(0.25, color="C1", ls=":", lw=0.8)
            ax.plot(tt, R["sel_speed"][:, e], color="C0", lw=0.8, label="leg speed demand [m/s]")
            ax.axhline(float(sel.p.up_ripple), color="C0", ls=":", lw=0.6)
            ax.axhline(float(sel.p.up_tripod), color="C0", ls=":", lw=0.6)
            ax.plot(tt, np.linalg.norm(R["vel"][:, e, :2], axis=1), color="C2", lw=0.6, alpha=0.7, label="actual speed")
            ax.step(tt, gait[:, e] * 0.2 + 0.45, color="k", lw=1.2, where="post", label="gait (wave/ripple/tripod)")
            for j in range(4):
                m = col[:, e] == j
                ax.fill_between(tt, 0, 0.03, where=m, color=f"C{j + 4}", step="mid", lw=0)
            for t_f in tt[R["fell"][:, e]]:
                ax.axvline(t_f, color="r", lw=1.5)
            ax.set_ylim(0, 0.9)
            ax.set_title(f"difficulty row {e // 4}, speed {PROFILES[e % 4]}", fontsize=9)
        axes[0, 0].legend(fontsize=6, loc="upper left")
        fig.suptitle("bottom band: terrain (flat / grid / stairs / flat), red line: fall, dotted: thresholds")
        fig.tight_layout()
        png = out.replace(".npz", ".png")
        fig.savefig(png, dpi=90)
        print("[diag] figure", png)
    except Exception as ex:  # noqa: BLE001
        print("[diag] 그림 실패:", ex)
