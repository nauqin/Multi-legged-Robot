"""CPG 단독 실행 확인 (Isaac Sim). env_cfg 를 전혀 건드리지 않는다.

놓을 위치:  scripts/check_cpg.py

동작 원리
    gym.make 로 환경을 만들되 env.step() 을 쓰지 않는다.
    액션 매니저를 우회하고 직접 set_joint_position_target 을 써서
    시뮬레이션을 한 스텝씩 돌린다. 따라서 정책도, 보상도, 관측도 관여하지 않는다.
    순수하게 "CPG 궤적만으로 이 로봇이 걷는가" 만 본다.

사용 예
    # 평지에서 tripod, 0.35 m/s, 15초 보고 싶다
    python scripts/check_cpg.py --task Hugo-Hexapod-v0 --flat

    # 파라미터를 바꿔가며
    python scripts/check_cpg.py --task Hugo-Hexapod-v0 --flat \
        --omega 5.0 --k1 0.08 --h 0.78 --speed 0.25

    # 화면 없이 수치만
    python scripts/check_cpg.py --task Hugo-Hexapod-v0 --flat --headless

    # 걸음새 비교
    python scripts/check_cpg.py --task Hugo-Hexapod-v0 --flat --gait ripple
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the CPG open-loop on Hugo.")
parser.add_argument("--task", type=str, default="Hugo-Hexapod-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--flat", action="store_true", help="지형을 평지로 바꾼다")

# CPG 파라미터
parser.add_argument("--gait", type=str, default="tripod",
                    choices=["tripod", "ripple", "wave"])
parser.add_argument("--omega", type=float, default=6.3, help="rad/s")
parser.add_argument("--alpha", type=float, default=50.0)
parser.add_argument("--coupling", type=float, default=0.5)
parser.add_argument("--h", type=float, default=0.76, help="입각기 다리 길이 [m]")
parser.add_argument("--k1", type=float, default=0.06, help="유각기 발 들림 [m]")
parser.add_argument("--k2", type=float, default=0.008, help="입각기 압입 [m]")
parser.add_argument("--roll_nom", type=float, default=0.20, help="다리 벌림 [rad]")
parser.add_argument("--speed", type=float, default=0.35, help="목표 속도 [m/s]")

# 실행
parser.add_argument("--settle", type=float, default=1.0, help="착지 대기 [s]")
parser.add_argument("--duration", type=float, default=15.0, help="보행 시간 [s]")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import Multi_Legged_Robot.tasks  # noqa: F401


# =====================================================================
# ver3.1 하드웨어 상수
# =====================================================================

L0 = 0.66
STROKE = 0.15
PRISMATIC_VEL_LIMIT = 0.25

LEG_PREFIXES = ("R1", "R2", "R3", "L1", "L2", "L3")
N_LEGS = 6

GAIT_PHASES = {
    "tripod": (0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    "ripple": (0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6),
    "wave":   (2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6),
}
GAIT_DUTY = {"tripod": 0.5, "ripple": 2 / 3, "wave": 5 / 6}

# 연속 토크 판정선 (액추에이터 문서 3절)
TORQUE_LIMITS = {"roll": 80.0, "hip": 160.0, "knee": 120.0, "prismatic": 500.0}


class SimpleCPG:
    """cpg_action.py 와 같은 수식. 의존성 없이 단독으로 쓰려고 복사했다."""

    def __init__(self, num_envs, device, args):
        self.device = device
        self.a = args

        phase = torch.tensor(GAIT_PHASES[args.gait], device=device) * 2.0 * math.pi
        theta = phase.unsqueeze(1) - phase.unsqueeze(0)
        self.C = torch.cos(theta)
        self.S = torch.sin(theta)
        self.C.fill_diagonal_(0.0)
        self.S.fill_diagonal_(0.0)
        self.phase = phase

        sign = [1.0 if p.startswith("L") else -1.0 for p in LEG_PREFIXES]
        self.roll_sign = torch.tensor(sign, device=device)

        self.x = torch.zeros(num_envs, N_LEGS, device=device)
        self.y = torch.zeros(num_envs, N_LEGS, device=device)
        self.reset()

        f = args.omega / (2.0 * math.pi)
        self.d_step = args.speed * GAIT_DUTY[args.gait] / (2.0 * f)
        
        # 듀티에 맞춰 입각기/유각기 각속도를 분리한다.
        # 이게 없으면 위/아래 반 바퀴가 같은 속도라 duty가 0.5로 고정되고,
        # ripple/wave는 위상만 흩어진 채 항상 3개가 떠 있게 된다.
        T = 2.0 * math.pi / args.omega
        beta = GAIT_DUTY[args.gait]
        self.omega_st = math.pi / (beta * T)
        self.omega_sw = math.pi / ((1.0 - beta) * T)
        self.b_sharp = 20.0

    def reset(self):
        """극한 순환 위, 목표 위상. 무작위로 놓으면 과도구간에 발이 튄다."""
        self.x[:] = torch.cos(self.phase)
        self.y[:] = torch.sin(self.phase)

    def step(self, dt):
        a = self.a
        x, y = self.x, self.y
        radial = a.alpha * (1.0 - (x * x + y * y))

        # y > 0 (유각기) -> omega_sw,  y < 0 (입각기) -> omega_st
        s = torch.sigmoid(self.b_sharp * y)
        omega = self.omega_sw * s + self.omega_st * (1.0 - s)
        
        x_dot = radial * x - omega * y
        y_dot = radial * y + omega * x

        if a.coupling != 0.0:
            x_dot = x_dot + a.coupling * (x @ self.C.t() - y @ self.S.t())
            y_dot = y_dot + a.coupling * (x @ self.S.t() + y @ self.C.t())

        self.x = x + dt * x_dot
        self.y = y + dt * y_dot

    def targets(self):
        """(roll, hip_pitch, d) 를 각각 (num_envs, 6) 으로 반환."""
        a = self.a
        px_l = -self.d_step * self.x
        pz_l = torch.where(self.y > 0.0,
                           -a.h + a.k1 * self.y,
                           -a.h + a.k2 * self.y)

        roll_nom = a.roll_nom * self.roll_sign
        c = torch.cos(roll_nom).unsqueeze(0)
        s = torch.sin(roll_nom).unsqueeze(0)
        px, py, pz = px_l, -s * pz_l, c * pz_l

        L = torch.sqrt(px * px + py * py + pz * pz).clamp_min(1e-6)
        q_roll = torch.atan2(py, -pz)
        q_hip = -torch.asin((px / L).clamp(-1.0, 1.0))
        d = ((L - L0) * 0.5).clamp(0.0, STROKE)
        return q_roll, q_hip, d


def main():
    # ---------------- 환경 ------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)

    if args_cli.flat:
        try:
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None
            print("[INFO] terrain -> plane")
        except Exception as e:
            print(f"[WARN] 평지 전환 실패, 원래 지형으로 진행: {e}")

    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.6)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    sim = env.sim
    scene = env.scene
    device = env.device
    dt = sim.get_physics_dt()

    # ---------------- 관절 인덱스 ------------------------------------
    names = list(robot.data.joint_names)

    def ids(suffix):
        out = []
        for p in LEG_PREFIXES:
            full = f"{p}_{suffix}"
            if full not in names:
                raise ValueError(f"joint '{full}' 없음. 사용 가능: {names}")
            out.append(names.index(full))
        return torch.tensor(out, dtype=torch.long, device=device)

    roll_ids = ids("joint1_roll")
    hip_ids = ids("joint2_pitch")
    knee_ids = ids("joint3_pitch")
    pr1_ids = ids("prismatic1")
    pr2_ids = ids("prismatic2")

    default_pos = robot.data.default_joint_pos.clone()

    # ---------------- 접촉 센서 --------------------------------------
    contact = None
    foot_ids = None
    for key in ("contact_forces",):
        if key in scene.keys():
            contact = scene[key]
            body_names = list(contact.body_names)
            foot_ids = [i for i, b in enumerate(body_names) if b.endswith("_feet")]
            foot_names = [body_names[i] for i in foot_ids]
            print(f"[INFO] 접촉 센서: {foot_names}")
            break
    if contact is None:
        print("[WARN] 접촉 센서 없음. 발 접지 통계는 건너뛴다.")

    # ---------------- CPG --------------------------------------------
    cpg = SimpleCPG(env.num_envs, device, args_cli)
    f = args_cli.omega / (2.0 * math.pi)
    margin = args_cli.h - args_cli.k1 - L0

    print()
    print("=" * 70)
    print("CPG 단독 실행")
    print("=" * 70)
    print(f"  gait {args_cli.gait}   f {f:.3f} Hz   목표 속도 {args_cli.speed} m/s")
    print(f"  h {args_cli.h}  k1 {args_cli.k1}  k2 {args_cli.k2}  "
          f"roll_nom {args_cli.roll_nom}")
    print(f"  d_step {cpg.d_step:.4f} m")
    print(f"  L0 여유 = h - k1 - L0 = {margin * 1000:+.0f} mm", end="")
    print("   <- 음수면 도달 불가능" if margin < 0 else "")
    print(f"  적분 dt {dt:.5f} s   2*alpha*dt = {2 * args_cli.alpha * dt:.2f}")
    print("=" * 70)
    print()

    # ---------------- 실행 -------------------------------------------
    n_settle = int(args_cli.settle / dt)
    n_walk = int(args_cli.duration / dt)

    log = {k: [] for k in ("vx", "vy", "height", "roll", "pitch",
                           "n_contact", "contact", "d_pos", "torque")}

    for step in range(n_settle + n_walk):
        walking = step >= n_settle

        if walking:
            cpg.step(dt)
            q_roll, q_hip, d = cpg.targets()
            targets = default_pos.clone()
            targets[:, roll_ids] = q_roll
            targets[:, hip_ids] = q_hip
            targets[:, knee_ids] = 0.0
            targets[:, pr1_ids] = d
            targets[:, pr2_ids] = d
        else:
            # 착지 대기: 기본 자세 + CPG 의 벌림만 미리 적용
            targets = default_pos.clone()
            q_roll, _, d = cpg.targets()
            targets[:, roll_ids] = q_roll
            targets[:, pr1_ids] = d
            targets[:, pr2_ids] = d

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(dt)

        if walking:
            lin = robot.data.root_lin_vel_b[0]
            log["vx"].append(lin[0].item())
            log["vy"].append(lin[1].item())
            log["height"].append(robot.data.root_pos_w[0, 2].item())

            g = robot.data.projected_gravity_b[0]
            log["roll"].append(math.asin(max(-1.0, min(1.0, g[1].item()))))
            log["pitch"].append(math.asin(max(-1.0, min(1.0, -g[0].item()))))

            log["d_pos"].append(robot.data.joint_pos[0, pr1_ids].clone())
            log["torque"].append(robot.data.applied_torque[0].clone())

            if contact is not None:
                fz = contact.data.net_forces_w[0, foot_ids, :]
                touching = (torch.norm(fz, dim=-1) > 5.0)
                log["n_contact"].append(touching.sum().item())
                log["contact"].append(touching.clone())

        if step == n_settle:
            print(f"[INFO] 착지 완료, 보행 시작. "
                  f"몸통 높이 {robot.data.root_pos_w[0, 2].item():.3f} m")

        # 넘어짐 조기 종료
        if walking and robot.data.root_pos_w[0, 2].item() < 0.35:
            print(f"\n[FAIL] {(step - n_settle) * dt:.2f} s 에 넘어짐 "
                  f"(몸통 높이 {robot.data.root_pos_w[0, 2].item():.3f} m)")
            break

    # ---------------- 요약 -------------------------------------------
    report(log, dt, names, roll_ids, hip_ids, knee_ids, pr1_ids, pr2_ids,
           foot_names if contact is not None else None)

    env.close()


def report(log, dt, names, roll_ids, hip_ids, knee_ids, pr1_ids, pr2_ids,
           foot_names):
    import statistics as st

    n = len(log["vx"])
    if n < 50:
        print("\n[결과] 데이터가 너무 짧다. 바로 넘어졌을 가능성.")
        return

    # 앞쪽 1초는 과도구간이므로 버린다
    skip = min(int(1.0 / dt), n // 4)
    sl = slice(skip, None)

    print()
    print("=" * 70)
    print(f"결과  ({(n - skip) * dt:.1f} s 분석)")
    print("=" * 70)

    vx = log["vx"][sl]
    print(f"  전진 속도       {st.mean(vx):+.3f} m/s  "
          f"(표준편차 {st.pstdev(vx):.3f}, 목표 {args_cli.speed})")
    vy = log["vy"][sl]
    print(f"  횡방향 속도     {st.mean(vy):+.3f} m/s   (0 에 가까울수록 좋음)")

    hgt = log["height"][sl]
    print(f"  몸통 높이       {st.mean(hgt):.3f} m  "
          f"(표준편차 {st.pstdev(hgt):.4f})")

    rr, pp = log["roll"][sl], log["pitch"][sl]
    print(f"  몸통 기울기     roll {math.degrees(st.mean(rr)):+.1f} deg  "
          f"pitch {math.degrees(st.mean(pp)):+.1f} deg")
    print(f"                  진폭 roll {math.degrees(st.pstdev(rr)):.1f}  "
          f"pitch {math.degrees(st.pstdev(pp)):.1f} deg")

    # 발 접지
    if log["contact"]:
        print()
        c = torch.stack(log["contact"][sl]).float()
        print(f"  동시 접지 다리 수 평균  {c.sum(dim=1).mean().item():.2f}  "
              f"(tripod 기대값 3.0)")
        print("  다리별 접지 비율 (duty, 기대값 0.5)")
        for i, nm in enumerate(foot_names):
            frac = c[:, i].mean().item()
            flag = "  <- 거의 안 닿음" if frac < 0.15 else ""
            print(f"    {nm:<14}{frac:6.3f}{flag}")

    # prismatic
    d = torch.stack(log["d_pos"][sl])
    dv = (d[1:] - d[:-1]).abs().max().item() / dt
    print()
    print(f"  prismatic 변위  {d.min().item():.4f} ~ {d.max().item():.4f} m  "
          f"(허용 0 ~ {STROKE})")
    print(f"  prismatic 속도  최대 {dv:.4f} m/s  "
          f"(한계 {PRISMATIC_VEL_LIMIT}, {dv / PRISMATIC_VEL_LIMIT * 100:.0f}%)"
          f"{'  <- 초과' if dv > PRISMATIC_VEL_LIMIT else ''}")

    # 토크
    tq = torch.stack(log["torque"][sl])
    print()
    print("  관절 토크 (연속 판정선 대비)")
    print(f"    {'group':<12}{'RMS':>9}{'peak':>9}{'판정선':>9}{'RMS%':>8}")
    groups = [("roll", roll_ids), ("hip", hip_ids),
              ("knee", knee_ids), ("prismatic", torch.cat([pr1_ids, pr2_ids]))]
    for label, idx in groups:
        sub = tq[:, idx]
        rms = sub.pow(2).mean().sqrt().item()
        peak = sub.abs().max().item()
        lim = TORQUE_LIMITS[label]
        print(f"    {label:<12}{rms:9.1f}{peak:9.1f}{lim:9.1f}"
              f"{rms / lim * 100:7.0f}%"
              f"{'  초과' if rms > lim else ''}")

    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
