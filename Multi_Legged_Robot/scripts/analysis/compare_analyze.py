#!/usr/bin/env python3
"""compare_probe.py 가 만든 CSV 로 RL only / CPG only / CPG + RL 을 한 번에 비교한다.

놓을 위치:  scripts/analysis/compare_analyze.py
Isaac Lab 의존성 없음.  pip install numpy pandas matplotlib

사용 예 (프로젝트 루트에서)
    python scripts/analysis/compare_analyze.py \
        --csv eval_results/raw/rl_bumps.csv     --name "RL only" \
        --csv eval_results/raw/cpg_bumps.csv    --name "CPG only" \
        --csv eval_results/raw/cpg_rl_bumps.csv --name "CPG + RL" \
        --outdir eval_results/compare_bumps

산출물 (--outdir)
    summary.md          발표 표와 같은 행 구성, 중앙값 [Q1–Q3]
    summary_slide.csv   중앙값만 — PPT 표에 바로 붙이는 용도
    windows.csv         창 하나하나의 값 (IQR 이 넓을 때 원인 추적)
    fig1_bars.png       주 지표 막대 + IQR
    fig2_footfall.png   조건별 대표 창의 발자국 도표 (막대 = 유각)
    fig3_phase.png      다리별 위상 vs 이상 걸음새
    fig4_tilt_cdf.png   틸트각 누적분포

지표 정의
    속도 달성률     창 평균 vx_b / 명령 vx × 100
    틸트각 P95      투영 중력에서 구한 몸통 기울기의 상위 5% 경계
    각속도 RMS      sqrt(mean(wx² + wy²))
    σ_z             z_world 의 창 내 표준편차. 체고(창 평균 z_world − ground_z) 로 정규화
    조기 종료율     time_out 이 아닌 done 의 비율 (전체 리셋 대비)
    평균 접지 다리  n_contact 의 창 평균
    보행 주파수     여섯 발 접지 신호 PSD 평균의 0.2~3 Hz 봉우리
    위상 오차       R1 착지 기준 각 다리 착지 시각 / 주기 → 이상 걸음새와의 원형 거리.
                    최대값(가장 어긋난 다리)을 대표값으로 쓴다. RL 은 tripod 기준으로 잰다.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#7B7B7B", "#2E5EAA", "#DC143C", "#EDAE49", "#00798C"]

LEGS = ["R1", "R2", "R3", "L1", "L2", "L3"]
GAIT_PHASES = {
    "tripod": (0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
    "ripple": (0.0, 2 / 6, 4 / 6, 3 / 6, 5 / 6, 1 / 6),
    "wave":   (2 / 6, 1 / 6, 0.0, 5 / 6, 4 / 6, 3 / 6),
}


# ====================================================================== #
# 로딩 / 창 나누기
# ====================================================================== #

def load_csv(path):
    meta = {}
    with open(path) as f:
        first = f.readline()
    skip = 0
    if first.startswith("#"):
        skip = 1
        for tok in first[1:].strip().split(","):
            if "=" in tok:
                k, v = tok.split("=", 1)
                meta[k.strip()] = v.strip()
    df = pd.read_csv(path, skiprows=skip)
    dt = float(meta.get("dt", 0.0)) or float(np.median(np.diff(np.sort(df["t"].unique()))))
    return df, dt, meta


def split_windows(df, dt, warmup_s, window_s):
    """env → 에피소드(ep_step 되감김) → warmup 제거 → 정확히 window_s 길이로 자름."""
    n = int(round(window_s / dt))
    warm = int(round(warmup_s / dt))
    wins, n_short = [], 0
    for env_id, ge in df.groupby("env_id", sort=True):
        ge = ge.sort_values("t").reset_index(drop=True)
        seg = (ge["ep_step"].diff() < 0).cumsum()
        for sid, g in ge.groupby(seg, sort=True):
            g = g[g["ep_step"] >= warm].reset_index(drop=True)
            if len(g) < n:
                n_short += 1
                continue
            k = 0
            while (k + 1) * n <= len(g):
                w = g.iloc[k * n:(k + 1) * n].copy()
                w["win_key"] = f"{int(env_id)}:{int(sid)}:{k}"
                wins.append(w)
                k += 1
    return wins, n_short


def audit(df):
    done = df["done"].to_numpy() > 0.5
    to = df["time_out"].to_numpy() > 0.5
    n_early = int((done & ~to).sum())
    n_timeout = int((done & to).sum())
    n_reset = n_early + n_timeout
    return dict(n_early_term=n_early, n_timeout=n_timeout, n_reset=n_reset,
                early_term_pct=(100.0 * n_early / n_reset) if n_reset else np.nan)


# ====================================================================== #
# 걸음새 지표
# ====================================================================== #

def gait_frequency(contacts, dt, fmin=0.2, fmax=3.0):
    """(T, 6) 접지 0/1 → 여섯 다리 PSD 평균의 봉우리 [Hz]."""
    T = contacts.shape[0]
    if T < 64:
        return np.nan
    win = np.hanning(T)
    nfft = 16 * T                    # 제로패딩: 12 s 창의 0.083 Hz 눈금을 0.005 Hz 로
    freqs = np.fft.rfftfreq(nfft, d=dt)
    psd = np.zeros_like(freqs)
    used = 0
    for i in range(contacts.shape[1]):
        x = contacts[:, i] - contacts[:, i].mean()
        if x.std() < 1e-6:          # 한 번도 안 들었거나 한 번도 안 닿은 다리
            continue
        psd += np.abs(np.fft.rfft(x * win, n=nfft)) ** 2
        used += 1
    if used == 0:
        return np.nan
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(freqs[band][np.argmax(psd[band])])


def touchdowns(c, dt, min_swing_s=0.06):
    """접지 신호에서 착지 시각 인덱스. 짧은 튐(<min_swing_s)은 무시."""
    c = (c > 0.5).astype(int)
    d = np.diff(c)
    ups = np.where(d == 1)[0] + 1
    downs = np.where(d == -1)[0] + 1
    if len(ups) == 0:
        return np.array([], dtype=int)
    keep = []
    min_n = int(round(min_swing_s / dt))
    for u in ups:
        prev_down = downs[downs < u]
        if len(prev_down) == 0 or (u - prev_down[-1]) >= min_n:
            keep.append(u)
    return np.array(keep, dtype=int)


def phase_error(contacts, dt, period, ideal):
    """R1 착지마다 각 다리의 다음 착지까지 걸린 시간 / 주기 → 위상.
    반환: (다리별 위상 중앙값 [6], 다리별 원형 오차 [6])"""
    if not np.isfinite(period) or period <= 0:
        return np.full(6, np.nan), np.full(6, np.nan)
    td = [touchdowns(contacts[:, i], dt) for i in range(6)]
    # 기준 다리: 착지가 있는 첫 다리 (R1 이 한 번도 안 들면 다음 다리로)
    ref_i = next((i for i in range(6) if len(td[i]) >= 2), None)
    if ref_i is None:
        return np.full(6, np.nan), np.full(6, np.nan)
    ref = td[ref_i]
    ideal = (np.asarray(ideal) - ideal[ref_i]) % 1.0      # 기준 다리에 맞춰 이상 위상 재정렬
    n_per = period / dt
    phases = np.full(6, np.nan)
    for i in range(6):
        vals = []
        for t0 in ref:
            nxt = td[i][(td[i] >= t0) & (td[i] < t0 + n_per)]
            if len(nxt):
                vals.append((nxt[0] - t0) / n_per)
        if vals:
            phases[i] = np.median(vals) % 1.0
    d = np.abs(phases - ideal)
    err = np.minimum(d, 1.0 - d)
    return phases, err


def cpg_phase_error(w, ideal):
    """발진기 상태로 잰 위상 오차 (있을 때만). R1 기준."""
    if "cpg_x1" not in w.columns or not np.isfinite(w["cpg_x1"]).any():
        return np.nan
    x = w[[f"cpg_x{i+1}" for i in range(6)]].to_numpy()
    y = w[[f"cpg_y{i+1}" for i in range(6)]].to_numpy()
    ph = np.arctan2(y, x) / (2 * np.pi)
    rel = (ph - ph[:, [0]]) % 1.0
    d = np.abs(rel - np.asarray(ideal)[None, :])
    err = np.minimum(d, 1.0 - d)
    return float(np.nanmax(np.nanmedian(err, axis=0)))


# ====================================================================== #
# 창 하나당 지표
# ====================================================================== #

def window_metrics(w, dt, ideal, body_height=None):
    m = {}
    z = w["z_world"].to_numpy(float)
    gz = w["ground_z"].to_numpy(float)
    zrel = z - gz
    bh = body_height if body_height else float(np.nanmean(zrel))
    m["body_height_m"] = bh

    vx = w["vx_b"].to_numpy(float)
    cmd = float(np.nanmean(w["cmd_vx"]))
    m["cmd_vx"] = cmd
    m["vx_mean"] = float(vx.mean())
    m["speed_pct"] = 100.0 * vx.mean() / cmd if cmd > 1e-6 else np.nan
    m["vx_std"] = float(vx.std())

    m["tilt_p95_deg"] = float(np.percentile(w["tilt_deg"], 95))
    m["tilt_rms_deg"] = float(np.sqrt(np.mean(w["tilt_deg"] ** 2)))
    m["angvel_xy_rms"] = float(np.sqrt(np.mean(w["wx"] ** 2 + w["wy"] ** 2)))
    m["yaw_rate_mean"] = float(w["wz"].mean())          # 요 드리프트 (부호 있음)

    m["sigma_z_mm"] = float(z.std()) * 1000.0
    m["sigma_z_pct"] = 100.0 * z.std() / bh
    m["sigma_zrel_mm"] = float(np.nanstd(zrel)) * 1000.0
    m["terrain_sigma_mm"] = float(np.nanstd(gz)) * 1000.0

    c = w[[f"c_{l}" for l in LEGS]].to_numpy(float)
    m["n_contact_mean"] = float(c.sum(axis=1).mean())
    m["n_contact_min"] = float(c.sum(axis=1).min())
    duty = c.mean(axis=0)
    for i, l in enumerate(LEGS):
        m[f"duty_{l}"] = float(duty[i])
    m["duty_mean"] = float(duty.mean())
    m["n_legs_never_lift"] = int((duty > 0.98).sum())

    f = gait_frequency(c, dt)
    m["gait_freq_hz"] = f
    phases, err = phase_error(c, dt, 1.0 / f if np.isfinite(f) and f > 0 else np.nan, ideal)
    for i, l in enumerate(LEGS):
        m[f"phase_{l}"] = float(phases[i])
    m["phase_err_max"] = float(np.nanmax(err)) if np.isfinite(err).any() else np.nan
    m["phase_err_mean"] = float(np.nanmean(err)) if np.isfinite(err).any() else np.nan
    m["cpg_phase_err_max"] = cpg_phase_error(w, ideal)
    return m


def aggregate(per_win):
    pw = pd.DataFrame(per_win)
    agg = {}
    for c in pw.columns:
        v = pw[c].to_numpy(float)
        if not np.isfinite(v).any():
            agg[c] = agg[c + "_q1"] = agg[c + "_q3"] = np.nan
            continue
        agg[c] = float(np.nanmedian(v))
        agg[c + "_q1"] = float(np.nanpercentile(v, 25))
        agg[c + "_q3"] = float(np.nanpercentile(v, 75))
    agg["n_windows"] = int(len(pw))
    return agg


# ====================================================================== #
# 표
# ====================================================================== #

ROWS = [
    ("speed_pct",        "속도 달성률 [%]",            1),
    ("tilt_p95_deg",     "틸트각 P95 [deg]",           2),
    ("angvel_xy_rms",    "각속도 RMS [rad/s]",         3),
    ("sigma_z_pct",      "σ_z [체고 %]",               2),
    ("early_term_pct",   "조기 종료율 [%]",            1),
    ("n_contact_mean",   "평균 접지 다리 수",          2),
    ("gait_freq_hz",     "보행 주파수 [Hz]",           3),
    ("phase_err_max",    "위상 오차 최대 [주기]",      3),
]
AUX = [
    ("vx_mean",          "평균 전진속도 [m/s]",        3),
    ("yaw_rate_mean",    "요 각속도 평균 [rad/s]",     3),
    ("tilt_rms_deg",     "틸트각 RMS [deg]",           2),
    ("sigma_z_mm",       "σ_z [mm]",                   2),
    ("sigma_zrel_mm",    "σ_z 발밑 대비 [mm]",         2),
    ("body_height_m",    "기준 체고 [m]",              3),
    ("duty_mean",        "듀티 평균",                  3),
    ("n_legs_never_lift","한 번도 안 든 다리 수",      1),
    ("n_contact_min",    "최소 접지 다리 수",          1),
    ("phase_err_mean",   "위상 오차 평균 [주기]",      3),
    ("cpg_phase_err_max","발진기 위상 오차 [주기]",    3),
    ("terrain_sigma_mm", "지형 높이 std [mm]",         2),
    ("n_windows",        "유효 창 수",                 0),
    ("n_reset",          "리셋 횟수",                  0),
    ("n_early_term",     "조기 종료 횟수",             0),
]
NO_IQR = {"early_term_pct", "n_windows", "n_reset", "n_early_term"}


def fmt(v, d):
    return "—" if not np.isfinite(v) else f"{v:.{d}f}"


def md_table(rows, spec, with_iqr=True):
    head = "| 지표 | " + " | ".join(r["name"] for r in rows) + " |"
    sep = "|---|" + "|".join(["---"] * len(rows)) + "|"
    lines = [head, sep]
    for key, label, d in spec:
        cells = []
        for r in rows:
            v = r.get(key, np.nan)
            if with_iqr and key not in NO_IQR and np.isfinite(r.get(key + "_q1", np.nan)):
                cells.append(f"{fmt(v, d)} [{fmt(r[key + '_q1'], d)}–{fmt(r[key + '_q3'], d)}]")
            else:
                cells.append(fmt(v, d))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdicts(rows):
    out = []
    for r in rows:
        p = r.get("early_term_pct", np.nan)
        if np.isfinite(p) and p > 10:
            out.append(f"🔴 **{r['name']}** 조기 종료율 {p:.1f}% — 넘어진 구간이 통계에서 빠졌다. 안정성 수치가 실제보다 좋게 보인다.")
        elif np.isfinite(p) and p > 3:
            out.append(f"🟡 **{r['name']}** 조기 종료율 {p:.1f}% — 표에 같이 보고할 것.")
        if r.get("n_legs_never_lift", 0) >= 1:
            out.append(f"🟡 **{r['name']}** 한 번도 들지 않은 다리가 {int(r['n_legs_never_lift'])}개 — 걸음새가 성립하지 않는다. 위상 오차는 남은 다리로만 계산됐다.")
        if abs(r.get("yaw_rate_mean", 0.0)) > 0.05:
            out.append(f"🟡 **{r['name']}** 요 각속도 평균 {r['yaw_rate_mean']:+.3f} rad/s — 직진 명령에 회전이 섞인다.")
    ts = [r.get("terrain_sigma_mm", np.nan) for r in rows]
    ts = [t for t in ts if np.isfinite(t) and t > 0.1]
    if len(ts) >= 2 and max(ts) / min(ts) > 1.1:
        out.append(f"🟡 지형 높이 std 가 조건마다 다르다 ({min(ts):.2f}~{max(ts):.2f} mm). 같은 지형을 밟았는지 확인할 것 (--seed, --terrain_seed 동일?).")
    cmds = [r.get("cmd_vx", np.nan) for r in rows]
    if len(set(round(c, 3) for c in cmds if np.isfinite(c))) > 1:
        out.append(f"🔴 명령 속도가 조건마다 다르다: {cmds}. 속도 달성률 비교가 성립하지 않는다.")
    out.append("ℹ️ IQR 은 같은 정책을 여러 창으로 쪼갠 변동이다. 조건 간 차이를 주장하려면 학습 시드 여러 개가 필요하다.")
    return "\n".join(f"- {s}" for s in out)


# ====================================================================== #
# 그림
# ====================================================================== #

def fig_bars(rows, outdir):
    keys = [("speed_pct", "Speed achieved [%]"), ("tilt_p95_deg", "Tilt P95 [deg]"),
            ("angvel_xy_rms", "Ang. vel. RMS [rad/s]"), ("sigma_z_pct", "σ_z [% body height]"),
            ("n_contact_mean", "Feet in contact (mean)"), ("gait_freq_hz", "Gait frequency [Hz]")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.8))
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    for ax, (k, label) in zip(axes, keys):
        med = np.array([r.get(k, np.nan) for r in rows])
        lo = med - np.array([r.get(k + "_q1", np.nan) for r in rows])
        hi = np.array([r.get(k + "_q3", np.nan) for r in rows]) - med
        ax.bar(x, np.nan_to_num(med), 0.6, color=[r["_color"] for r in rows], alpha=0.9)
        ax.errorbar(x, med, yerr=[np.abs(np.nan_to_num(lo)), np.abs(np.nan_to_num(hi))],
                    fmt="none", ecolor="k", capsize=4, lw=1.1)
        for i, v in enumerate(med):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_title(label, loc="left", fontsize=9.5, fontweight="bold")
        ax.set_ylim(bottom=0)
    fig.suptitle("Median with interquartile range", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_bars.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_footfall(reps, outdir, seconds=8.0):
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, w, dt, color) in zip(axes, reps):
        t = w["t"].to_numpy() - w["t"].iloc[0]
        keep = t <= seconds
        for i, l in enumerate(LEGS):
            c = w[f"c_{l}"].to_numpy()[keep]
            swing = c < 0.5
            tt = t[keep]
            # 연속 유각 구간을 막대로
            j = 0
            while j < len(swing):
                if swing[j]:
                    k = j
                    while k < len(swing) and swing[k]:
                        k += 1
                    ax.barh(5 - i, tt[k - 1] - tt[j] + dt, left=tt[j], height=0.6, color=color)
                    j = k
                else:
                    j += 1
        ax.set_yticks(range(6)); ax.set_yticklabels(LEGS[::-1])
        ax.set_xlabel("time [s]"); ax.set_xlim(0, seconds)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.grid(False)
    fig.suptitle("Footfall (bars = swing, foot in air)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_footfall.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_phase(rows, ideal_by_row, outdir):
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(6)
    for ax, r, ideal in zip(axes, rows, ideal_by_row):
        meas = np.array([r.get(f"phase_{l}", np.nan) for l in LEGS])
        ref_i = next((i for i in range(6) if np.isfinite(meas[i]) and abs(meas[i]) < 1e-9), 0)
        ideal = (np.asarray(ideal) - ideal[ref_i]) % 1.0
        ax.bar(x - 0.18, ideal, 0.36, color="#BFBFBF", label="ideal")
        ax.bar(x + 0.18, meas, 0.36, color=r["_color"], label="measured")
        ax.set_xticks(x); ax.set_xticklabels(LEGS)
        ax.set_ylim(0, 1); ax.set_title(r["name"], loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("phase lead over R1 [cycles]")
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_phase.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_tilt_cdf(datasets, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, wins, color in datasets:
        v = np.sort(np.concatenate([w["tilt_deg"].to_numpy() for w in wins]))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, lw=2, color=color, label=name)
    ax.set_xlabel("tilt [deg]"); ax.set_ylabel("cumulative share of time [%]")
    ax.set_ylim(0, 100); ax.legend(frameon=False, loc="lower right")
    ax.set_title("Body tilt CDF", loc="left", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_tilt_cdf.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ====================================================================== #

def main():
    ap = argparse.ArgumentParser(description="RL / CPG / CPG+RL 비교 분석")
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--name", action="append", default=None)
    ap.add_argument("--outdir", default="eval_results/compare")
    ap.add_argument("--warmup", type=float, default=2.0, help="리셋 직후 제외 [s]")
    ap.add_argument("--window", type=float, default=12.0, help="창 길이 [s]")
    ap.add_argument("--body-height", type=float, default=None, help="σ_z 정규화 체고 고정값 [m] (기본: 창 평균 실측)")
    args = ap.parse_args()

    names = args.name or [os.path.splitext(os.path.basename(p))[0] for p in args.csv]
    if len(names) != len(args.csv):
        raise SystemExit("--csv 와 --name 개수가 다르다.")
    os.makedirs(args.outdir, exist_ok=True)

    rows, datasets, reps, ideals, win_dump = [], [], [], [], []
    for gi, (path, name) in enumerate(zip(args.csv, names)):
        df, dt, meta = load_csv(path)
        gait = meta.get("gait", "none")
        ideal = GAIT_PHASES.get(gait, GAIT_PHASES["tripod"])   # RL 은 tripod 기준으로 잰다
        wins, n_short = split_windows(df, dt, args.warmup, args.window)
        if not wins:
            raise SystemExit(f"{name}: 유효 창이 없다. --warmup / --window 를 줄여볼 것.")
        per = [window_metrics(w, dt, ideal, args.body_height) for w in wins]
        agg = aggregate(per)
        agg.update(audit(df))
        agg["n_too_short"] = n_short
        color = PALETTE[gi % len(PALETTE)]
        agg.update(name=name, _color=color, _gait=gait, _mode=meta.get("mode", "?"),
                   _terrain=meta.get("terrain", "?"), _ckpt=meta.get("checkpoint", ""))
        rows.append(agg)
        datasets.append((name, wins, color))
        ideals.append(ideal)
        # 대표 창: 전진속도가 중앙값에 가장 가까운 창
        vxs = np.array([m["vx_mean"] for m in per])
        reps.append((name, wins[int(np.argmin(np.abs(vxs - np.median(vxs))))], dt, color))
        for w, m in zip(wins, per):
            win_dump.append(dict(_name=name, _win=w["win_key"].iloc[0], **m))
        print(f"[{name}] mode={agg['_mode']} terrain={agg['_terrain']} gait={gait} "
              f"창 {len(wins)}개 (짧아서 제외 {n_short}), 조기종료 {agg['n_early_term']}/{agg['n_reset']}")

    # ---------------- 표 ----------------
    md = ["# RL only / CPG only / CPG + RL 비교\n",
          f"warmup {args.warmup:g}s · 창 {args.window:g}s · 값은 **중앙값 [Q1–Q3]**\n",
          "## 조건", "| 조건 | mode | terrain | gait | checkpoint |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['_mode']} | {r['_terrain']} | {r['_gait']} | {os.path.basename(r['_ckpt']) or '—'} |")
    md += ["\n## 0. 자동 판정 (먼저 읽을 것)\n", verdicts(rows),
           "\n## 1. 발표 표 (슬라이드 행 구성)\n", md_table(rows, ROWS),
           "\n## 2. 보조 지표\n", md_table(rows, AUX)]
    text = "\n".join(md)
    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write(text)

    slide = pd.DataFrame({r["name"]: [fmt(r.get(k, np.nan), d) for k, _, d in ROWS] for r in rows},
                         index=[lbl for _, lbl, _ in ROWS])
    slide.to_csv(os.path.join(args.outdir, "summary_slide.csv"), encoding="utf-8-sig")
    pd.DataFrame(win_dump).to_csv(os.path.join(args.outdir, "windows.csv"), index=False, encoding="utf-8-sig")

    made = [fig_bars(rows, args.outdir), fig_footfall(reps, args.outdir),
            fig_phase(rows, ideals, args.outdir), fig_tilt_cdf(datasets, args.outdir)]
    print("\n" + text)
    print("\n생성:")
    for p in [os.path.join(args.outdir, "summary.md"), os.path.join(args.outdir, "summary_slide.csv"),
              os.path.join(args.outdir, "windows.csv")] + made:
        print("  " + p)


if __name__ == "__main__":
    main()
