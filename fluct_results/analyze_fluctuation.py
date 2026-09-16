#!/usr/bin/env python3
# analyze_fluctuation.py
"""fluctuation_probe.py 가 만든 CSV들을 읽어 지표 표와 그래프를 만든다.

Isaac Lab 의존성이 전혀 없습니다. 일반 파이썬 환경에서 돌리세요.

    pip install pandas numpy matplotlib

===========================================================================
사용법
===========================================================================
    # 두 로봇 비교
    python analyze_fluctuation.py \
        --csv go2_rough_vx0.3.csv   --name "Go2"   --body-height 0.30 \
        --csv hugo_rough_vx0.3.csv  --name "Hugo"  --body-height 1.05 \
        --outdir results_rough

    # 조건이 여러 개면 그냥 계속 붙이면 됩니다
    python analyze_fluctuation.py \
        --csv go2_flat.csv   --name "Go2 flat"   --body-height 0.30 \
        --csv go2_rough.csv  --name "Go2 rough"  --body-height 0.30 \
        --csv hugo_flat.csv  --name "Hugo flat"  --body-height 1.05 \
        --csv hugo_rough.csv --name "Hugo rough" --body-height 1.05 \
        --outdir results_all

산출물 (--outdir 아래):
    summary.csv          지표 표 (엑셀/한글로 바로 붙여넣기)
    summary.md           마크다운 표 (보고서용)
    fig1_timeseries.png  z 변동 시계열 비교
    fig2_distribution.png z 변동 분포 (박스플롯)
    fig3_tilt.png        틸트각 시계열 + 분포
    fig4_bars.png        주요 지표 막대 비교
    fig5_psd.png         주파수 스펙트럼 (보행 주파수 확인)

===========================================================================
지표 정의
===========================================================================
z_detrend : z_world 에서 이동평균(기본 1.0초 창)을 뺀 나머지.
            지형을 따라 완만히 오르내리는 성분을 제거하고, 보행에 의한
            진동만 남긴 값. ★ 주 지표.
            평지에서는 이동평균이 거의 상수라 z_world 표준편차와 비슷해지고,
            험지에서는 지형 추종 성분이 제거되어 공정한 비교가 된다.

tilt      : 몸체 z축과 연직선 사이 각도. roll/pitch 를 합친 총 기울기이며
            회전 순서(ZYX vs XYZ)에 무관해서 로봇 간 비교에 안전하다.

정규화    : Go2(체고 ~0.30m)와 Hugo(~1.05m)는 스케일이 달라 절대 mm 비교가
            불공정하다. 체고 대비 % 를 반드시 함께 본다.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 그래프 라벨은 영문 고정 (한글 폰트 누락으로 □□□ 나오는 사고 방지).
# 한글이 꼭 필요하면 아래 두 줄의 주석을 풀고 폰트를 설치하세요.
#   sudo apt install fonts-nanum && rm -rf ~/.cache/matplotlib
# matplotlib.rcParams["font.family"] = "NanumGothic"
# matplotlib.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = ["#2E5EAA", "#D1495B", "#00798C", "#EDAE49", "#7B6D8D", "#66A182"]


# ====================================================================== #
# 로딩
# ====================================================================== #

def load_csv(path: str) -> tuple[pd.DataFrame, float]:
    """probe CSV 를 읽고 (DataFrame, dt) 를 반환."""
    dt = None
    with open(path) as f:
        first = f.readline()
    skip = 0
    if first.startswith("#"):
        skip = 1
        for tok in first.strip().split(","):
            if "dt=" in tok:
                dt = float(tok.split("dt=")[1])
    df = pd.read_csv(path, skiprows=skip)
    if dt is None:
        t = np.sort(df["t"].unique())
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    return df, dt


def preprocess(df: pd.DataFrame, dt: float, warmup_s: float,
               detrend_window_s: float) -> pd.DataFrame:
    """에피소드 단위로 잘라내고, 각 구간 안에서 z_detrend 를 만든다.

    ★ 왜 에피소드 단위인가
      측정 중 로봇은 20초마다 리셋된다(패치 밖으로 나가는 걸 막기 위해).
      리셋 지점에서 z 가 불연속으로 튀는데, 이걸 무시하고 전체를 이어붙여
      이동평균을 걸면 이음매마다 가짜 스파이크가 생겨 RMS 가 부풀려진다.
      그래서 ep_step 이 되감기는 지점을 기준으로 구간을 나누고,
      구간 안에서만 detrend 한다.
    """
    win = max(int(round(detrend_window_s / dt)), 3)
    warm = int(round(warmup_s / dt))
    half = win // 2
    min_len = win * 2 + 2 * half

    out = []
    n_seg_total = n_seg_kept = 0

    for env_id, g_env in df.groupby("env_id", sort=True):
        g_env = g_env.sort_values("t").reset_index(drop=True)

        # ep_step 이 감소하는 지점 = 에피소드 리셋
        if "ep_step" in g_env.columns:
            seg_id = (g_env["ep_step"].diff() < 0).cumsum()
        else:
            seg_id = pd.Series(0, index=g_env.index)

        for sid, g in g_env.groupby(seg_id, sort=True):
            n_seg_total += 1
            g = g.reset_index(drop=True)

            # 리셋 직후 구간 제거 (낙하 / 자세 정렬)
            if "ep_step" in g.columns:
                g = g[g["ep_step"] >= warm]
            else:
                g = g.iloc[warm:]
            if len(g) < min_len:
                continue

            # 이동평균 제거 = 지형 추종 성분 제거
            base = g["z_world"].rolling(win, center=True, min_periods=1).mean()
            g = g.assign(z_detrend=g["z_world"] - base)
            if "z_rel" in g.columns:
                base_r = g["z_rel"].rolling(win, center=True, min_periods=1).mean()
                g = g.assign(z_rel_detrend=g["z_rel"] - base_r)

            # 이동평균 창 가장자리는 왜곡되므로 잘라낸다
            g = g.iloc[half:-half]
            if len(g) < win:
                continue

            g = g.assign(seg_key=f"{int(env_id)}:{int(sid)}")
            out.append(g)
            n_seg_kept += 1

    if not out:
        raise RuntimeError(
            "전처리 후 남은 데이터가 없습니다.\n"
            f"  에피소드 구간 {n_seg_total}개를 찾았지만 모두 너무 짧습니다.\n"
            "  --warmup 을 줄이거나(기본 2.0초), --detrend-window 를 줄여보세요."
        )
    print(f"[seg] 에피소드 구간 {n_seg_kept}/{n_seg_total}개 사용")
    return pd.concat(out, ignore_index=True)


# ====================================================================== #
# 지표
# ====================================================================== #

def _follow_slope(g: pd.DataFrame) -> float:
    """z_world 를 ground_z 에 대해 1차 회귀한 기울기.

    각 구간 안에서 두 신호의 평균을 뺀 뒤 회귀한다.
    지형이 평평하면(분산 0) 정의되지 않으므로 nan 을 돌려준다.
    """
    if "ground_z" not in g.columns:
        return float("nan")
    x = g["ground_z"].to_numpy(dtype=float)
    y = g["z_world"].to_numpy(dtype=float)
    if len(x) < 10 or np.std(x) < 1e-4:      # 평지: 기울기 정의 불가
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    return float(np.polyfit(x, y, 1)[0])


def _dt_of(df: pd.DataFrame) -> float:
    t = np.sort(df["t"].unique())
    return float(np.median(np.diff(t))) if len(t) > 1 else 0.02


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def metrics(df: pd.DataFrame, body_height: float | None) -> dict:
    """env 별로 지표를 낸 뒤 env 간 평균 ± 표준편차로 집계."""
    group_key = "seg_key" if "seg_key" in df.columns else "env_id"
    per_env = []
    for _, g in df.groupby(group_key):
        m = {
            "z_detrend_rms_mm": rms(g["z_detrend"]) * 1000.0,
            "z_detrend_p2p_mm": (g["z_detrend"].max() - g["z_detrend"].min()) * 1000.0,
            "z_world_std_mm": float(g["z_world"].std()) * 1000.0,
            "tilt_rms_deg": rms(g["tilt_deg"]),
            "tilt_p95_deg": float(np.percentile(g["tilt_deg"], 95)),
            "tilt_max_deg": float(g["tilt_deg"].max()),
            "roll_rms_deg": rms(g["roll_deg"]),
            "pitch_rms_deg": rms(g["pitch_deg"]),
            "ang_vel_xy_rms": rms(np.hypot(g["wx"], g["wy"])),
            "vz_rms": rms(g["vz_w"]),
            "vx_mean": float(g["vx_b"].mean()),
            "vx_track_err": float((g["vx_b"] - g["cmd_vx"]).abs().mean()),
            # 지형 검증용: 두 로봇의 이 값이 같아야 같은 지형을 밟은 것
            "terrain_range_mm": (float(g["ground_z"].max() - g["ground_z"].min()) * 1000.0
                                 if "ground_z" in g.columns else float("nan")),
            "terrain_std_mm": (float(g["ground_z"].std()) * 1000.0
                               if "ground_z" in g.columns else float("nan")),
            # ★ 지형 추종 계수: z_world 를 ground_z 로 회귀한 기울기.
            #   0 = 지형이 오르내려도 몸체는 수평 유지 (완벽한 anti-fluctuation)
            #   1 = 몸체가 지형을 그대로 따라 오르내림
            "terrain_follow": _follow_slope(g),
        }
        if "z_rel_detrend" in g.columns:
            m["z_rel_detrend_rms_mm"] = rms(g["z_rel_detrend"]) * 1000.0
        per_env.append(m)

    pe = pd.DataFrame(per_env)
    agg = {}
    for c in pe.columns:
        agg[c] = float(pe[c].mean())
        agg[c + "_sd"] = float(pe[c].std())

    if body_height:
        agg["z_detrend_rms_pct"] = agg["z_detrend_rms_mm"] / (body_height * 1000.0) * 100.0
        agg["z_detrend_p2p_pct"] = agg["z_detrend_p2p_mm"] / (body_height * 1000.0) * 100.0

    agg["n_envs"] = int(df["env_id"].nunique())
    agg["n_segments"] = int(pe.shape[0])
    agg["duration_s"] = float(len(df) * 0.0 + df.shape[0] / max(df["env_id"].nunique(), 1)
                              * _dt_of(df))
    return agg


def gait_psd(df: pd.DataFrame, dt: float, col: str = "z_detrend"):
    """env 별 PSD 를 평균낸다. 반환 (freqs, psd, 주파수 피크)."""
    key = "seg_key" if "seg_key" in df.columns else "env_id"
    specs, freqs = [], None
    for _, g in df.groupby(key):
        x = g[col].to_numpy()
        x = x - x.mean()
        n = len(x)
        if n < 64:
            continue
        w = np.hanning(n)
        sp = np.abs(np.fft.rfft(x * w)) ** 2 / n
        f = np.fft.rfftfreq(n, d=dt)
        if freqs is None or len(f) < len(freqs):
            freqs = f
        specs.append((f, sp))
    if not specs:
        return None, None, None
    # 길이 맞추기
    L = min(len(s[1]) for s in specs)
    freqs = specs[0][0][:L]
    psd = np.mean([s[1][:L] for s in specs], axis=0)
    band = freqs > 0.3
    peak = float(freqs[band][np.argmax(psd[band])]) if band.any() else None
    return freqs, psd, peak


# ====================================================================== #
# 그래프
# ====================================================================== #

def _boxplot(ax, data, names):
    """matplotlib 3.9에서 labels -> tick_labels 로 이름이 바뀌어 양쪽 다 지원."""
    kw = dict(showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1.4))
    try:
        return ax.boxplot(data, tick_labels=names, **kw)
    except TypeError:
        return ax.boxplot(data, labels=names, **kw)


def fig_timeseries(datasets, outdir, seconds=8.0):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 2.4 * len(datasets)),
                             sharex=True, squeeze=False)
    for i, (name, df, dt, _, color) in enumerate(datasets):
        ax = axes[i][0]
        key = "seg_key" if "seg_key" in df.columns else "env_id"
        eid = sorted(df[key].unique())[0]
        g = df[df[key] == eid].sort_values("t")
        t0 = g["t"].iloc[0]
        g = g[g["t"] - t0 <= seconds]
        y = g["z_detrend"] * 1000.0
        ax.plot(g["t"] - t0, y, color=color, lw=1.3)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        r = rms(y)
        ax.axhspan(-r, r, color=color, alpha=0.12)
        ax.set_ylabel("z dev [mm]")
        ax.set_title(f"{name}   (RMS = {r:.2f} mm, shaded = ±1 RMS)",
                     loc="left", fontsize=10, fontweight="bold")
    axes[-1][0].set_xlabel("time [s]")
    fig.suptitle("Detrended vertical body motion (representative env)",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_timeseries.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_distribution(datasets, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    data = [d[1]["z_detrend"] * 1000.0 for d in datasets]
    names = [d[0] for d in datasets]
    colors = [d[4] for d in datasets]

    bp = _boxplot(axes[0], data, names)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    axes[0].set_ylabel("z deviation [mm]")
    axes[0].set_title("Vertical deviation distribution", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=15)

    for (name, df, _, _, color) in datasets:
        axes[1].hist(df["z_detrend"] * 1000.0, bins=80, density=True,
                     histtype="step", lw=1.6, color=color, label=name)
    axes[1].set_xlabel("z deviation [mm]")
    axes[1].set_ylabel("density")
    axes[1].set_title("Histogram", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    p = os.path.join(outdir, "fig2_distribution.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_tilt(datasets, outdir, seconds=8.0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (name, df, dt, _, color) in datasets:
        key = "seg_key" if "seg_key" in df.columns else "env_id"
        eid = sorted(df[key].unique())[0]
        g = df[df[key] == eid].sort_values("t")
        t0 = g["t"].iloc[0]
        g = g[g["t"] - t0 <= seconds]
        axes[0].plot(g["t"] - t0, g["tilt_deg"], lw=1.2, color=color, label=name)
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("tilt [deg]")
    axes[0].set_title("Body tilt from vertical", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    data = [d[1]["tilt_deg"] for d in datasets]
    names = [d[0] for d in datasets]
    bp = _boxplot(axes[1], data, names)
    for patch, d in zip(bp["boxes"], datasets):
        patch.set_facecolor(d[4]); patch.set_alpha(0.5)
    axes[1].set_ylabel("tilt [deg]")
    axes[1].set_title("Tilt distribution", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    p = os.path.join(outdir, "fig3_tilt.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_bars(rows, outdir):
    specs = [
        ("z_detrend_rms_mm", "Vertical RMS [mm]"),
        ("z_detrend_p2p_mm", "Vertical peak-to-peak [mm]"),
        ("tilt_rms_deg", "Tilt RMS [deg]"),
        ("ang_vel_xy_rms", "Roll/pitch rate RMS [rad/s]"),
    ]
    if "z_detrend_rms_pct" in rows[0]:
        specs.insert(2, ("z_detrend_rms_pct", "Vertical RMS [% of body height]"))

    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.0))
    if n == 1:
        axes = [axes]
    names = [r["name"] for r in rows]
    colors = [r["_color"] for r in rows]

    for ax, (key, title) in zip(axes, specs):
        vals = [r[key] for r in rows]
        errs = [r.get(key + "_sd", 0.0) for r in rows]
        bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                      color=colors, alpha=0.85, edgecolor="white", lw=1.2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.18)
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_bars.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_psd(datasets, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ok = False
    for (name, df, dt, _, color) in datasets:
        f, psd, peak = gait_psd(df, dt)
        if f is None:
            continue
        ok = True
        lbl = f"{name}" + (f"  (peak {peak:.2f} Hz)" if peak else "")
        ax.semilogy(f, psd, lw=1.4, color=color, label=lbl)
        if peak:
            ax.axvline(peak, color=color, ls=":", lw=1.0, alpha=0.7)
    if not ok:
        plt.close(fig); return None
    ax.set_xlim(0, 15)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("PSD of z deviation")
    ax.set_title("Spectral content — peak indicates gait frequency",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(outdir, "fig5_psd.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_interaction(rows, outdir):
    """★ 상호작용 그래프. 지형 난이도에 따라 두 로봇이 어떻게 갈라지는가.

    --robot / --terrain 을 준 경우에만 그린다. 이번 실험의 핵심 그림.
    """
    if not all(r.get("_robot") and r.get("_terrain") for r in rows):
        return None
    robots, terrains = [], []
    for r in rows:
        if r["_robot"] not in robots:
            robots.append(r["_robot"])
        if r["_terrain"] not in terrains:
            terrains.append(r["_terrain"])
    if len(robots) < 2 or len(terrains) < 2:
        return None

    specs = [("z_detrend_rms_mm", "Vertical RMS [mm]"),
             ("tilt_rms_deg", "Tilt RMS [deg]")]
    if "z_detrend_rms_pct" in rows[0]:
        specs.insert(1, ("z_detrend_rms_pct", "Vertical RMS [% of body height]"))

    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.4))
    if len(specs) == 1:
        axes = [axes]
    xs = list(range(len(terrains)))

    for ax, (key, title) in zip(axes, specs):
        for i, rb in enumerate(robots):
            ys, es = [], []
            for tr in terrains:
                m = [r for r in rows if r["_robot"] == rb and r["_terrain"] == tr]
                ys.append(m[0][key] if m else np.nan)
                es.append(m[0].get(key + "_sd", 0.0) if m else 0.0)
            c = PALETTE[i % len(PALETTE)]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=9, lw=2.4, capsize=5,
                        color=c, label=rb, zorder=3)
            for x, y in zip(xs, ys):
                if np.isfinite(y):
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(0, 11), ha="center", fontsize=9, color=c,
                                fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(terrains)
        ax.set_xlim(-0.35, len(terrains) - 0.65)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.margins(y=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Terrain sensitivity — flat line means the terrain barely matters",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig6_interaction.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_terrain_following(datasets, outdir, max_points=4000):
    """★ 지형 높이 vs 몸체 높이. 기울기 0 = 완벽한 수평 유지."""
    usable = []
    for (name, df, dt, bh, color) in datasets:
        if "ground_z" not in df.columns:
            continue
        if df["ground_z"].std() < 1e-4:        # 평지는 의미 없음
            continue
        usable.append((name, df, color))
    if not usable:
        return None

    fig, axes = plt.subplots(1, len(usable), figsize=(4.6 * len(usable), 4.4),
                             squeeze=False, sharex=True, sharey=True)
    for ax, (name, df, color) in zip(axes[0], usable):
        key = "seg_key" if "seg_key" in df.columns else "env_id"
        xs, ys, slopes = [], [], []
        for _, g in df.groupby(key):
            x = g["ground_z"].to_numpy(float)
            y = g["z_world"].to_numpy(float)
            if len(x) < 10 or np.std(x) < 1e-4:
                continue
            x = (x - x.mean()) * 1000.0
            y = (y - y.mean()) * 1000.0
            xs.append(x); ys.append(y)
            slopes.append(np.polyfit(x, y, 1)[0])
        if not xs:
            continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        if len(X) > max_points:
            idx = np.random.default_rng(0).choice(len(X), max_points, replace=False)
            X, Y = X[idx], Y[idx]
        ax.scatter(X, Y, s=3, alpha=0.18, color=color, edgecolors="none")

        sl = float(np.mean(slopes))
        lim = np.percentile(np.abs(X), 99)
        gx = np.linspace(-lim, lim, 50)
        ax.plot(gx, sl * gx, color=color, lw=2.4, zorder=3,
                label=f"fit  slope = {sl:.3f}")
        ax.plot(gx, gx, color="0.35", lw=1.4, ls="--", zorder=2,
                label="slope 1  (follows terrain)")
        ax.axhline(0, color="0.35", lw=1.4, ls=":", zorder=2,
                   label="slope 0  (stays level)")
        ax.set_xlabel("terrain height deviation [mm]")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("body height deviation [mm]")
    fig.suptitle("Does the body ride the terrain, or stay level?", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig7_terrain_following.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_cdf(datasets, outdir, targets=(5.0, 15.0)):
    """누적분포. 목표치 이하로 유지되는 시간 비율을 읽을 수 있다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for (name, df, dt, bh, color) in datasets:
        v = np.sort(np.abs(df["z_detrend"].to_numpy()) * 1000.0)
        cdf = np.arange(1, len(v) + 1) / len(v) * 100.0
        axes[0].plot(v, cdf, lw=2.0, color=color, label=name)
    for t in targets:
        axes[0].axvline(t, color="0.4", lw=1.2, ls="--")
        axes[0].annotate(f"{t:g} mm", (t, 3), rotation=90, fontsize=8,
                         color="0.3", ha="right", va="bottom")
    axes[0].set_xlabel("|vertical deviation| [mm]")
    axes[0].set_ylabel("cumulative share of time [%]")
    axes[0].set_title("Vertical deviation CDF", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="lower right")
    xmax = max(np.percentile(np.abs(d[1]["z_detrend"]) * 1000.0, 99.5) for d in datasets)
    axes[0].set_xlim(0, xmax)

    for (name, df, dt, bh, color) in datasets:
        v = np.sort(df["tilt_deg"].to_numpy())
        cdf = np.arange(1, len(v) + 1) / len(v) * 100.0
        axes[1].plot(v, cdf, lw=2.0, color=color, label=name)
    axes[1].set_xlabel("tilt [deg]")
    axes[1].set_ylabel("cumulative share of time [%]")
    axes[1].set_title("Tilt CDF", loc="left", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    p = os.path.join(outdir, "fig8_cdf.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ====================================================================== #

REPORT_COLS = [
    ("z_detrend_rms_mm", "수직변동 RMS [mm]", 2),
    ("z_detrend_p2p_mm", "수직변동 P2P [mm]", 2),
    ("z_detrend_rms_pct", "수직변동 RMS [체고%]", 3),
    ("z_world_std_mm", "z 원시 표준편차 [mm]", 2),
    ("tilt_rms_deg", "틸트각 RMS [deg]", 3),
    ("tilt_p95_deg", "틸트각 P95 [deg]", 3),
    ("tilt_max_deg", "틸트각 최대 [deg]", 3),
    ("roll_rms_deg", "roll RMS [deg]", 3),
    ("pitch_rms_deg", "pitch RMS [deg]", 3),
    ("ang_vel_xy_rms", "각속도 xy RMS [rad/s]", 3),
    ("vx_mean", "평균 전진속도 [m/s]", 3),
    ("vx_track_err", "속도 추종오차 [m/s]", 3),
    ("terrain_std_mm", "★지형 높이 표준편차 [mm]", 2),
    ("terrain_range_mm", "★지형 높이 범위 [mm]", 1),
    ("terrain_follow", "지형 추종 계수 (0=수평유지, 1=추종)", 3),
    ("duration_s", "env당 측정시간 [s]", 1),
    ("n_envs", "env 수", 0),
    ("n_segments", "유효 구간 수", 0),
]


def main():
    ap = argparse.ArgumentParser(description="보행 fluctuation 지표 분석 및 시각화")
    ap.add_argument("--csv", action="append", required=True, help="probe CSV 경로 (반복 가능)")
    ap.add_argument("--name", action="append", default=None, help="각 CSV의 표시 이름")
    ap.add_argument("--body-height", action="append", type=float, default=None,
                    help="각 로봇의 기준 체고 [m]. Go2≈0.30, Hugo≈1.05")
    ap.add_argument("--robot", action="append", default=None,
                    help="각 CSV의 로봇 이름 (예: Go2, Hugo). "
                         "--terrain 과 함께 주면 상호작용 그래프를 그린다.")
    ap.add_argument("--terrain", action="append", default=None,
                    help="각 CSV의 지형 이름 (예: flat, bumps30)")
    ap.add_argument("--outdir", default="fluct_results")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="리셋 직후 버릴 시간 [s]")
    ap.add_argument("--detrend-window", type=float, default=1.0,
                    help="이동평균 창 [s]. 보행 주기보다 충분히 길게.")
    args = ap.parse_args()

    names = args.name or [os.path.splitext(os.path.basename(p))[0] for p in args.csv]
    heights = args.body_height or [None] * len(args.csv)
    if len(names) != len(args.csv):
        raise SystemExit("--name 개수가 --csv 개수와 다릅니다.")
    if len(heights) != len(args.csv):
        raise SystemExit("--body-height 개수가 --csv 개수와 다릅니다.")
    robots = args.robot or [None] * len(args.csv)
    terrains = args.terrain or [None] * len(args.csv)
    if len(robots) != len(args.csv) or len(terrains) != len(args.csv):
        raise SystemExit("--robot / --terrain 개수가 --csv 개수와 다릅니다.")

    os.makedirs(args.outdir, exist_ok=True)

    datasets, rows = [], []
    for i, (path, name, bh) in enumerate(zip(args.csv, names, heights)):
        raw, dt = load_csv(path)
        df = preprocess(raw, dt, args.warmup, args.detrend_window)
        m = metrics(df, bh)
        color = PALETTE[i % len(PALETTE)]
        datasets.append((name, df, dt, bh, color))
        row = {"name": name, "_color": color,
               "_robot": robots[i], "_terrain": terrains[i], **m}
        rows.append(row)
        print(f"[load] {name:<16} {path}  →  {len(df):,} rows, dt={dt:.4f}s")

    # ---- 표 ---------------------------------------------------------- #
    table = {}
    for key, label, _ in REPORT_COLS:
        if key not in rows[0]:
            continue
        table[label] = [r.get(key, float("nan")) for r in rows]
    tdf = pd.DataFrame(table, index=[r["name"] for r in rows]).T
    tdf = tdf.dropna(how="all")

    csv_path = os.path.join(args.outdir, "summary.csv")
    tdf.to_csv(csv_path, encoding="utf-8-sig")   # 엑셀 한글 깨짐 방지

    lines = ["| 지표 | " + " | ".join(tdf.columns) + " |",
             "|---|" + "---|" * len(tdf.columns)]
    dec = {label: d for _, label, d in REPORT_COLS}
    for label, vals in tdf.iterrows():
        d = dec.get(label, 3)
        lines.append(f"| {label} | " + " | ".join(f"{v:.{d}f}" for v in vals) + " |")
    md = "\n".join(lines)
    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write("# Fluctuation 비교 결과\n\n" + md + "\n")

    print("\n" + md + "\n")

    # ---- 그래프 ------------------------------------------------------ #
    made = [
        fig_interaction(rows, args.outdir),          # ★ 핵심 메시지
        fig_terrain_following(datasets, args.outdir),  # ★ P-R 직접 증거
        fig_bars(rows, args.outdir),
        fig_cdf(datasets, args.outdir),
        fig_timeseries(datasets, args.outdir),
        fig_distribution(datasets, args.outdir),
        fig_tilt(datasets, args.outdir),
        fig_psd(datasets, args.outdir),
    ]
    print("생성된 파일:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.outdir, 'summary.md')}")
    for p in made:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
