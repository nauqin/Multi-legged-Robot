#!/usr/bin/env python3
# analyze_fluct2.py
"""fluctuation_probe.py 가 만든 CSV를 다시 분석한다. (v2)

기존 analyze_fluctuation.py 의 아래 문제를 고친 버전이다.
재측정은 필요 없다. 같은 CSV를 다시 읽는다.

  [문제 1a] 이동평균 detrend 가 로봇마다 다른 양을 지웠다
            (Go2 79% / Hexapod 49%)
            -> 지우지 않는다. 느린 성분과 보행 성분으로 '나눠서' 둘 다 낸다.

  [문제 1b] 높이 스캐너 전방 오프셋 탓에 지형 상관의 부호가 뒤집혔다
            -> 지연(lag)을 스캔해서 상관이 최대가 되는 지점을 찾고,
               lag=0 값과 나란히 보고한다. 아티팩트인지 바로 보인다.

  [문제 2]  최댓값/P2P 는 오래 잴수록 커진다
            -> 모든 구간을 정확히 같은 길이로 자른다. P95 를 쓴다.

  [문제 3]  RMS 가 스파이크 하나에 끌려간다
            -> 유지율 곡선 P(|dz| < X) 를 주력으로. 순위 기반이라 안 흔들린다.

  [문제 4]  오차막대가 음수로 내려간다 (평균±SD 가 안 맞는 분포)
            -> 중앙값 + 사분위범위(IQR).

  [문제 5]  detrend 잔차의 PSD 로 보행주파수를 읽어 가짜 봉우리를 봤다
            + gait_psd() 가 길이 다른 구간의 주파수 눈금을 안 맞추는 버그
            -> 원시 z 로 PSD 를 내고, 눈금을 맞춘다. 그래도 잠정값으로 표기.

  [생존편향] 넘어져서 짧아진 구간이 조용히 버려졌다
            -> 조기 종료율을 지표로 만든다.

===========================================================================
사용법
===========================================================================
    python analyze_fluct2.py \
        --csv go2_flat.csv    --name "Go2 flat" \
        --csv go2_bumps30.csv --name "Go2 bumps" \
        --csv hexapod_flat.csv   --name "Hexapod flat" \
        --csv hexapod_bumps30.csv --name "Hexapod bumps" \
        --outdir results_v2

    # 시드가 여러 개면 (권장):
    #   --csv hexapod_bumps_s0.csv --name "Hexapod bumps" --seed 0
    #   --csv hexapod_bumps_s1.csv --name "Hexapod bumps" --seed 1
    #   --csv hexapod_bumps_s2.csv --name "Hexapod bumps" --seed 2
    # 같은 --name 끼리 묶여서 시드 간 통계가 나온다.

필요 패키지: numpy, pandas, matplotlib  (scipy 불필요)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = ["#2E5EAA", "#D1495B", "#00798C", "#EDAE49", "#7B6D8D", "#66A182"]

# 느린 성분 / 보행 성분을 가르는 주파수 [Hz].
# 실측 보행주파수가 0.88~1.00 Hz 이므로 그 아래로 넉넉히 잡는다.
BAND_SPLIT_HZ = 0.5

# 유지율 표에 쓸 문턱값 [mm]
RETENTION_MM = (5.0, 10.0, 15.0, 20.0, 30.0)

# 논문용 폰트 배율. 그림을 1단 폭(약 8cm)으로 축소하면 글자가 작아지므로
# 미리 키워둔다. 제목은 캡션이 대신하므로 배율 대상에서 제외.
FONT_SCALE = 1.3        # 축 라벨, 눈금, 범례
NUMBER_SCALE = 1.4      # 막대 '위'에 찍히는 합계 수치
INBAR_SCALE = 1.8       # 막대 '안'에 찍히는 성분 수치 (기본 폰트가 더 작아서 별도)
TITLE_SCALE = 1.4       # 그림 제목

def _fs(base: float, scale: float = None) -> float:
    return base * (FONT_SCALE if scale is None else scale)


# ====================================================================== #
# 1. 로딩
# ====================================================================== #

def load_csv(path: str):
    dt = None
    with open(path) as f:
        first = f.readline()
    skip = 1 if first.startswith("#") else 0
    if skip:
        for tok in first.strip().split(","):
            if "dt=" in tok:
                try:
                    dt = float(tok.split("dt=")[1])
                except ValueError:
                    pass
    df = pd.read_csv(path, skiprows=skip)
    if dt is None:
        t = np.sort(df["t"].unique())
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    return df, dt


# ====================================================================== #
# 2. 구간 자르기  —  [문제 2] + [생존편향]
# ====================================================================== #

def cut_windows(df: pd.DataFrame, dt: float, warmup_s: float, window_s: float):
    """에피소드별로 자르고, 모든 창을 '정확히 같은 길이'로 만든다.

    반환:
        windows : list[DataFrame]   길이가 전부 동일
        audit   : dict              조기 종료율 등 유효성 정보

    ★ 기존 코드는 짧은 구간을 조용히 버렸다. 여기서는 버리되 '센다'.
      넘어져서 짧아진 구간을 안 세면 잘 넘어지는 로봇이 안정적으로 보인다.
    """
    warm = int(round(warmup_s / dt))
    W = int(round(window_s / dt))

    # 에피소드 전체 길이(스텝)를 데이터에서 추정한다.
    # 리셋으로 끝난 구간의 마지막 ep_step 중 최댓값 ≈ 에피소드 길이.
    ep_ends = []
    segments = []          # (env_id, seg_idx, DataFrame, ended_by_reset)

    for env_id, g_env in df.groupby("env_id", sort=True):
        g_env = g_env.sort_values("t").reset_index(drop=True)
        if "ep_step" in g_env.columns:
            seg_id = (g_env["ep_step"].diff() < 0).cumsum()
        else:
            seg_id = pd.Series(0, index=g_env.index)
        keys = list(dict.fromkeys(seg_id.tolist()))
        for k_i, k in enumerate(keys):
            g = g_env[seg_id == k].reset_index(drop=True)
            # 마지막 구간은 '녹화 종료'로 잘린 것이지 리셋이 아니다.
            ended_by_reset = (k_i < len(keys) - 1)
            # 첫 구간은 '녹화 시작'으로 앞이 잘려 있다 (probe warmup 때문).
            truncated_start = (k_i == 0)
            if ended_by_reset and "ep_step" in g.columns:
                ep_ends.append(float(g["ep_step"].iloc[-1]))
            segments.append((env_id, k, g, ended_by_reset, truncated_start))

    ep_len = int(max(ep_ends)) if ep_ends else 0

    windows = []
    n_reset = n_timeout = n_early = n_short = 0

    for env_id, k, g, ended_by_reset, truncated_start in segments:
        # --- 종료 사유 분류 (앞이 잘린 첫 구간도 '끝'은 유효하다) ---
        if ended_by_reset and ep_len > 0 and "ep_step" in g.columns:
            n_reset += 1
            # 에피소드 길이의 98% 이상 채웠으면 타임아웃(정상), 아니면 조기종료
            if g["ep_step"].iloc[-1] >= 0.98 * ep_len:
                n_timeout += 1
            else:
                n_early += 1

        # --- 지표용 창 추출 ---
        # 앞이 잘린 첫 구간은 warmup 을 신뢰할 수 없으므로 통계에서 제외한다.
        if truncated_start:
            continue
        if "ep_step" in g.columns:
            g = g[g["ep_step"] >= warm]
        else:
            g = g.iloc[warm:]
        if len(g) < W:
            n_short += 1
            continue
        w = g.iloc[:W].reset_index(drop=True)          # ★ 정확히 W 스텝
        w = w.assign(win_key=f"{int(env_id)}:{int(k)}")
        windows.append(w)

    audit = {
        "window_steps": W,
        "window_s": W * dt,
        "episode_steps_est": ep_len,
        "n_reset": n_reset,
        "n_timeout": n_timeout,
        "n_early_term": n_early,
        "early_term_pct": (100.0 * n_early / n_reset) if n_reset else float("nan"),
        "n_windows": len(windows),
        "n_too_short": n_short,
    }
    return windows, audit


# ====================================================================== #
# 3. 대역 분해  —  [문제 1a]
# ====================================================================== #

def band_split_std(x: np.ndarray, dt: float, f_split: float = BAND_SPLIT_HZ):
    """신호를 느린 성분 / 빠른 성분으로 나눠 각각의 표준편차를 낸다.

    아무것도 버리지 않는다. 하드 컷 FFT 분할이라 에너지가 정확히 보존된다:
        sigma_total^2 = sigma_slow^2 + sigma_fast^2

    ★ 기존 코드의 이동평균 detrend 는 느린 성분을 '지웠고', 지운 양이
      로봇마다 79% / 49% 로 달랐다. 여기서는 지우지 않고 나눈다.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()                      # DC 제거
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=dt)

    slow = X.copy()
    slow[f > f_split] = 0.0
    fast = X.copy()
    fast[f <= f_split] = 0.0

    xs = np.fft.irfft(slow, n=n)
    xf = np.fft.irfft(fast, n=n)
    return float(np.std(x)), float(np.std(xs)), float(np.std(xf))


# ====================================================================== #
# 4. 지형 응답 + 지연 보정  —  [문제 1b]
# ====================================================================== #

def _slope_r2(x, y):
    x = x - x.mean(); y = y - y.mean()
    vx = float(np.dot(x, x))
    if vx < 1e-12:
        return np.nan, np.nan
    b = float(np.dot(x, y) / vx)
    resid = y - b * x
    vy = float(np.dot(y, y))
    r2 = 1.0 - float(np.dot(resid, resid)) / vy if vy > 1e-12 else np.nan
    return b, r2


def _shift_pairs(wins, dt, k, band_hi=None):
    """모든 창에서 (ground_z[t], z_world[t+k]) 쌍을 모아 이어붙인다.

    창 경계를 넘어 섞지 않으려고 창마다 따로 자른 뒤 합친다.
    band_hi 를 주면 그 주파수 이하만 남긴다 (지형 성분은 저주파에 있음).
    """
    xs, ys = [], []
    for w in wins:
        x = w["ground_z"].to_numpy(float)
        y = w["z_world"].to_numpy(float)
        if band_hi is not None:
            x = _lowpass(x, dt, band_hi)
            y = _lowpass(y, dt, band_hi)
        if k > 0:
            x2, y2 = x[:len(x) - k], y[k:]
        elif k < 0:
            x2, y2 = x[-k:], y[:len(y) + k]
        else:
            x2, y2 = x, y
        if len(x2) < 50:
            continue
        xs.append(x2 - x2.mean())
        ys.append(y2 - y2.mean())
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def _lowpass(x, dt, f_hi):
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, d=dt)
    X[f > f_hi] = 0.0
    return np.fft.irfft(X, n=n)


def terrain_response(wins: list, dt: float, max_lag_s: float = 1.5,
                     band_hi: float = 1.0, n_boot: int = 300):
    """지면 높이에 몸통/다리가 어떻게 반응하는지. ★ 전체 창을 풀링해서 추정.

    ★ 왜 창마다 따로 안 하고 합치나
      지형이 유발하는 몸통 움직임은 아주 작다. 실제 데이터에서 지형 std 는
      5.7mm 이고 응답 계수가 0.2 라면 몸통 기여는 1.1mm 인데, 몸통 전체
      변동은 12mm 다. 즉 R^2 가 1% 수준이라 창 하나(600스텝)로는 기울기를
      전혀 못 잡는다. 창 20~80개를 합쳐야 겨우 추정이 선다.
      (기존 코드는 창마다 회귀한 뒤 평균내서, 잡음을 평균낸 값을 봤다.)

    반환:
        body_slope_lag0  : 기존 방식 (지연 무시)
        body_slope_best  : 상관 최대 지연에서의 기울기
        best_lag_s       : 그 지연 [s]. (스캐너 전방오프셋 / 속도) 와 비교할 것
        best_r2          : 결정계수. 0.02 미만이면 기울기를 믿으면 안 된다
        slope_ci_lo/hi   : 창 단위 부트스트랩 95% 신뢰구간
        leg_slope_best   : 다리 흡수 계수 = body_slope_best - 1
                           -1 = 지면 변화를 다리가 100% 흡수 (완벽한 수평유지)
                            0 = 다리가 아무것도 안 함, 몸통이 지형을 그대로 탐
    """
    out = dict(body_slope_lag0=np.nan, body_slope_best=np.nan,
               best_lag_s=np.nan, best_r2=np.nan, leg_slope_best=np.nan,
               slope_ci_lo=np.nan, slope_ci_hi=np.nan)
    wins = [w for w in wins if "ground_z" in w.columns]
    if not wins:
        return out
    if np.std(np.concatenate([w["ground_z"].to_numpy() for w in wins])) < 1e-5:
        return out                                     # 평지 = 정의 불가

    x0, y0 = _shift_pairs(wins, dt, 0, band_hi)
    if x0 is None:
        return out
    out["body_slope_lag0"] = _slope_r2(x0, y0)[0]

    # --- 지연 스캔 (풀링) ---
    K = int(round(max_lag_s / dt))
    step = max(1, K // 60)                              # 60점 정도면 충분
    best = (-np.inf, 0, np.nan, np.nan)
    for k in range(-K, K + 1, step):
        x, y = _shift_pairs(wins, dt, k, band_hi)
        if x is None:
            continue
        b, r2 = _slope_r2(x, y)
        if np.isfinite(r2) and r2 > best[0]:
            best = (r2, k, b, r2)
    if best[0] == -np.inf:
        return out
    kbest = best[1]
    out["best_lag_s"] = kbest * dt
    out["body_slope_best"] = best[2]
    out["best_r2"] = best[3]
    out["leg_slope_best"] = best[2] - 1.0

    # --- 부트스트랩 신뢰구간 (창을 단위로 재추출) ---
    rng = np.random.default_rng(0)
    idx = np.arange(len(wins))
    boots = []
    for _ in range(n_boot):
        pick = [wins[i] for i in rng.choice(idx, len(idx), replace=True)]
        x, y = _shift_pairs(pick, dt, kbest, band_hi)
        if x is None:
            continue
        b, _r = _slope_r2(x, y)
        if np.isfinite(b):
            boots.append(b)
    if len(boots) > 20:
        out["slope_ci_lo"] = float(np.percentile(boots, 2.5))
        out["slope_ci_hi"] = float(np.percentile(boots, 97.5))
    return out


# ====================================================================== #
# 5. 창 하나당 지표
# ====================================================================== #

def window_metrics(w: pd.DataFrame, dt: float) -> dict:
    m = {}

    # --- 높이: 대역 분해 (mm) ------------------------------------------
    tot, slow, fast = band_split_std(w["z_world"].to_numpy(), dt)
    m["z_sigma_total_mm"] = tot * 1000.0
    m["z_sigma_slow_mm"] = slow * 1000.0
    m["z_sigma_gait_mm"] = fast * 1000.0
    m["z_slow_share_pct"] = 100.0 * (slow ** 2) / (tot ** 2) if tot > 0 else np.nan

    # 유지율에 쓸 편차: 창 평균 기준 (아무 필터도 안 씀)
    dz = (w["z_world"].to_numpy() - w["z_world"].mean()) * 1000.0
    m["z_p95_mm"] = float(np.percentile(np.abs(dz), 95))
    for thr in RETENTION_MM:
        m[f"z_within_{thr:g}mm_pct"] = float(np.mean(np.abs(dz) < thr) * 100.0)

    # --- 발밑 대비 높이 (다리 길이) ------------------------------------
    if "z_rel" in w.columns:
        t2, s2, f2 = band_split_std(w["z_rel"].to_numpy(), dt)
        m["zrel_sigma_total_mm"] = t2 * 1000.0
        m["zrel_sigma_gait_mm"] = f2 * 1000.0

    # --- 자세 ----------------------------------------------------------
    m["tilt_p95_deg"] = float(np.percentile(w["tilt_deg"], 95))
    m["tilt_median_deg"] = float(np.median(w["tilt_deg"]))
    m["tilt_rms_deg"] = float(np.sqrt(np.mean(w["tilt_deg"] ** 2)))
    m["roll_rms_deg"] = float(np.sqrt(np.mean(w["roll_deg"] ** 2)))
    m["pitch_rms_deg"] = float(np.sqrt(np.mean(w["pitch_deg"] ** 2)))

    # --- 각속도 --------------------------------------------------------
    m["angvel_xy_rms"] = float(np.sqrt(np.mean(w["wx"] ** 2 + w["wy"] ** 2)))
    m["angvel_xy_p95"] = float(np.percentile(np.hypot(w["wx"], w["wy"]), 95))

    # --- 수직 속도 (lin_vel_z_l2 보상항 대응) ---------------------------
    if "vz_w" in w.columns:
        m["vz_rms"] = float(np.sqrt(np.mean(w["vz_w"] ** 2)))

    # --- 속도: 평균만이 아니라 '편향'과 '출렁임'을 나눈다 ---------------
    vx = w["vx_b"].to_numpy(float)
    cmd = w["cmd_vx"].to_numpy(float) if "cmd_vx" in w.columns else np.full_like(vx, np.nan)
    m["vx_mean"] = float(vx.mean())
    m["vx_std"] = float(vx.std())                       # ★ 출렁임
    m["vx_bias"] = float(np.nanmean(vx - cmd))          # ★ 편향
    m["vx_mae"] = float(np.nanmean(np.abs(vx - cmd)))

    # --- 지형 ----------------------------------------------------------
    if "ground_z" in w.columns:
        gz = w["ground_z"].to_numpy(float)
        m["terrain_sigma_mm"] = float(np.std(gz)) * 1000.0
        m["terrain_p2p_mm"] = float(gz.max() - gz.min()) * 1000.0
    return m


# ====================================================================== #
# 6. 집계  —  [문제 4]
# ====================================================================== #

def aggregate(per_window: list[dict], seeds: list) -> dict:
    """중앙값 + 사분위범위. 시드가 여러 개면 시드 평균을 먼저 낸다.

    ★ 기존 코드는 세그먼트 80개의 평균±SD 를 냈는데, 이건 '정책 하나를
      80번 쪼개 본' 변동이라 오차막대가 실제보다 좁거나(유사반복) 스파이크
      때문에 음수로 내려갔다(평균±SD 부적합).
    """
    pw = pd.DataFrame(per_window)
    if seeds and len(set(seeds)) > 1:
        pw = pw.assign(_seed=seeds).groupby("_seed").mean(numeric_only=True)
        level = "seed"
    else:
        level = "window"

    agg = {}
    for c in pw.columns:
        v = pw[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            agg[c] = np.nan
            agg[c + "_q1"] = agg[c + "_q3"] = np.nan
            continue
        agg[c] = float(np.median(v))
        agg[c + "_q1"] = float(np.percentile(v, 25))
        agg[c + "_q3"] = float(np.percentile(v, 75))
    agg["_n"] = int(len(pw))
    agg["_level"] = level
    return agg


# ====================================================================== #
# 7. 그림
# ====================================================================== #

def fig_retention(datasets, outdir, show_title=True):
    """★ 대표 그림. P(|dz| < X). 순위 기반이라 스파이크에 안 흔들린다."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for (name, wins, dt, color) in datasets:
        dz = np.concatenate([
            np.abs(w["z_world"].to_numpy() - w["z_world"].mean()) * 1000.0
            for w in wins])
        v = np.sort(dz)
        cdf = np.arange(1, len(v) + 1) / len(v) * 100.0
        ax.plot(v, cdf, lw=2.4, color=color, label=name)
    for thr in (10.0, 15.0):
        ax.axvline(thr, color="0.5", lw=1.0, ls="--")
        ax.annotate(f"{thr:g} mm", (thr, 4), rotation=90,
                    fontsize=_fs(8, NUMBER_SCALE),
                    color="0.35", ha="right", va="bottom")
    ax.set_xlabel("|body height deviation| [mm]", fontsize=_fs(10))
    ax.set_ylabel("share of time kept within [%]", fontsize=_fs(10))
    ax.tick_params(labelsize=_fs(10))
    if show_title:
        ax.set_title("Height retention  —  higher & lefter is better",
                     loc="left", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 40)
    ax.legend(frameon=False, loc="lower right", fontsize=_fs(10))
    fig.tight_layout()
    p = os.path.join(outdir, "v2_fig1_retention.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_bands(rows, outdir, show_title=True):
    """동체 높이 변동을 저주파 / 고주파로 나눈 누적 막대.

    ★ 두 대역은 '느린 흔들림'과 '빠른 흔들림'이라는 대등한 짝이다.
      이전에 쓰던 slow / gait 는 한쪽은 속도, 한쪽은 원인을 가리켜
      층위가 어긋났다. 주파수 기준으로 통일한다.

    ★ 합계 수치는 막대 '안' 상단에 넣는다. 막대 위에 두면 제목과 자리를
      다투게 되고, ylim 을 늘리면 막대가 납작해져 보기 나빠진다.
    """
    names = [r["name"] for r in rows]
    lo_b = [r["z_sigma_slow_mm"] for r in rows]     # 저주파
    hi_b = [r["z_sigma_gait_mm"] for r in rows]     # 고주파
    tot = [r["z_sigma_total_mm"] for r in rows]
    x = np.arange(len(names))
    top = max(t for t in tot if np.isfinite(t))
    stack_top = max(a + b for a, b in zip(lo_b, hi_b))

    fig, ax = plt.subplots(figsize=(1.85 * len(names) + 2.2, 4.6))
    ax.bar(x, lo_b, 0.66, color="#7B6D8D", zorder=2,
           label=f"Low-frequency  (< {BAND_SPLIT_HZ:g} Hz)")
    ax.bar(x, hi_b, 0.66, bottom=lo_b, color="#EDAE49", zorder=2,
           label=f"High-frequency  (> {BAND_SPLIT_HZ:g} Hz)")

    for i, (a, b, t) in enumerate(zip(lo_b, hi_b, tot)):
        # 합계: 막대 바로 위. 아래 ylim 여유(1.16배)가 이 글자 자리다.
        ax.text(i, a + b + stack_top * 0.025, f"{t:.1f}", ha="center",
                va="bottom", fontsize=_fs(9, NUMBER_SCALE),
                fontweight="bold", zorder=3)
        # 성분: 각 구간 중앙.
        if a > stack_top * 0.10:
            ax.text(i, a / 2, f"{a:.1f}", ha="center", va="center",
                    fontsize=_fs(8, INBAR_SCALE), color="white", zorder=3)
        if b > stack_top * 0.10:
            ax.text(i, a + b / 2, f"{b:.1f}", ha="center", va="center",
                    fontsize=_fs(8, INBAR_SCALE), color="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=_fs(10))
    ax.tick_params(axis="y", labelsize=_fs(10))
    ax.set_ylabel("body height std [mm]", fontsize=_fs(10))
    ax.set_ylim(0, stack_top * 1.16)   # 막대 위 합계 숫자 자리
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=_fs(10), ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.11),
              handlelength=1.5, columnspacing=2.5)
    if show_title:
        ax.set_title("Height variation split by frequency band",
                     loc="left", fontweight="bold", pad=12, fontsize=_fs(12, TITLE_SCALE))
    fig.tight_layout()
    p = os.path.join(outdir, "v2_fig2_bands.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_lag(rows, outdir):
    """지연 보정 전후의 지형 응답. 부호가 뒤집히면 스캐너 오프셋 아티팩트."""
    use = [r for r in rows if np.isfinite(r.get("body_slope_best", np.nan))]
    if not use:
        return None
    names = [r["name"] for r in use]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(x - 0.2, [r["body_slope_lag0"] for r in use], 0.38,
                label="lag = 0  (old)", color="#B0B0B0")
    axes[0].bar(x + 0.2, [r["body_slope_best"] for r in use], 0.38,
                label="lag corrected", color="#2E5EAA")
    axes[0].axhline(0, color="k", lw=0.9)
    axes[0].set_title("Body response to terrain", loc="left", fontweight="bold")
    axes[0].set_ylabel("slope  (0 = level, 1 = rides terrain)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, [r["best_lag_s"] for r in use], 0.55, color="#D1495B")
    axes[1].axhline(0, color="k", lw=0.9)
    axes[1].set_title("Best-correlation lag", loc="left", fontweight="bold")
    axes[1].set_ylabel("lag [s]")

    axes[2].bar(x, [r["best_r2"] for r in use], 0.55, color="#00798C")
    axes[2].set_title("R-squared  (low = slope not trustworthy)",
                      loc="left", fontweight="bold")
    axes[2].set_ylabel("R²")

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, fontsize=9)
    fig.suptitle("Terrain response — does lag correction flip the sign?",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "v2_fig3_lag.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_interaction(rows, outdir):
    """로봇 x 지형. 선이 평평하면 지형이 별 영향을 못 준다는 뜻."""
    use = [r for r in rows if r.get("_robot") and r.get("_terrain")]
    if len(use) < 4:
        return None
    terr = list(dict.fromkeys(r["_terrain"] for r in use))
    robs = list(dict.fromkeys(r["_robot"] for r in use))
    keys = [("z_sigma_total_mm", "Height std [mm]"),
            ("z_sigma_slow_mm", "Slow component [mm]"),
            ("z_sigma_gait_mm", "Gait component [mm]"),
            ("tilt_p95_deg", "Tilt P95 [deg]")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4.2))
    for ax, (k, label) in zip(axes, keys):
        for ri, rb in enumerate(robs):
            xs, ys, lo, hi = [], [], [], []
            for ti, t in enumerate(terr):
                m = [r for r in use if r["_robot"] == rb and r["_terrain"] == t]
                if not m:
                    continue
                r = m[0]
                xs.append(ti); ys.append(r.get(k, np.nan))
                lo.append(abs(r.get(k, np.nan) - r.get(k + "_q1", np.nan)))
                hi.append(abs(r.get(k + "_q3", np.nan) - r.get(k, np.nan)))
            c = PALETTE[ri % len(PALETTE)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=8, lw=2.0,
                        capsize=4, color=c, label=rb)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8,
                            fontweight="bold", color=c)
        ax.set_xticks(range(len(terr))); ax.set_xticklabels(terr)
        ax.set_xlim(-0.4, len(terr) - 0.6)
        ax.margins(y=0.25)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Robot x terrain  —  flat line means terrain barely matters",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "v2_fig5_interaction.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_spread(rows, outdir):
    """중앙값 + IQR. 겹치면 '차이 있다'고 말할 수 없다."""
    keys = [("z_sigma_total_mm", "Height std [mm]"),
            ("tilt_p95_deg", "Tilt P95 [deg]"),
            ("angvel_xy_rms", "Roll/pitch rate RMS [rad/s]"),
            ("vx_mean", "Forward speed [m/s]")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 4.2))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    for ax, (k, label) in zip(axes, keys):
        med = [r.get(k, np.nan) for r in rows]
        lo = [r.get(k) - r.get(k + "_q1", np.nan) for r in rows]
        hi = [r.get(k + "_q3", np.nan) - r.get(k) for r in rows]
        cols = [r["_color"] for r in rows]
        ax.bar(x, med, 0.6, color=cols, alpha=0.85)
        ax.errorbar(x, med, yerr=[np.abs(lo), np.abs(hi)], fmt="none",
                    ecolor="k", capsize=4, lw=1.2)
        for i, v in enumerate(med):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=14, fontsize=8)
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
        ax.set_ylim(bottom=0)
    fig.suptitle("Median with interquartile range  —  overlap means no claim",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "v2_fig4_spread.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ====================================================================== #
# 8. 보고
# ====================================================================== #

GATE_ROWS = [
    ("early_term_pct", "조기 종료율 [%]", 1),
    ("n_early_term", "조기 종료 횟수", 0),
    ("n_reset", "전체 리셋 횟수", 0),
    ("n_windows", "유효 창 수", 0),
    ("n_too_short", "길이 미달로 제외", 0),
    ("window_s", "창 길이 [s]", 1),
    ("terrain_sigma_mm", "지형 높이 std [mm]", 2),
]

MAIN_ROWS = [
    ("z_sigma_total_mm", "몸통 높이 std — 전체 [mm]", 2),
    ("z_sigma_slow_mm", "  저주파 성분 (<0.5Hz) [mm]", 2),
    ("z_sigma_gait_mm", "  고주파 성분 (>0.5Hz) [mm]", 2),
    ("z_slow_share_pct", "  저주파 성분 비중 [%]", 1),
    ("z_p95_mm", "높이 편차 P95 [mm]", 2),
    ("tilt_p95_deg", "틸트각 P95 [deg]", 2),
    ("tilt_median_deg", "틸트각 중앙값 [deg]", 2),
    ("angvel_xy_rms", "각속도 xy RMS [rad/s]", 3),
    ("vx_mean", "평균 전진속도 [m/s]", 3),
    ("vx_std", "속도 출렁임 std [m/s]", 3),
    ("vx_bias", "속도 편향 (실측−명령) [m/s]", 3),
]

NORM_ROWS = [
    ("body_height_m", "기준 체고 [m]", 3),
    ("z_sigma_total_pct", "몸통 높이 std — 전체 [체고%]", 3),
    ("z_sigma_slow_pct", "  저주파 성분 [체고%]", 3),
    ("z_sigma_gait_pct", "  고주파 성분 [체고%]", 3),
    ("z_p95_pct", "높이 편차 P95 [체고%]", 3),
]

SUB_ROWS = [
    ("zrel_sigma_total_mm", "발밑대비 높이 std [mm]", 2),
    ("vz_rms", "수직속도 RMS [m/s]", 3),
    ("tilt_rms_deg", "틸트각 RMS [deg] (참고)", 2),
    ("roll_rms_deg", "roll RMS [deg]", 2),
    ("pitch_rms_deg", "pitch RMS [deg]", 2),
    ("body_slope_lag0", "지형응답 기울기 (lag=0, 구방식)", 3),
    ("body_slope_best", "지형응답 기울기 (지연보정)", 3),
    ("best_lag_s", "  최대상관 지연 [s]", 3),
    ("best_r2", "  그때의 R²", 3),
    ("slope_ci_lo", "  기울기 95%CI 하한", 3),
    ("slope_ci_hi", "  기울기 95%CI 상한", 3),
    ("leg_slope_best", "다리 흡수 계수 (−1=완전흡수)", 3),
]


def verdicts(rows, level: str) -> str:
    """어떤 숫자를 믿으면 안 되는지 스크립트가 직접 말해준다.

    지표를 잘못 읽는 사고를 막는 게 목적이다. 사람이 표만 보고
    '우세'를 판단하면 이 판정들을 놓치기 쉽다.
    """
    out = []

    if level == "window":
        out.append("🔴 **시드 1개** — IQR 은 같은 정책을 여러 창으로 쪼갠 변동일 뿐입니다. "
                   "이 오차범위로는 조건 간 차이를 주장할 수 없습니다. "
                   "학습 시드 3개 이상이 필요합니다.")

    # 조기 종료율
    for r in rows:
        p = r.get("early_term_pct", float("nan"))
        if np.isfinite(p) and p > 10.0:
            out.append(f"🔴 **{r['name']}**: 조기 종료율 {p:.1f}% — 넘어진 구간이 "
                       f"통계에서 빠졌습니다. 안정성 수치가 실제보다 좋게 나옵니다.")
        elif np.isfinite(p) and p > 3.0:
            out.append(f"🟡 **{r['name']}**: 조기 종료율 {p:.1f}% — 본문에 함께 보고하세요.")

    # 지형 동일성
    ts = [(r["name"], r.get("terrain_sigma_mm", float("nan"))) for r in rows]
    ts = [(n, v) for n, v in ts if np.isfinite(v) and v > 0.1]
    if len(ts) >= 2:
        lo, hi = min(v for _, v in ts), max(v for _, v in ts)
        if hi / lo > 1.10:
            out.append(f"🟡 지형 높이 std 가 조건마다 다릅니다 ({lo:.2f}~{hi:.2f} mm, "
                       f"{hi/lo:.2f}배). 같은 지형을 밟았다는 근거가 약합니다. "
                       f"지형 생성 해시로 확인하세요.")

    # 지형 응답
    for r in rows:
        r2 = r.get("best_r2", float("nan"))
        lag = r.get("best_lag_s", float("nan"))
        lo, hi = r.get("slope_ci_lo", np.nan), r.get("slope_ci_hi", np.nan)
        s0, sb = r.get("body_slope_lag0", np.nan), r.get("body_slope_best", np.nan)
        if not np.isfinite(r2):
            continue
        if r2 < 0.02:
            out.append(f"🔴 **{r['name']}**: 지형응답 R²={r2:.3f} — 지형이 몸통 변동을 "
                       f"거의 설명하지 못합니다. 기울기({sb:+.3f})는 신뢰할 수 없으니 "
                       f"논문에 쓰지 마세요.")
        if np.isfinite(lo) and np.isfinite(hi) and lo < 0 < hi:
            out.append(f"🟡 **{r['name']}**: 지형응답 기울기 95%CI [{lo:+.3f}, {hi:+.3f}] 가 "
                       f"0 을 포함합니다 — 0 과 구별되지 않습니다.")
        if np.isfinite(lag) and abs(lag) > 0.10:
            out.append(f"🟡 **{r['name']}**: 최대상관 지연 {lag:+.3f}s — 높이 스캐너가 "
                       f"몸통과 다른 곳을 보고 있다는 뜻입니다. "
                       f"(전방오프셋 ÷ 전진속도) 와 비교해보세요.")
        if np.isfinite(s0) and np.isfinite(sb) and s0 * sb < 0:
            out.append(f"🔴 **{r['name']}**: 지연을 보정하니 지형응답 부호가 "
                       f"{s0:+.3f} → {sb:+.3f} 로 뒤집혔습니다. "
                       f"기존 값은 스캐너 위치가 만든 착시입니다.")

    # 느린 성분 비중이 조건마다 크게 다른가 (구방식 detrend 가 왜 위험했는지)
    sh = [(r["name"], r.get("z_slow_share_pct", float("nan"))) for r in rows]
    sh = [(n, v) for n, v in sh if np.isfinite(v)]
    if len(sh) >= 2 and (max(v for _, v in sh) - min(v for _, v in sh)) > 15:
        detail = ", ".join(f"{n} {v:.0f}%" for n, v in sh)
        out.append(f"🟡 느린 성분 비중이 조건마다 다릅니다 ({detail}). "
                   f"구방식(이동평균 detrend)은 이 성분을 통째로 지웠으므로, "
                   f"조건마다 다른 양을 버린 셈이었습니다. 전체·느린·보행 세 값을 "
                   f"모두 보고하세요.")

    # 절대값과 정규화값의 순위가 뒤바뀌는가
    have = [r for r in rows if np.isfinite(r.get("z_sigma_total_pct", np.nan))]
    if len(have) >= 2:
        best_abs = min(have, key=lambda r: r["z_sigma_total_mm"])["name"]
        best_pct = min(have, key=lambda r: r["z_sigma_total_pct"])["name"]
        if best_abs != best_pct:
            out.append(f"🔴 **정규화가 결론을 뒤집습니다** — 절대 mm 로는 "
                       f"'{best_abs}' 가, 체고% 로는 '{best_pct}' 가 가장 안정적입니다. "
                       f"둘 다 본문에 싣고, 어느 쪽을 근거로 무엇을 주장하는지 "
                       f"반드시 명시하세요. 한쪽만 쓰면 체리피킹으로 읽힙니다.")

    if not out:
        return "✅ 자동 점검에서 걸린 항목이 없습니다.\n"
    return "\n".join(f"- {o}" for o in out) + "\n"


def make_table(rows, spec, with_iqr=True):
    lines = ["| 지표 | " + " | ".join(r["name"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for key, label, dec in spec:
        if not any(key in r for r in rows):
            continue
        cells = []
        for r in rows:
            v = r.get(key, np.nan)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
                continue
            s = f"{v:.{dec}f}"
            q1, q3 = r.get(key + "_q1"), r.get(key + "_q3")
            if with_iqr and q1 is not None and q3 is not None \
                    and np.isfinite(q1) and np.isfinite(q3):
                s += f" [{q1:.{dec}f}–{q3:.{dec}f}]"
            cells.append(s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def retention_table(rows):
    lines = ["| 유지 문턱 | " + " | ".join(r["name"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for thr in RETENTION_MM:
        k = f"z_within_{thr:g}mm_pct"
        cells = [f"{r.get(k, float('nan')):.1f}%" for r in rows]
        lines.append(f"| ±{thr:g} mm 이내 유지 시간 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ====================================================================== #

def main():
    global BAND_SPLIT_HZ
    ap = argparse.ArgumentParser(description="보행 fluctuation 재분석 v2")
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--name", action="append", default=None)
    ap.add_argument("--seed", action="append", default=None,
                    help="같은 --name 의 CSV 를 시드별로 줄 때 사용")
    ap.add_argument("--outdir", default="figs",
                    help="결과 폴더 접두어. 실행 시각이 뒤에 붙는다 "
                         "(예: figs_2026-09-01_12-54)")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="시각 접미어 없이 --outdir 그대로 사용 (덮어씀)")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="리셋 직후 버릴 시간 [s]")
    ap.add_argument("--window", type=float, default=12.0,
                    help="모든 구간을 자를 공통 길이 [s]. ★ 조건별로 같아야 함")
    ap.add_argument("--band-split", type=float, default=0.5,
                    help="느린/보행 성분을 가르는 주파수 [Hz]")
    ap.add_argument("--body-height", action="append", type=float, default=None,
                    help="각 CSV 로봇의 기준 체고 [m]. 주면 체고 정규화 지표가 추가됨")
    ap.add_argument("--robot", action="append", default=None,
                    help="각 CSV 의 로봇 이름. --terrain 과 함께 주면 상호작용 그래프")
    ap.add_argument("--terrain", action="append", default=None,
                    help="각 CSV 의 지형 이름")
    ap.add_argument("--no-fig-title", action="store_true",
                    help="논문용. 그림 안의 제목을 빼고 캡션에 맡긴다")
    args = ap.parse_args()

    BAND_SPLIT_HZ = args.band_split

    names = args.name or [os.path.splitext(os.path.basename(p))[0] for p in args.csv]
    if len(names) != len(args.csv):
        raise SystemExit("--name 개수가 --csv 개수와 다릅니다.")
    n_csv = len(args.csv)
    seeds = args.seed or [None] * n_csv
    heights = args.body_height or [None] * n_csv
    robots = args.robot or [None] * n_csv
    terrains = args.terrain or [None] * n_csv
    for lbl, lst in (("--seed", seeds), ("--body-height", heights),
                     ("--robot", robots), ("--terrain", terrains)):
        if len(lst) != n_csv:
            raise SystemExit(f"{lbl} 개수({len(lst)})가 --csv 개수({n_csv})와 다릅니다.")

    # 실행할 때마다 새 폴더를 만든다. 옛 결과를 덮어써서 비교를 못 하게 되는
    # 사고를 막는다. 조건을 바꿔가며 여러 번 돌릴 때 특히 중요하다.
    if args.no_timestamp:
        outdir = args.outdir
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir = f"{args.outdir}_{stamp}"
    os.makedirs(outdir, exist_ok=True)
    args.outdir = outdir
    print(f"[out] 결과 폴더: {outdir}")

    # 재현을 위해 실행 명령과 설정을 남긴다
    with open(os.path.join(outdir, "run_info.txt"), "w") as f:
        f.write(f"실행 시각 : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"작업 폴더 : {os.getcwd()}\n")
        f.write(f"창 길이   : {args.window}s\n")
        f.write(f"warmup    : {args.warmup}s\n")
        f.write(f"대역 경계 : {args.band_split} Hz\n\n")
        f.write("명령:\n" + " ".join(sys.argv) + "\n")

    # --- 같은 name 끼리 묶는다 (시드 병합) ------------------------------
    groups = OrderedDict()
    meta = {}
    for path, name, sd, bh, rb, tr in zip(args.csv, names, seeds, heights,
                                          robots, terrains):
        groups.setdefault(name, []).append((path, sd))
        meta.setdefault(name, dict(body_height=bh, robot=rb, terrain=tr))

    rows, datasets, win_dump = [], [], []
    print()
    for gi, (name, items) in enumerate(groups.items()):
        all_windows, all_seeds, audits, dt_ref = [], [], [], None
        for path, sd in items:
            df, dt = load_csv(path)
            dt_ref = dt if dt_ref is None else dt_ref
            wins, audit = cut_windows(df, dt, args.warmup, args.window)
            all_windows += wins
            all_seeds += [sd] * len(wins)
            audits.append(audit)
            print(f"[load] {name:<14} {os.path.basename(path):<24} "
                  f"dt={dt:.4f}s  창 {audit['n_windows']}개  "
                  f"조기종료 {audit['early_term_pct']:.1f}%")
        if not all_windows:
            raise SystemExit(f"{name}: 유효한 창이 없습니다. --window 를 줄여보세요.")

        per_win = [window_metrics(w, dt_ref) for w in all_windows]
        agg = aggregate(per_win, all_seeds)

        # 지형 응답은 창 하나로는 신호가 너무 약하다. 전체를 풀링해서 한 번만.
        agg.update(terrain_response(all_windows, dt_ref))

        # 유효성 정보는 합산
        for k in ("n_reset", "n_timeout", "n_early_term", "n_windows", "n_too_short"):
            agg[k] = int(sum(a[k] for a in audits))
        agg["early_term_pct"] = (100.0 * agg["n_early_term"] / agg["n_reset"]
                                 if agg["n_reset"] else float("nan"))
        agg["window_s"] = audits[0]["window_s"]

        mt = meta.get(name, {})
        bh = mt.get("body_height")
        if bh:
            for k in ("z_sigma_total_mm", "z_sigma_slow_mm",
                      "z_sigma_gait_mm", "z_p95_mm"):
                for suf in ("", "_q1", "_q3"):
                    src, dst = k + suf, k.replace("_mm", "_pct") + suf
                    if src in agg and np.isfinite(agg.get(src, np.nan)):
                        agg[dst] = agg[src] / (bh * 1000.0) * 100.0
            agg["body_height_m"] = bh

        for i, pwm in enumerate(per_win):
            pwm["_name"] = name
            pwm["_win"] = all_windows[i]["win_key"].iloc[0]
        win_dump.extend(per_win)

        color = PALETTE[gi % len(PALETTE)]
        agg.update(name=name, _color=color,
                   _robot=mt.get("robot"), _terrain=mt.get("terrain"))
        rows.append(agg)
        datasets.append((name, all_windows, dt_ref, color))

    # ---------------- 출력 ---------------- #
    level = rows[0].get("_level", "window")
    warn = ""

    md = []
    md.append("# Fluctuation 재분석 결과 (v2)\n")
    md.append(f"창 길이 {args.window:g}s · warmup {args.warmup:g}s · "
              f"대역 경계 {BAND_SPLIT_HZ:g} Hz · 값은 **중앙값 [Q1–Q3]**\n")
    md.append("## 0. 자동 판정  (★ 먼저 읽으세요)\n")
    md.append(verdicts(rows, level) + "\n")
    md.append("## 1. 유효성 게이트\n")
    md.append("여기가 이상하면 아래 숫자는 의미가 없습니다.\n")
    md.append(make_table(rows, GATE_ROWS, with_iqr=False) + "\n")
    md.append("## 2. 주 지표\n")
    md.append(make_table(rows, MAIN_ROWS) + "\n")
    md.append(warn)
    if any("z_sigma_total_pct" in r for r in rows):
        md.append("\n## 2b. 체고 정규화  (⚠️ 절대값과 함께 읽을 것)\n")
        md.append(make_table(rows, NORM_ROWS) + "\n")
    md.append("\n## 3. 높이 유지율  (★ 대표 지표)\n")
    md.append(retention_table(rows) + "\n")
    md.append("\n## 4. 보조 / 진단 지표\n")
    md.append(make_table(rows, SUB_ROWS) + "\n")

    text = "\n".join(md)
    with open(os.path.join(args.outdir, "summary_v2.md"), "w") as f:
        f.write(text)

    # 창 하나하나의 값. IQR 이 넓을 때 어느 env 가 범인인지 찾는 용도.
    if win_dump:
        wd = pd.DataFrame(win_dump)
        cols = ["_name", "_win"] + [c for c in wd.columns if not c.startswith("_")]
        wd[cols].to_csv(os.path.join(args.outdir, "windows_v2.csv"),
                        index=False, encoding="utf-8-sig")

    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if not k.startswith("_")})
    pd.DataFrame(flat).set_index("name").T.to_csv(
        os.path.join(args.outdir, "summary_v2.csv"), encoding="utf-8-sig")

    print("\n" + text)

    show_t = not args.no_fig_title
    made = [fig_retention(datasets, args.outdir, show_title=show_t),
            fig_bands(rows, args.outdir, show_title=show_t),
            fig_lag(rows, args.outdir),
            fig_spread(rows, args.outdir),
            fig_interaction(rows, args.outdir)]
    print("생성:")
    for p in [os.path.join(args.outdir, "summary_v2.md"),
              os.path.join(args.outdir, "summary_v2.csv"),
              os.path.join(args.outdir, "windows_v2.csv"),
              os.path.join(args.outdir, "run_info.txt")] + [m for m in made if m]:
        print("  " + p)


if __name__ == "__main__":
    main()
