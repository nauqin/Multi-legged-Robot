#!/usr/bin/env bash
# 놓을 위치:  scripts/run_compare.sh
# 실행:       bash scripts/run_compare.sh            (프로젝트 루트 ~/WS/Hugo_Multi/Multi_Legged_Robot 에서)
#             bash scripts/run_compare.sh bumps      (지형 하나만)
#
# 3 조건 × 2 지형 추출 → 지형별 분석.
set -euo pipefail

PROBE=scripts/rsl_rl/compare_probe.py
ANALYZE=scripts/analysis/compare_analyze.py

USD="/home/sejong/WS/Hugo_Multi/usd files/hugo_hexapod_ver3_1_sphere/hugo_hexapod_ver3_1_sphere.usd"
CKPT_RL=logs/rsl_rl/cartpole_direct/2026-09-12_22-28-23/model_9199.pt
CKPT_CPG_RL=logs/rsl_rl/hugo_cpg/2026-09-17_20-21-23_fix_noise_curr/model_10998.pt

COMMON=(--num_envs 16 --duration 62 --speed 0.3 --seed 42 --terrain_seed 7 --headless --usd_path "$USD")
RAW=eval_results/raw
mkdir -p "$RAW"

TERRAINS=("${@:-bumps flat}")   # 인자로 지형을 주면 그것만
[ $# -eq 0 ] && TERRAINS=(bumps flat)

# 추출 한 번: 끝나고 CSV 가 없으면 즉시 중단 (Isaac 은 예외 후에도 종료코드 0 인 경우가 있다)
probe() {  # probe <out.csv> <probe 인자들...>
  local out="$1"; shift
  python $PROBE "$@" --out "$out" || true
  if [ ! -s "$out" ]; then
    echo "!!! 실패: $out 이 만들어지지 않았다. 위 로그의 Traceback 을 확인할 것." >&2
    exit 1
  fi
}

for TERRAIN in "${TERRAINS[@]}"; do
  echo "===== [$TERRAIN] RL only ====="
  probe "$RAW/rl_${TERRAIN}.csv"     --mode rl     --task Hugo-Hexapod-v0          --checkpoint "$CKPT_RL" \
      --terrain "$TERRAIN" "${COMMON[@]}"

  echo "===== [$TERRAIN] CPG only ====="
  probe "$RAW/cpg_${TERRAIN}.csv"    --mode cpg    --task Hugo-Hexapod-CPG-Play-v0 \
      --terrain "$TERRAIN" "${COMMON[@]}"

  echo "===== [$TERRAIN] CPG + RL ====="
  probe "$RAW/cpg_rl_${TERRAIN}.csv" --mode cpg_rl --task Hugo-Hexapod-CPG-Play-v0 --checkpoint "$CKPT_CPG_RL" \
      --terrain "$TERRAIN" "${COMMON[@]}"
done

for TERRAIN in "${TERRAINS[@]}"; do
  python $ANALYZE \
    --csv "$RAW/rl_${TERRAIN}.csv"     --name "RL only" \
    --csv "$RAW/cpg_${TERRAIN}.csv"    --name "CPG only" \
    --csv "$RAW/cpg_rl_${TERRAIN}.csv" --name "CPG + RL" \
    --outdir "eval_results/compare_${TERRAIN}"
done

echo "done →"; for T in "${TERRAINS[@]}"; do echo "  eval_results/compare_${T}/summary.md"; done
