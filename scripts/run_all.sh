#!/usr/bin/env bash
# 完整实验矩阵的 Linux / WSL2 版本。
# 用法：bash scripts/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBSETS="${SUBSETS:-SMAP,MSL,SWaT,SMD}"
SEEDS="${SEEDS:-42,43,44}"
MODELS="${MODELS:-zscore,adaptive_zscore,iforest,convae,ours,ours_tta}"
DRIFTS="${DRIFTS:-none temporal channel channel_mask amplitude}"

for seed in ${SEEDS//,/ }; do
  for drift in $DRIFTS; do
    for model in ${MODELS//,/ }; do
      echo "==> seed=$seed model=$model drift=$drift"
      python -m src.train \
        -c configs/base.yaml \
        -c "configs/model/${model}.yaml" \
        --drift "$drift" \
        --seed "$seed" \
        -o "data.subsets=[${SUBSETS}]" \
        --tag "s${seed}_${model}_${drift}"
    done
  done
done

python -m src.evaluate \
  --results results \
  --out results/summary.md \
  --metric vus_pr \
  --ours ours \
  --baseline zscore \
  --baseline adaptive_zscore \
  --baseline iforest \
  --baseline convae \
  --per-seed