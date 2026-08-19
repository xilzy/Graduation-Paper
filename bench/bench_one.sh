#!/bin/bash
# Benchmark one trained model end-to-end through the fusion_bench pipeline.
# Usage: bench_one.sh <model_dir_name> <method_name> [device]
set -e
MODEL=$1; NAME=$2; DEV=${3:-cuda:0}
PY=/ytech_m2v4_hdd/lizhongyin/venv/bin/python
CODE=/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper
BENCH=/ytech_m2v4_hdd/lizhongyin/fusion_bench
cd $CODE
F="FutureWarning|warnings.warn|meshgrid|cholesky|should be|This transform|^L = |^U = |^and$|requires_grad|Consider"
$PY bench/run_ours.py --model "$MODEL" --name "$NAME" --device "$DEV" 2>&1 | grep -vE "$F" | grep -E "wrote|model="
$PY bench/recombine_rescore.py --method "$NAME" 2>&1 | grep -E "recombine" || true
$PY bench/eval_method.py --task irvis   --name "$NAME" --fused-dir $BENCH/fused/$NAME/irvis        2>&1 | grep -E "leaderboard" >/dev/null
$PY bench/eval_method.py --task medical --name "$NAME" --fused-dir $BENCH/fused_final/$NAME/medical 2>&1 | grep -E "leaderboard" >/dev/null
$PY bench/eval_method.py --task gfp_pc  --name "$NAME" --fused-dir $BENCH/fused_final/$NAME/gfp_pc  2>&1 | grep -E "leaderboard" >/dev/null
echo "[$NAME] evaluated on 3 tasks"
