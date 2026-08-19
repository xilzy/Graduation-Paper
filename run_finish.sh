#!/bin/bash
# Orchestrate the tail of the ablation/param sweep:
#  1) launch the last two OOM-deferred reruns (hpK4, hpNs2) at bs6 as GPUs free
#  2) wait until all 17 new runs report "training done"
#  3) benchmark every new model (8-parallel), producing reports/<task>/<name>__means.csv
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper || exit 1
PY=/ytech_m2v4_hdd/lizhongyin/venv/bin/python
BASE6="--config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
--epochs 20 --batch-size 6 --lr 0.001 --weight 2 2 2 2 4 0 3 --alpha 2.0 --lr-decay 0.8 \
--window-size 8 --patch 170 --crops-per-task 4000 --n-routed 12 --k 2 --n-shared 1 \
--out-channel 96 --depth 4 --num-heads 8 --fusion-head blend --res-scale 0.0 \
--routing softmax --aux-weight 0.01 --out-scale --loss-mode maxfuse --ssim-target max --workers 4"

gpu_free () {  # echo first free gpu (mem<2000MiB) among args
  for g in "$@"; do
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 2000 ] && { echo "$g"; return; }
  done
}

# ---- Phase 1: launch hpK4, hpNs2 when a wave-2 GPU frees (avoid 0=wave3,4,6) ----
for pair in "hpK4:--k 4" "hpNs2:--n-shared 2"; do
  name=${pair%%:*}; ovr=${pair#*:}
  while :; do
    g=$(gpu_free 1 2 3 5 7)
    if [ -n "$g" ]; then
      CUDA_VISIBLE_DEVICES=$g nohup $PY train_moe.py $BASE6 --name "$name" --device cuda:0 $ovr \
        > "logs_${name}.txt" 2>&1 </dev/null &
      echo "launched $name on gpu$g $(date)"; sleep 40; break
    fi
    sleep 60
  done
done

# ---- Phase 2: wait for all 17 new runs ----
ALL="ab_noMoE_direct ab_direct_orig ab_ws1_orig hpNr4 hpNr16 hpK1 hpK4 hpNs0 hpNs2 \
hpD2 hpD5 hpOc64 hpOc128 hpWs4 hpWs16 hpAux001 hpAux1"
while :; do
  d=0; for n in $ALL; do grep -q "training done" "logs_${n}.txt" 2>/dev/null && d=$((d+1)); done
  echo "[finish] train $d/17 done $(date)"
  [ "$d" -ge 17 ] && break
  sleep 120
done
echo ALL_TRAIN_DONE

# ---- Phase 3: benchmark all (model_dir -> report method name) ----
map () { case $1 in
  ab_noMoE_direct) echo abNoMoE_direct;; ab_direct_orig) echo abDirect_orig;;
  ab_ws1_orig) echo abWs1_orig;; *) echo "$1";; esac; }
i=0
for md in $ALL; do
  g=$((i%8))
  bash bench/bench_one.sh "$md" "$(map "$md")" "cuda:$g" > "/tmp/bench_${md}.log" 2>&1 &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
done
wait
echo ALL_BENCH_DONE
