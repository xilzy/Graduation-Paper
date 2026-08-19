#!/bin/bash
# Ablation(double) + parameter sweep for the v3 study. 17 runs, waves of 8 GPUs.
# Each run = v3 base config with ONE (or two, for double-ablation) flags overridden.
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper || exit 1
PY=/ytech_m2v4_hdd/lizhongyin/venv/bin/python
BASE="--config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
--epochs 20 --batch-size 10 --lr 0.001 --weight 2 2 2 2 4 0 3 --alpha 2.0 --lr-decay 0.8 \
--window-size 8 --patch 170 --crops-per-task 4000 --n-routed 12 --k 2 --n-shared 1 \
--out-channel 96 --depth 4 --num-heads 8 --fusion-head blend --res-scale 0.0 \
--routing softmax --aux-weight 0.01 --out-scale --loss-mode maxfuse --ssim-target max --workers 6"

run () {  # name  gpu  "override flags"
  local name=$1 gpu=$2 ovr=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY train_moe.py $BASE --name "$name" --device cuda:0 $ovr \
     > "logs_${name}.txt" 2>&1 &
}

wave () { wait; echo "=== wave done: $(date) ==="; }

# ---- Wave 1 (8) : double-ablations + n_routed/k ----
run ab_noMoE_direct 0 "--n-routed 0 --fusion-head direct"
run ab_direct_orig  1 "--fusion-head direct --loss-mode orig"
run ab_ws1_orig     2 "--window-size 1 --loss-mode orig"
run hpNr4           3 "--n-routed 4"
run hpNr16          4 "--n-routed 16"
run hpK1            5 "--k 1"
run hpK4            6 "--k 4"
run hpNs0           7 "--n-shared 0"
wave

# ---- Wave 2 (8) : n_shared/depth/out_channel/window ----
run hpNs2    0 "--n-shared 2"
run hpD2     1 "--depth 2"
run hpD5     2 "--depth 5 --batch-size 6"
run hpOc64   3 "--out-channel 64"
run hpOc128  4 "--out-channel 128"
run hpWs4    5 "--window-size 4"
run hpWs16   6 "--window-size 16"
run hpAux001 7 "--aux-weight 0.001"
wave

# ---- Wave 3 (1) : aux ----
run hpAux1   0 "--aux-weight 0.1"
wave
echo "ALL_SWEEP_DONE"
