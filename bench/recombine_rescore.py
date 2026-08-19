"""Fix the RGB metric protocol for output_mode=rgb tasks (medical, gfp_pc).

The comparison methods produced a fused luminance (Y). The repo + MATLAB main.m
score the FINAL fused image, which for color tasks is the RGB recombination of
fused-Y with the source CbCr (infer_fusion.py: ycbcr_to_rgb(Y, fuse_chroma(...))),
then rgb2gray'd (== PIL 'L'). The uint8 clip in ycbcr_to_rgb shifts metrics in
saturated-color regions (PET/SPECT pseudocolor, GFP green), so scoring the bare Y
is wrong for these two tasks.

This script, for each method and each rgb task:
  1. reads the method's fused-Y png  fusion_bench/fused/<M>/<task>/<stem>.png
  2. reads the color source's CbCr    fusion_bench/inputs/<task>/cbcr/<stem>.npy
  3. builds RGB = ycbcr_to_rgb(Y, fuse_chroma(cba, neutral), fuse_chroma(cra, neutral))
     (B is grayscale so its chroma is neutral 128 -> fused chroma == source A chroma,
      exactly as infer_fusion.py computes it)
  4. saves the RGB final image to fusion_bench/fused_final/<M>/<task>/<stem>.png

Then eval_method.py is run on fused_final/ (its load_gray = rgb2gray) to refresh
the leaderboard rows for these tasks. irvis is output_mode=gray and is untouched.
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

ROOT = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
sys.path.insert(0, ROOT)
import ycbcr  # repo's YCbCr helpers (BT.601, same as MATLAB)

RGB_TASKS = ["medical", "gfp_pc"]
NEUTRAL = 128.0


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def recombine_method(method, tasks):
    man = json.load(open(os.path.join(BENCH, "inputs", "manifest.json")))
    done = {}
    for task in tasks:
        stems = man[task]["stems"]
        fused_dir = os.path.join(BENCH, "fused", method, task)
        cbcr_dir = os.path.join(BENCH, "inputs", task, "cbcr")
        out_dir = os.path.join(BENCH, "fused_final", method, task)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for stem in stems:
            fp = os.path.join(fused_dir, stem + ".png")
            cp = os.path.join(cbcr_dir, stem + ".npy")
            if not (os.path.exists(fp) and os.path.exists(cp)):
                continue
            y = load_gray(fp)                       # method's fused luminance
            cb, cr = np.load(cp)                     # color source's Cb, Cr (A's size)
            if cb.shape != y.shape:                  # defensive: align chroma to fused size
                cb = np.asarray(Image.fromarray(cb.astype("uint8")).resize(
                    (y.shape[1], y.shape[0]), Image.BILINEAR), dtype=np.float64)
                cr = np.asarray(Image.fromarray(cr.astype("uint8")).resize(
                    (y.shape[1], y.shape[0]), Image.BILINEAR), dtype=np.float64)
            # B is grayscale -> neutral chroma; fuse_chroma(c, neutral) == c
            cbf = ycbcr.fuse_chroma(cb, np.full_like(cb, NEUTRAL))
            crf = ycbcr.fuse_chroma(cr, np.full_like(cr, NEUTRAL))
            rgb = ycbcr.ycbcr_to_rgb(y, cbf, crf)    # uint8, clipped (== infer_fusion.py)
            Image.fromarray(rgb).save(os.path.join(out_dir, stem + ".png"))
            n += 1
        done[task] = n
        print(f"[recombine] {method}/{task}: {n} RGB-final images -> {out_dir}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--tasks", nargs="+", default=RGB_TASKS)
    args = ap.parse_args()
    recombine_method(args.method, args.tasks)
