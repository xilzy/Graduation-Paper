"""Generate naive classical fusion baselines (per-pixel average / maximum).

These are cheap, always-available references so the S1 average-rank table has a
meaningful yardstick from day one (a learned method should beat naive avg/max
on most metrics).

Usage:
  venv/bin/python make_baselines.py --config configs/gfp_pc.json --set all \
      --outroot outputs
"""
import os
import argparse
import numpy as np
from PIL import Image

import fusion_data


def load_gray(path, ref_shape=None):
    img = Image.open(path).convert("L")
    if ref_shape is not None and img.size != (ref_shape[1], ref_shape[0]):
        img = img.resize((ref_shape[1], ref_shape[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", default="all", choices=["all", "probe"])
    ap.add_argument("--outroot", default="outputs")
    args = ap.parse_args()

    cfg = fusion_data.load_config(args.config)
    pairs = fusion_data.select(cfg, args.set)
    task = cfg["task"]
    out_avg = os.path.join(args.outroot, f"avg_{task}")
    out_max = os.path.join(args.outroot, f"max_{task}")
    os.makedirs(out_avg, exist_ok=True)
    os.makedirs(out_max, exist_ok=True)

    for stem, a_path, b_path in pairs:
        A = load_gray(a_path)
        B = load_gray(b_path, ref_shape=A.shape)
        favg = np.clip(np.round((A + B) / 2), 0, 255).astype(np.uint8)
        fmax = np.clip(np.round(np.maximum(A, B)), 0, 255).astype(np.uint8)
        Image.fromarray(favg).save(os.path.join(out_avg, stem + ".png"))
        Image.fromarray(fmax).save(os.path.join(out_max, stem + ".png"))
    print(f"avg -> {out_avg}\nmax -> {out_max}\n{len(pairs)} pairs each")


if __name__ == "__main__":
    main()
