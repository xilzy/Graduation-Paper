"""Materialize standardized fusion inputs for the 3 benchmark modalities.

For each task (gfp_pc / irvis / medical) and the TEST split, write a dead-simple
contract that every comparison method can consume without knowing anything about
YCbCr, pairing, or color:

  fusion_bench/inputs/<task>/A/<stem>.png   # source A as 8-bit GRAY (Y of color src)
  fusion_bench/inputs/<task>/B/<stem>.png   # source B as 8-bit GRAY
  fusion_bench/inputs/<task>/cbcr/<stem>.npy # (Cb,Cr) of the color source, for RGB recomb
  fusion_bench/inputs/<task>/colorA/<stem>.png # original color source A (RGB) for methods wanting it

Contract for a method: read A/*.png and B/*.png (both grayscale), output one fused
grayscale (or RGB) PNG per stem named <stem>.png into its own fused dir. Then run
bench eval_method.py to score against A/B.

MSRS has 361 test pairs; for tractability AND comparability across deep + slow
traditional methods we use a FIXED evenly-spaced subset (default 50). GFP-PC (30)
and medical (48) use the full test split.
"""
import os, json, argparse
import numpy as np
from PIL import Image

ROOT = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
import sys
sys.path.insert(0, ROOT)
import mm_fusion_data as D
from ycbcr import load_ycbcr

TASKS = {
    "gfp_pc":  (os.path.join(ROOT, "configs/gfp_pc.json"),  None),
    "irvis":   (os.path.join(ROOT, "configs/irvis_msrs.json"), 50),
    "medical": (os.path.join(ROOT, "configs/medical_harvard.json"), None),
}


def subset(pairs, k):
    if k is None or k >= len(pairs):
        return pairs
    n = len(pairs)
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [pairs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()

    manifest = {}
    for task in args.tasks:
        cfg_path, cap = TASKS[task]
        cfg = D.load_config(cfg_path)
        pairs = D.list_pairs(cfg, "test", root=ROOT)
        pairs = subset(pairs, cap)
        color_a = cfg.get("color_a", cfg.get("color_source") == "a")
        color_b = cfg.get("color_b", False)

        base = os.path.join(BENCH, "inputs", task)
        for sub in ("A", "B", "cbcr", "colorA"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)

        stems = []
        for stem, a_path, b_path in pairs:
            ya, cba, cra = load_ycbcr(a_path)
            yb, cbb, crb = load_ycbcr(b_path)
            Image.fromarray(ya.clip(0, 255).astype("uint8")).save(os.path.join(base, "A", stem + ".png"))
            Image.fromarray(yb.clip(0, 255).astype("uint8")).save(os.path.join(base, "B", stem + ".png"))
            # chroma comes from whichever source is color (for RGB recomb at viz time)
            if color_a:
                np.save(os.path.join(base, "cbcr", stem + ".npy"), np.stack([cba, cra]))
                Image.open(a_path).convert("RGB").save(os.path.join(base, "colorA", stem + ".png"))
            elif color_b:
                np.save(os.path.join(base, "cbcr", stem + ".npy"), np.stack([cbb, crb]))
                Image.open(b_path).convert("RGB").save(os.path.join(base, "colorA", stem + ".png"))
            stems.append(stem)
        manifest[task] = {"n": len(stems), "stems": stems,
                          "color_source": "a" if color_a else ("b" if color_b else None)}
        print(f"[{task}] wrote {len(stems)} pairs -> {base}")

    os.makedirs(BENCH, exist_ok=True)
    with open(os.path.join(BENCH, "inputs", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {os.path.join(BENCH, 'inputs', 'manifest.json')}")


if __name__ == "__main__":
    main()
