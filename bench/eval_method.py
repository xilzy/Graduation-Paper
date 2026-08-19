"""Score one method's fused outputs for one task against standardized A/B inputs.

Usage:
  python eval_method.py --task irvis --name CDDFuse --fused-dir /path/to/fused/irvis

Reads fusion_bench/inputs/<task>/{A,B}/<stem>.png and the method's fused <stem>.png,
computes the full metric panel (metrics.compute_all) per image, writes:
  fusion_bench/reports/<task>/<name>__per_image.csv
  fusion_bench/reports/<task>/<name>__means.csv
and appends a row to fusion_bench/reports/<task>/leaderboard.csv
Prints the per-method means so a sub-agent sees the numbers immediately.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

ROOT = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
sys.path.insert(0, ROOT)
import metrics as M


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def find_fused(d, stem):
    for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _resize_to(X, ref):
    if X.shape != ref.shape:
        return np.asarray(Image.fromarray(X.astype("uint8")).resize(
            (ref.shape[1], ref.shape[0]), Image.BILINEAR), dtype=np.float64)
    return X


def _job(args):
    stem, a_path, b_path, f_path, func_source = args
    A = load_gray(a_path); B = load_gray(b_path); F = load_gray(f_path)
    B = _resize_to(B, A); F = _resize_to(F, A)
    vals = M.compute_all(A, B, F)
    if func_source in ("a", "b"):
        fy = A if func_source == "a" else B
        vals["FuncCorr"] = M.func_corr(fy, F)
        vals["FuncSal"] = M.func_sal(fy, F)
    vals["stem"] = stem
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fused-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    man = json.load(open(os.path.join(BENCH, "inputs", "manifest.json")))
    info = man[args.task]
    func_source = info.get("func_source")
    # func axis source per task: gfp_pc=a, medical=a, irvis=b  (from configs)
    func_map = {"gfp_pc": "a", "medical": "a", "irvis": "b"}
    func_source = func_map.get(args.task)

    a_dir = os.path.join(BENCH, "inputs", args.task, "A")
    b_dir = os.path.join(BENCH, "inputs", args.task, "B")
    jobs, miss = [], 0
    for stem in info["stems"]:
        f = find_fused(args.fused_dir, stem)
        if f is None:
            miss += 1; continue
        jobs.append((stem, os.path.join(a_dir, stem + ".png"),
                     os.path.join(b_dir, stem + ".png"), f, func_source))
    if miss:
        print(f"[warn] {args.name}/{args.task}: {miss}/{len(info['stems'])} fused images MISSING")
    if not jobs:
        print(f"[error] no fused images found in {args.fused_dir}"); sys.exit(2)

    with Pool(args.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    df = pd.DataFrame(rows)
    outdir = os.path.join(BENCH, "reports", args.task)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{args.name}__per_image.csv"), index=False)

    mcols = [c for c in df.columns if c != "stem"]
    means = df[mcols].mean()
    means_row = means.to_dict()
    means_row["method"] = args.name
    means_row["n"] = len(jobs)
    pd.DataFrame([means_row]).to_csv(os.path.join(outdir, f"{args.name}__means.csv"), index=False)

    # append/update leaderboard
    lb_path = os.path.join(outdir, "leaderboard.csv")
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
        lb = lb[lb["method"] != args.name]
    else:
        lb = pd.DataFrame()
    lb = pd.concat([lb, pd.DataFrame([means_row])], ignore_index=True)
    front = ["method", "n"]
    lb = lb[front + [c for c in lb.columns if c not in front]]
    lb.to_csv(lb_path, index=False)

    core = [c for c in M.CORE_METRICS if c in means.index]
    diag = [c for c in ("SCD", "Nabf", "CC", "Qabf") if c in means.index]
    extra = [c for c in ("FuncCorr", "FuncSal") if c in means.index]
    pd.set_option("display.width", 200)
    print(f"\n=== {args.name} / {args.task}  (n={len(jobs)}) ===")
    print(means[core + diag + extra].round(4).to_string())
    print(f"\nleaderboard -> {lb_path}")


if __name__ == "__main__":
    main()
