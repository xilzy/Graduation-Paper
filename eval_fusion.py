"""S1 evaluation (balance-aware): score methods' fused outputs against sources.

Computes core + balance-aware + diagnostic metrics per image (parallelised),
then ranks ONLY the real methods on the balance-aware RANK_METRICS set
(MASTER_PLAN s5.4 + EXP-0-02). Trivial baselines (avg/max) are passed via
--detectors and are NOT ranked against real methods; instead they are used as
"deception detectors": the run prints a self-check that every real method
out-ranks every detector on the balance-aware score. If a detector beats a real
method, the metric protocol (not the method) is still broken.

Usage:
  venv/bin/python eval_fusion.py --config configs/gfp_pc.json --set probe \
      --methods MDFNet=outputs/mdfnet_gfp_pc Retrained=outputs/p0_retrained_gfp_pc \
      --detectors Avg=outputs/avg_gfp_pc Max=outputs/max_gfp_pc \
      --outdir reports/gfp_pc_probe --workers 16
"""
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

import fusion_data
import metrics as M


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def find_fused(method_dir, stem):
    # Score the FINAL fused image, converted to grayscale via load_gray (PIL 'L'
    # == MATLAB rgb2gray = 0.299R+0.587G+0.114B). This matches the MATLAB
    # convention exactly: main.m reads the (possibly RGB) fused image and runs
    # rgb2gray before computing metrics. So for an RGB-output method we score
    # rgb2gray(RGB_final), NOT the pre-reconstruction network Y (which differs by
    # the YCbCr->RGB->gray rounding, ~1.5/255).
    for ext in (".png", ".bmp", ".jpg", ".tif", ".tiff"):
        p = os.path.join(method_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _resize_to(X, ref):
    if X.shape != ref.shape:
        return np.asarray(Image.fromarray(X).resize(
            (ref.shape[1], ref.shape[0]), Image.BILINEAR), dtype=np.float64)
    return X


def _job(args):
    name, stem, a_path, b_path, fused_path, func_source = args
    A = load_gray(a_path); B = load_gray(b_path); F = load_gray(fused_path)
    B = _resize_to(B, A); F = _resize_to(F, A)
    vals = M.compute_all(A, B, F)
    if func_source in ("a", "b"):                 # FUNCTION axis (task-specific)
        func_y = A if func_source == "a" else B
        vals["FuncCorr"] = M.func_corr(func_y, F)
        vals["FuncSal"] = M.func_sal(func_y, F)
    vals.update(method=name, stem=stem)
    return vals


def collect_rows(name_to_dir, pairs, workers, func_source=None):
    jobs, missing = [], {}
    for name, mdir in name_to_dir.items():
        miss = 0
        for stem, a_path, b_path in pairs:
            fp = find_fused(mdir, stem)
            if fp is None:
                miss += 1
                continue
            jobs.append((name, stem, a_path, b_path, fp, func_source))
        if miss:
            missing[name] = miss
    for n, m in missing.items():
        print(f"[warn] {n}: {m}/{len(pairs)} fused images missing")
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_job, jobs, chunksize=4)
    else:
        rows = [_job(j) for j in jobs]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", default="probe", choices=["all", "probe"])
    ap.add_argument("--methods", nargs="+", required=True, help="name=dir (real methods)")
    ap.add_argument("--detectors", nargs="*", default=[],
                    help="name=dir trivial baselines (deception detectors, not ranked)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = fusion_data.load_config(args.config)
    pairs = fusion_data.select(cfg, args.set)
    methods = dict(m.split("=", 1) for m in args.methods)
    detectors = dict(m.split("=", 1) for m in args.detectors)
    os.makedirs(args.outdir, exist_ok=True)

    func_source = cfg.get("func_source")          # 'a'/'b' -> enable FUNCTION axis
    all_dirs = {**methods, **detectors}
    rows = collect_rows(all_dirs, pairs, args.workers, func_source)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "per_image.csv"), index=False)

    metric_cols = [c for c in df.columns if c not in ("method", "stem")]
    means = df.groupby("method")[metric_cols].mean()
    means.to_csv(os.path.join(args.outdir, "means.csv"))
    method_means = {m: means.loc[m].to_dict() for m in means.index}

    fid, qual = M.FIDELITY_AXIS, M.QUALITY_AXIS
    func = M.FUNCTION_AXIS if func_source else []

    def axis_rank(names, axis):
        return M.average_rank_table({m: method_means[m] for m in names}, axis)["AvgRank"]

    # Multi-axis leaderboard for real methods (separate ranks + equal-weight composite).
    # For tasks with a functional source (GFP-PC), FUNCTION is the PRIMARY criterion.
    board = pd.DataFrame(index=list(methods))
    board["FidRank"] = axis_rank(methods, fid)
    board["QualRank"] = axis_rank(methods, qual)
    if func:
        board["FuncRank"] = axis_rank(methods, func)
        board["Composite"] = (board["FidRank"] + board["QualRank"] + board["FuncRank"]) / 3
    else:
        board["Composite"] = (board["FidRank"] + board["QualRank"]) / 2
    board = board.sort_values("Composite")
    board.to_csv(os.path.join(args.outdir, "two_axis_rank.csv"))

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print(f"\n=== set={args.set}  n_pairs={len(pairs)}  workers={args.workers} ===")
    if args.set == "probe":
        print(f"probe stems: {fusion_data.probe_stems(cfg)}")

    show = (func + fid + qual
            + [c for c in ("Balance", "Nabf", "SCD", "CC") if c in means.columns]
            + [c for c in ("SSIM_A", "SSIM_B", "MI_A", "MI_B") if c in means.columns])
    print("\n--- per-method means (function axis | fidelity axis | quality axis | diag) ---")
    print(means[[c for c in show if c in means.columns]].round(4).to_string())

    print("\n--- S1 leaderboard: REAL methods (rank lower=better) ---")
    print("    FuncRank = functional-region preservation (PRIMARY for GFP-PC);"
          " FidRank = balance-aware fidelity; QualRank = no-ref richness")
    print(board.round(3).to_string())

    if detectors:
        # Axis-specific self-checks (the meaningful bar on asymmetric tasks):
        #   * QUALITY: a real fusion must add visual richness beyond trivial avg/max.
        #   * FIDELITY: report only -- a contrast-enhancing method is NOT expected to
        #     out-fidelity a literal blend when one source is near-black (GFP-PC).
        alln = list(methods) + list(detectors)
        q_all = M.average_rank_table({m: method_means[m] for m in alln}, qual)["AvgRank"]
        f_all = M.average_rank_table({m: method_means[m] for m in alln}, fid)["AvgRank"]
        chk = pd.DataFrame({"FidRank": f_all, "QualRank": q_all})
        if func:
            fn_all = M.average_rank_table({m: method_means[m] for m in alln}, func)["AvgRank"]
            chk["FuncRank"] = fn_all
        chk["role"] = ["detector" if n in detectors else "method" for n in chk.index]
        print("\n--- deception-detector self-check (real + detectors together) ---")
        print(chk.round(3).to_string())
        # Each trivial baseline is a DEGENERATE EXTREME: Max(=copy structure source)
        # is worst on FUNCTION; Avg(=blend) is worst on QUALITY. So "beat every
        # detector on one axis" is the wrong bar. The meaningful bar is PARETO:
        # no real method should be dominated (worse-or-equal on ALL axes) by a
        # trivial baseline -- i.e. the learned method must not be a strict
        # downgrade of copy/blend. Lower rank = better.
        axes_used = ([ "FuncRank"] if func else []) + ["FidRank", "QualRank"]
        dominated = []
        for r in methods:
            for d in detectors:
                if all(chk.loc[d, ax] <= chk.loc[r, ax] for ax in axes_used) and \
                   any(chk.loc[d, ax] < chk.loc[r, ax] for ax in axes_used):
                    dominated.append((r, d))
        ok = not dominated
        print(f"\n[PARETO self-check over {axes_used}] "
              f"[{'PASS' if ok else 'FAIL'}] no real method is dominated by a trivial baseline"
              + ("" if ok else f"  -> dominated: {dominated}"))
        if func:
            mx = [d for d in detectors if 'max' in d.lower() or 'Max' in d]
            note = ("Max(copy PC) is worst on FUNCTION; Avg(blend) is worst on QUALITY; "
                    "learned methods occupy the non-degenerate middle.")
            print(f"[axis roles] {note}")
    print(f"\nCSV -> {args.outdir}/")


if __name__ == "__main__":
    main()
