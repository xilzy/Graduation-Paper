"""Print per-task core-9 average-rank leaderboard from reports/<task>/*__means.csv,
highlighting given methods. Quick view while iterating.

Usage: rank_view.py [--mark Ours-MoE R1_maxfuse_oc48 ...]
"""
import os, glob, csv, argparse
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
CORE = ["EN", "MI", "SD", "SF", "AG", "SSIM", "MS_SSIM", "Qabf", "VIF"]  # all higher=better

ap = argparse.ArgumentParser(); ap.add_argument("--mark", nargs="*", default=[])
args = ap.parse_args()

for task in ["irvis", "medical", "gfp_pc"]:
    rows = {}
    for f in glob.glob(f"{BENCH}/reports/{task}/*__means.csv"):
        m = os.path.basename(f).split("__")[0]
        r = next(csv.DictReader(open(f)))
        rows[m] = {k: float(r[k]) for k in CORE}
    if not rows:
        continue
    methods = list(rows)
    # rank each metric (1=best, higher better)
    rk = {m: 0.0 for m in methods}
    for k in CORE:
        order = sorted(methods, key=lambda m: rows[m][k], reverse=True)
        for i, m in enumerate(order, 1):
            rk[m] += i
    avg = {m: rk[m] / len(CORE) for m in methods}
    print(f"\n=== {task} (n_methods={len(methods)}) — core-9 AvgRank (lower=better) ===")
    for i, m in enumerate(sorted(methods, key=lambda m: avg[m]), 1):
        star = "  <<<" if m in args.mark else ""
        print(f"  {i:2d}. {m:16s} {avg[m]:5.2f}  SD={rows[m]['SD']:6.1f} SF={rows[m]['SF']:6.2f} "
              f"EN={rows[m]['EN']:.2f} SSIM={rows[m]['SSIM']:.3f} Qabf={rows[m]['Qabf']:.3f} MI={rows[m]['MI']:.3f}{star}")
