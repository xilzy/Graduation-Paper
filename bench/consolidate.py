"""Consolidate all methods' per-task leaderboards into a cross-method comparison.

For each task, read fusion_bench/reports/<task>/leaderboard.csv (one row per method,
mean metrics), compute an average-rank over the core 9 metrics (lower rank=better,
direction-aware via metrics.HIGHER_IS_BETTER), and emit:
  fusion_bench/reports/<task>/comparison.csv   (means + per-metric rank + AvgRank, sorted)
  fusion_bench/reports/COMPARISON.md           (markdown tables for all tasks)
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
sys.path.insert(0, ROOT)
import metrics as M

CORE = ["EN", "MI", "SD", "SF", "AG", "SSIM", "MS_SSIM", "Qabf", "VIF"]
DIAG = ["SCD", "Nabf", "CC"]
TASKS = ["irvis", "medical", "gfp_pc"]
HIB = M.HIGHER_IS_BETTER


def to_md(tbl):
    cols = list(tbl.columns)
    lines = ["| method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in tbl.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def rank_table(df):
    df = df.copy()
    cols = [c for c in CORE if c in df.columns]
    ranks = pd.DataFrame(index=df.index)
    for c in cols:
        asc = not HIB.get(c, True)   # higher-is-better -> rank descending (best=1)
        ranks[c] = df[c].rank(ascending=asc, method="average")
    df["AvgRank"] = ranks[cols].mean(axis=1)
    return df.sort_values("AvgRank")


def main():
    md = ["# 对比方法综合比较（16 方法 × 3 模态）\n",
          "聚合方式：核心 9 指标（EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF）的平均排名（方向感知，越低越好）。",
          "数据来源：各 `fusion_bench/reports/<task>/leaderboard.csv`。\n"]
    for task in TASKS:
        lb = os.path.join(BENCH, "reports", task, "leaderboard.csv")
        if not os.path.exists(lb):
            md.append(f"## {task}\n(no leaderboard)\n"); continue
        df = pd.read_csv(lb)
        # drop trivial detectors (names starting with "_") from the method ranking
        df = df[~df["method"].astype(str).str.startswith("_")].set_index("method")
        ranked = rank_table(df)
        ranked.to_csv(os.path.join(BENCH, "reports", task, "comparison.csv"))
        show = [c for c in (CORE + DIAG) if c in ranked.columns] + ["AvgRank"]
        tbl = ranked[show].round(3)
        md.append(f"## {task}  (n={int(df['n'].iloc[0]) if 'n' in df.columns else '?'})\n")
        md.append(to_md(tbl))
        md.append("")
        md.append(f"**Top-3 (avg-rank):** {', '.join(ranked.index[:3])}\n")
    out = os.path.join(BENCH, "reports", "COMPARISON.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {out}")
    # also echo the ranking order per task
    for task in TASKS:
        cp = os.path.join(BENCH, "reports", task, "comparison.csv")
        if os.path.exists(cp):
            r = pd.read_csv(cp)
            print(f"\n[{task}] methods by AvgRank:")
            print(r[["method", "AvgRank"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
