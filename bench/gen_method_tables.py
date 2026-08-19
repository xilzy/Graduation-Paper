"""Emit the corrected 3-task core-9 metric table (markdown) for each method,
reading the (now RGB-protocol-corrected) leaderboards. irvis = gray (unchanged),
medical & gfp_pc = rgb2gray(RGB-final). Used to refresh the EXP/summary docs.
"""
import os, sys
import pandas as pd

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
CORE = ["EN", "MI", "SD", "SF", "AG", "SSIM", "MS_SSIM", "Qabf", "VIF"]
DIAG = ["SCD", "Nabf", "CC"]
TASKS = ["irvis", "medical", "gfp_pc"]


def lb(task):
    return pd.read_csv(os.path.join(BENCH, "reports", task, "leaderboard.csv")).set_index("method")


def main():
    method = sys.argv[1]
    tbls = {t: lb(t) for t in TASKS}
    cols = CORE + DIAG
    print(f"### {method}")
    print("| task | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for t in TASKS:
        df = tbls[t]
        if method not in df.index:
            continue
        row = df.loc[method]
        n = int(row["n"]) if "n" in row else "?"
        vals = " | ".join(f"{row[c]:.3f}" if c in row and pd.notna(row[c]) else "-" for c in cols)
        print(f"| {t} | {n} | {vals} |")


if __name__ == "__main__":
    main()
