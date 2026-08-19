"""Append a clearly-marked metric-revision section to each EXP-CMP doc, with the
method's RGB-protocol-corrected 3-task table read from the updated leaderboards.

Protocol fix: for output_mode=rgb tasks (medical, gfp_pc) the FINAL fused image is
the RGB recombination of fused-Y with the source CbCr (repo infer_fusion.py), and
metrics are computed on rgb2gray(RGB-final) -- matching MATLAB evaluation/main.m,
which rgb2grays any RGB fused image before scoring. irvis is output_mode=gray
(consistent with how MDFNet itself is scored) and is unchanged.
"""
import os
import pandas as pd

MD = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
CORE = ["EN", "MI", "SD", "SF", "AG", "SSIM", "MS_SSIM", "Qabf", "VIF"]
DIAG = ["SCD", "Nabf", "CC"]
TASKS = ["irvis", "medical", "gfp_pc"]
MARK = "## 指标修订（RGB-final 协议，2026-06-28）"

DOC_METHODS = {
    "EXP-CMP-01-CDDFuse.md": ["CDDFuse"],
    "EXP-CMP-02-SwinFusion.md": ["SwinFusion"],
    "EXP-CMP-03-U2Fusion.md": ["U2Fusion"],
    "EXP-CMP-04-DenseFuse.md": ["DenseFuse"],
    "EXP-CMP-05-IFCNN.md": ["IFCNN"],
    "EXP-CMP-06-SeAFusion.md": ["SeAFusion"],
    "EXP-CMP-07-TarDAL.md": ["TarDAL"],
    "EXP-CMP-08-PIAFusion.md": ["PIAFusion"],
    "EXP-CMP-09-traditional.md": ["LP", "DWT", "DTCWT", "GTF", "NSCT*"],
    "EXP-CMP-10-RFN-Nest.md": ["RFN-Nest"],
    "EXP-CMP-11-LRRNet.md": ["LRRNet"],
    "EXP-CMP-12-DATFuse.md": ["DATFuse"],
    "EXP-CMP-13-DDFM.md": ["DDFM"],
    "EXP-CMP-14-MURF.md": ["MURF"],
}

LB = {t: pd.read_csv(os.path.join(BENCH, "reports", t, "leaderboard.csv")).set_index("method")
      for t in TASKS}


def table_for(method):
    cols = CORE + DIAG
    lines = ["| 任务 | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for t in TASKS:
        df = LB[t]
        if method not in df.index:
            lines.append(f"| {t} | - | (缺) |")
            continue
        r = df.loc[method]
        n = int(r["n"]) if "n" in r and pd.notna(r["n"]) else "?"
        vals = " | ".join(f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "-" for c in cols)
        lines.append(f"| {t} | {n} | {vals} |")
    return "\n".join(lines)


def main():
    note = (
        f"\n\n{MARK}\n\n"
        "> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 "
        "原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，"
        "计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区"
        "（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。\n>\n"
        "> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → "
        "`rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`"
        "（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。\n\n"
        "修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：\n\n"
    )
    for doc, methods in DOC_METHODS.items():
        path = os.path.join(MD, doc)
        if not os.path.exists(path):
            print(f"[skip] missing {doc}")
            continue
        txt = open(path).read()
        if MARK in txt:
            print(f"[skip] already revised {doc}")
            continue
        body = note + "\n\n".join(f"**{m}**\n\n{table_for(m)}" for m in methods) + "\n"
        with open(path, "a") as f:
            f.write(body)
        print(f"[revised] {doc}  ({', '.join(methods)})")


if __name__ == "__main__":
    main()
