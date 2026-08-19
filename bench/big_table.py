"""Emit the full comparison big-table (markdown): every metric, Ours + 18 methods,
per task, with Ours's rank and bold #1. Also per-modality strong-metric counts.

Usage: big_table.py [--ours Ours] > table.md
"""
import glob, os, csv, argparse
B = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/reports"
PUB = ["CDDFuse", "SeAFusion", "SwinFusion", "PIAFusion", "IFCNN", "DenseFuse",
       "DATFuse", "LP", "DTCWT", "NSCT*", "DWT", "TarDAL", "RFN-Nest", "LRRNet",
       "DDFM", "MURF", "U2Fusion", "GTF"]
# columns shown (direction: 1=higher better, 0=lower)
COLS = [("EN", 1), ("MI", 1), ("SD", 1), ("SF", 1), ("AG", 1), ("SSIM", 1),
        ("MS_SSIM", 1), ("Qabf", 1), ("VIF", 1), ("SCD", 1), ("CC", 1), ("Nabf", 0)]

ap = argparse.ArgumentParser(); ap.add_argument("--ours", default="Ours")
a = ap.parse_args(); OURS = a.ours


def load(t):
    r = {}
    for f in glob.glob(f"{B}/{t}/*__means.csv"):
        m = os.path.basename(f).split("__")[0]; r[m] = next(csv.DictReader(open(f)))
    return r


def fmt(v, k):
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


print(f"# 大表：我们的方法（{OURS}）vs 18 个对比方法 · 全指标\n")
print("约定：↑ 越大越好（Nabf ↓）。**加粗=全场第一**；Ours 名次见末列。\n")
summary = {}
for t, tn in [("irvis", "irvis 红外-可见光 (n=50)"), ("medical", "medical 医学 (n=48)"),
              ("gfp_pc", "gfp_pc 显微 (n=30)")]:
    rows = load(t)
    methods = [m for m in PUB if m in rows] + ([OURS] if OURS in rows else [])
    # rank of OURS per col
    best = {}
    for k, hi in COLS:
        vals = {m: float(rows[m][k]) for m in methods if k in rows[m] and rows[m][k] not in ("", "nan")}
        best[k] = (max(vals.values()) if hi else min(vals.values())) if vals else None
    print(f"\n## {tn}\n")
    hdr = "| 方法 | " + " | ".join(k for k, _ in COLS) + " |"
    print(hdr); print("|" + "---|" * (len(COLS) + 1))
    for m in methods:
        cells = []
        for k, hi in COLS:
            if k not in rows[m] or rows[m][k] in ("", "nan"):
                cells.append("-"); continue
            v = float(rows[m][k]); s = fmt(v, k)
            if best[k] is not None and abs(v - best[k]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        tag = " **(Ours)**" if m == OURS else ""
        print(f"| {m}{tag} | " + " | ".join(cells) + " |")
    # Ours strong-metric summary
    if OURS in rows:
        t3 = t5 = n1 = 0
        for k, hi in COLS:
            vals = {m: float(rows[m][k]) for m in methods if k in rows[m] and rows[m][k] not in ("", "nan")}
            rank = sorted(vals.values(), reverse=bool(hi)).index(vals[OURS]) + 1
            if rank == 1: n1 += 1
            if rank <= 3: t3 += 1
            if rank <= 5: t5 += 1
        summary[t] = (n1, t3, t5)
        print(f"\n> Ours 本任务：**全场第一 {n1}** 个 · Top-3 {t3} 个 · Top-5 {t5} 个（共 {len(COLS)} 指标）")
print("\n## 汇总（Ours 强指标数 / 模态）\n")
print("| 模态 | #1 | Top-3 | Top-5 |\n|---|---|---|---|")
for t in ["irvis", "medical", "gfp_pc"]:
    if t in summary:
        n1, t3, t5 = summary[t]; print(f"| {t} | {n1} | {t3} | {t5} |")
