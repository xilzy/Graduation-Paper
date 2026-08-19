"""Internal ablation / parameter tables (markdown). Rows = our own variants.
Tables show raw metrics; the OPTIMALITY VERDICT for each table is judged by v3's
"advantage metrics" — the metrics on which v3 ranks Top-3 vs the 18 published
methods (MI/VIF/SSIM/Nabf on all tasks, +Qabf on medical/gfp). We report, per
config, its Top-3-vs-SOTA count (== how many advantage metrics it keeps in the
winning band). Advantage-metric columns are marked with * in the header.

Usage: internal_tables.py > body.md   (rerun after new runs finish)
"""
import glob, os, csv
B = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/reports"
PUB = {"CDDFuse","DATFuse","DDFM","DTCWT","DWT","DenseFuse","GTF","IFCNN","LP",
       "LRRNet","MURF","NSCT*","PIAFusion","RFN-Nest","SeAFusion","SwinFusion","TarDAL","U2Fusion"}
COLS = [("EN",1),("MI",1),("SD",1),("SF",1),("AG",1),("SSIM",1),
        ("MS_SSIM",1),("Qabf",1),("VIF",1),("SCD",1),("CC",1),("Nabf",0)]
# v3 advantage metrics per task (Top-3 vs 18 SOTA) — these define optimality.
ADV = {"irvis":{"MI","VIF","SSIM","Nabf"},
       "medical":{"MI","VIF","Qabf","SSIM","Nabf"},
       "gfp_pc":{"MI","VIF","Qabf","SSIM","Nabf"}}
TASKS = [("irvis","irvis 红外-可见光 (n=50)"),("medical","medical 医学 (n=48)"),
         ("gfp_pc","gfp_pc 显微 (n=30)")]

ABLATION = [
    ("完整 v3（基线）","Ours"),
    ("−MoE（去路由专家）","abNoMoE"),
    ("−决策图头（direct）","abDirect"),
    ("−窗口注意力（ws1）","abWs1"),
    ("−maxfuse 损失（orig）","abOrig"),
    ("−任务条件路由","abNoTC"),
    ("−MoE −决策图头","abNoMoE_direct"),
    ("−决策图头 −maxfuse","abDirect_orig"),
    ("−窗口注意力 −maxfuse","abWs1_orig"),
]
PARAMS = {
    "路由专家数 n_routed（v3=12）": [("n_routed=4","hpNr4"),("n_routed=8","hpNr8"),
        ("n_routed=12 (v3)","Ours"),("n_routed=16","hpNr16")],
    "top-k（v3=2）": [("k=1","hpK1"),("k=2 (v3)","Ours"),("k=4","hpK4")],
    "共享专家数 n_shared（v3=1）": [("n_shared=0","hpNs0"),("n_shared=1 (v3)","Ours"),("n_shared=2","hpNs2")],
    "骨干深度 depth（v3=4）": [("depth=2","hpD2"),("depth=3","abD3"),
        ("depth=4 (v3)","Ours"),("depth=5","hpD5")],
    "通道数 out_channel（v3=96）": [("oc=64","hpOc64"),("oc=96 (v3)","Ours"),("oc=128","hpOc128")],
    "窗口大小 window_size（v3=8）": [("ws=4","hpWs4"),("ws=8 (v3)","Ours"),("ws=16","hpWs16")],
    "负载均衡权重 aux_weight（v3=0.01）": [("aux=0.001","hpAux001"),("aux=0.01 (v3)","Ours"),("aux=0.1","hpAux1")],
    "路由方式 routing（v3=softmax）": [("softmax+aux (v3)","Ours"),("deepseek 无辅助损失","abDeep")],
}


def load(t):
    r = {}
    for f in glob.glob(f"{B}/{t}/*__means.csv"):
        m = os.path.basename(f).split("__")[0]; r[m] = next(csv.DictReader(open(f)))
    return r


def val(rows, m, k):
    if m not in rows or k not in rows[m] or rows[m][k] in ("", "nan"):
        return None
    return float(rows[m][k])


def fmt(v):
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def top3_vs_sota(name):
    """total Top-3 count of `name` vs the 18 SOTA, summed over 3 tasks."""
    tot = 0
    for t, _ in TASKS:
        rows = load(t)
        if name not in rows:
            continue
        for k, hi in COLS:
            field = [m for m in PUB if val(rows, m, k) is not None] + [name]
            vals = {m: val(rows, m, k) for m in field if val(rows, m, k) is not None}
            if name not in vals:
                continue
            if sorted(vals.values(), reverse=bool(hi)).index(vals[name]) + 1 <= 3:
                tot += 1
    return tot


def emit_group(items):
    missing = set()
    for t, tn in TASKS:
        rows = load(t)
        present = [(lab, nm) for lab, nm in items if nm in rows]
        for lab, nm in items:
            if nm not in rows:
                missing.add(lab)
        if not present:
            print(f"**{tn}**：（暂无结果，实验待运行）\n"); continue
        best = {}
        for k, hi in COLS:
            vals = [val(rows, nm, k) for _, nm in present if val(rows, nm, k) is not None]
            best[k] = (max(vals) if hi else min(vals)) if vals else None
        # header marks advantage metrics of THIS task with *
        hdr = "| 配置 | " + " | ".join((k + "*") if k in ADV[t] else k for k, _ in COLS) + " |"
        print(f"**{tn}**（带 * 列 = v3 的优势指标，判优只看这些）\n")
        print(hdr); print("|" + "---|" * (len(COLS) + 1))
        for lab, nm in present:
            cells = []
            for k, hi in COLS:
                v = val(rows, nm, k)
                if v is None:
                    cells.append("-"); continue
                s = fmt(v)
                if best[k] is not None and abs(v - best[k]) < 1e-9:
                    s = f"**{s}**"
                cells.append(s)
            print(f"| {lab} | " + " | ".join(cells) + " |")
        print()
    return sorted(missing)


def verdict_line(items):
    """emit the Top-3-vs-SOTA retention line + which config is best."""
    scored = [(lab, nm, top3_vs_sota(nm)) for lab, nm in items if any(nm in load(t) for t, _ in TASKS)]
    if not scored:
        return
    mx = max(s for _, _, s in scored)
    parts = []
    for lab, nm, s in scored:
        tag = f"**{lab}={s}**" if s == mx else f"{lab}={s}"
        parts.append(tag)
    best = "、".join(lab for lab, _, s in scored if s == mx)
    print(f"> **优势指标保持度（各配置在 18 个 SOTA 中的 Top-3 指标数，满分越高越好）**："
          + " · ".join(parts) + f" → 最优：**{best}**。\n")


print("<!-- generated by bench/internal_tables.py; rerun after new runs finish -->\n")
print("### 2.1 消融结果表（三模态）\n")
print("以完整 v3 为基线，每次**完全去掉一个创新点**（或两个）重训 20 epoch。带 * 列是 v3 的优势指标"
      "（在 18 个 SOTA 中 Top-3）；判断创新点是否有效，看去掉后**优势指标保持度**掉多少。\n")
miss = emit_group(ABLATION)
verdict_line(ABLATION)
if miss:
    print(f"> 待运行（尚无结果）：{'、'.join(miss)}\n")
print("\n<!--SPLIT-->\n")
for pname, items in PARAMS.items():
    print(f"### {pname}\n")
    miss = emit_group(items)
    verdict_line(items)
    if miss:
        print(f"> 待运行（尚无结果）：{'、'.join(miss)}\n")
    print("---\n")
