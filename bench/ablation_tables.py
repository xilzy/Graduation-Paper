"""Per-experiment big tables (markdown) for the ablation/hyperparameter study.

For each variant, emit the three-modality full-metric tables (18 published
methods + Ours(v3) + the variant), so the data itself shows how the variant
ranks vs v3 vs the field. bold = best among the rows shown; each variant/v3
row is annotated with its Top-3 count computed vs the 18 published methods.

Usage: ablation_tables.py > ablation_tables.md
"""
import glob, os, csv
B = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/reports"
PUB = ["CDDFuse", "SeAFusion", "SwinFusion", "PIAFusion", "IFCNN", "DenseFuse",
       "DATFuse", "LP", "DTCWT", "NSCT*", "DWT", "TarDAL", "RFN-Nest", "LRRNet",
       "DDFM", "MURF", "U2Fusion", "GTF"]
COLS = [("EN", 1), ("MI", 1), ("SD", 1), ("SF", 1), ("AG", 1), ("SSIM", 1),
        ("MS_SSIM", 1), ("Qabf", 1), ("VIF", 1), ("SCD", 1), ("CC", 1), ("Nabf", 0)]
TASKS = [("irvis", "irvis 红外-可见光 (n=50)"), ("medical", "medical 医学 (n=48)"),
         ("gfp_pc", "gfp_pc 显微 (n=30)")]

# (variant_report_name, variant_display, change, conclusion)
EXPS = [
    ("abNoMoE", "−MoE (只留共享专家)", "去掉 12 个路由专家，只保留 1 个共享专家",
     "去掉路由专家后 irvis 掉 1 个 Top-3（丢 SSIM/Qabf；MI/SD 反略升但不进 Top-3），medical/gfp 保持 5/5。**结论：MoE 带来 irvis 净 +1 且不牺牲其他模态**——MoE 的价值需在“决策图头+真实注意力+细粒度专家”组合下才释放（修正早期“MoE 不如 shared”的观察）。MoE 是核心创新，贡献虽为边际但方向明确为正，且与专家数（n_routed）强联动。"),
    ("abDirect", "−决策图头 (direct)", "blend 决策图头 → 网络直接回归融合图",
     "去掉 blend 后**三模态全面崩塌**：irvis MI 5.20→3.55、SD 39.9→22.8、Qabf 0.646→0.377；gfp MI 5.45→3.91、SD 25.3→15.9。**结论：决策图头 F=w·A+(1−w)·B 把融合图锚定为两源凸组合**，无条件继承动态范围（SD/EN）并线性保留双源互信息（MI/VIF/SCD）；直接回归缺此约束会被拉向低对比度均值图。**这是本方法从中游跃入前列的根本原因，是论文核心卖点（消融 −10，最决定性）。**"),
    ("abWs1", "−窗口注意力 (ws1)", "窗口 ws8 → ws1（退化为逐像素、无空间注意力）",
     "ws8→ws1 后 **irvis 直接归零**：MI 5.20→2.76、Qabf 0.646→0.480；AG/Nabf 升高实为噪声增加（Nabf 0.026→0.096，伪影翻 3.7×）。**结论：窗口注意力提供的跨窗空间上下文对 IR-VIS 的互信息与边缘保真是必需的**；去掉后融合退化为局部点运算，MI 腰斩、伪影激增。medical 结构本身强故受影响小，irvis 完全依赖它。"),
    ("abD3", "−深度 (depth3)", "骨干 depth4 → depth3",
     "depth4→depth3 后 irvis 丢 SCD/VIF、medical 丢 1。**结论：更深骨干提升表示能力，是 v3 相对 v1 把 irvis 从 3 提到 4 的来源**（已验证加宽到 oc128 反降）。“深度”是有效的容量扩展方向，“宽度”不是。"),
    ("abOrig", "−maxfuse (orig 损失)", "maxfuse 损失 → 原 GFP 标定 orig 损失",
     "换回 orig 损失后 **medical 高频指标崩塌**：SF 26.7→11.0、AG 9.9→4.1、Qabf 0.691→0.201；gfp MI 5.45→2.06。**结论：maxfuse 让强度/SSIM/梯度朝 per-pixel max(A,B) 对齐，显式取双源中更强的结构响应，直接支撑 SF/AG/Qabf**；orig 损失把目标拉向均值、丢失细节。对 medical/gfp 这类“两源各有强结构”的任务尤为关键（消融 −5）。"),
    ("abDeep", "路由=deepseek", "softmax+aux 路由 → DeepSeek 无辅助损失路由",
     "DeepSeek 无辅助损失路由在 medical 上与 softmax 近乎持平（Qabf 甚至略高 0.698>0.691），但整体 Top-3 计数少 2。**结论：DeepSeek 路由不掉性能但也无增益**，在本小数据/小模型规模上 softmax+Switch 辅助损失已足够均衡；保留为可选消融。论文可述为“更复杂的无辅助损失路由在此规模无额外收益”。"),
    ("abNoTC", "−任务条件路由", "路由不再注入任务嵌入 (task_cond=False)",
     "路由不再注入任务嵌入后掉 2。**结论：任务条件路由（TC-MoA 式）让专家按任务/模态特化**，对多任务联合训练有正贡献，验证了“任务感知路由”设计的必要性。"),
    ("hpNr8", "n_routed=8", "路由专家数 12 → 8",
     "n_routed 12→8 后 **medical 崩到 2**（MI 4.56→3.78、SF 26.7→19.9、Qabf 0.691→0.549），gfp 不受影响。**结论：medical（MRI-PET/SPECT 三子模态）需要更多细粒度专家分别特化，8 个容量不足**；gfp（单一显微模态）对专家数不敏感。**12 路由专家的选择由 medical 的多子模态需求决定**，是 n_routed 的合理下界。"),
]
OURS = "Ours"


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


def top3(rows, m):
    """count of Top-3 metrics for method m vs the 18 published methods."""
    field = [x for x in PUB if x in rows] + ([m] if m in rows else [])
    n = 0
    for k, hi in COLS:
        vals = {x: val(rows, x, k) for x in field if val(rows, x, k) is not None}
        if m not in vals:
            continue
        if sorted(vals.values(), reverse=bool(hi)).index(vals[m]) + 1 <= 3:
            n += 1
    return n


def table(t, variant):
    rows = load(t)
    shown = [x for x in PUB if x in rows] + [OURS, variant]
    best = {}
    for k, hi in COLS:
        vals = [val(rows, x, k) for x in shown if val(rows, x, k) is not None]
        best[k] = (max(vals) if hi else min(vals)) if vals else None
    out = ["| 方法 | " + " | ".join(k for k, _ in COLS) + " |",
           "|" + "---|" * (len(COLS) + 1)]
    for m in shown:
        cells = []
        for k, hi in COLS:
            v = val(rows, m, k)
            if v is None:
                cells.append("-"); continue
            s = fmt(v)
            if best[k] is not None and abs(v - best[k]) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        label = "**Ours (v3)**" if m == OURS else (f"**{variant} (变体)**" if m == variant else m)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out), top3(rows, OURS), top3(rows, variant)


for i, (vkey, vlabel, vdesc, vconc) in enumerate(EXPS, 1):
    print(f"### 4.{i} {vlabel}\n")
    print(f"改动：{vdesc}。下表为三模态全指标大表（18 方法 + Ours v3 + 本变体同场排名），**加粗=该表最优**。\n")
    counts = []
    for t, tn in TASKS:
        tbl, o3, v3c = table(t, vkey)
        print(f"**{tn}**\n")
        print(tbl)
        print(f"\n> v3 Top-3 = {o3} · 本变体 Top-3 = {v3c}（各自 vs 18 方法；共 12 指标）\n")
        counts.append((t.split('_')[0], o3, v3c))
    cs = "，".join(f"{n} {v3c}/{o3}" for n, o3, v3c in counts)
    print(f"**小结（变体 Top-3 / v3 Top-3）**：{cs}。\n")
    print(f"**结论**：{vconc}\n\n---\n")
