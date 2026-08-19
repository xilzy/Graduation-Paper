"""Quantitative S1-style comparison of trained fusion models (dense vs MoE).

For each method and task, runs full-image inference on the frozen probe set,
computes the balance-aware RANK_METRICS via the metrics package, averages per
task, then ranks the methods (lower average rank = better). This is the S1
"average rank" aggregate the project decided on.

Each method is given as models/<name>; its arch + hyperparams are read from the
saved models/<name>/args.txt, so dense (train_mm) and MoE (train_moe) just work.

Usage:
  venv/bin/python mm_eval_compare.py --methods h2h_dense h2h_moe_sym h2h_moe_ta \
     --epoch last --config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json
"""
import os, argparse, glob
import numpy as np
from PIL import Image
import torch
import mm_fusion_data as mfd
import metrics as M


def read_args(name):
    d = {}
    p = f"models/{name}/args.txt"
    if os.path.exists(p):
        for line in open(p):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def y(path):
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA", "P", "CMYK"):
        a = np.asarray(im.convert("YCbCr"), np.float32)[:, :, 0]
    else:
        a = np.asarray(im.convert("L"), np.float32)
    return a


def build_model(name, dev):
    a = read_args(name)
    is_moe = "n_routed" in a
    ck = sorted(glob.glob(f"models/{name}/model_*.pth"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
    if is_moe:
        from Networks.net_moe import MODEL_MoE
        m = MODEL_MoE(in_channel=2, n_tasks=3,
                      out_channel=int(a.get("out_channel", 16)),
                      depth=int(a.get("depth", 3)),
                      num_heads=int(a.get("num_heads", 8)),
                      n_routed=int(a.get("n_routed", 4)),
                      n_shared=int(a.get("n_shared", 1)),
                      out_scale=a.get("out_scale", "False") == "True",
                      use_task_bias=a.get("no_task_bias", "False") != "True")
    else:
        from Networks.net import MODEL
        m = MODEL(in_channel=2)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m.eval().to(dev)
    return m, is_moe, os.path.basename(ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    models = {}
    for nm in args.methods:
        models[nm] = build_model(nm, dev)
        print(f"loaded {nm}: arch={'moe' if models[nm][1] else 'dense'} ckpt={models[nm][2]}")

    # per task: method -> {metric: mean over probe}
    overall_rank = {nm: [] for nm in args.methods}
    for tid, cfgp in enumerate(args.config):
        cfg = mfd.load_config(cfgp)
        task = cfg["task"]
        probe = set(mfd.probe_stems(cfg))
        pairs = [p for p in mfd.list_pairs(cfg, "test") if p[0] in probe]
        method_means = {}
        for nm in args.methods:
            m, is_moe, _ = models[nm]
            accum = {}
            with torch.no_grad():
                for stem, ap_, bp_ in pairs:
                    A = y(ap_); B = y(bp_)
                    if B.shape != A.shape:
                        B = np.asarray(Image.fromarray(B.astype("uint8")).resize(
                            (A.shape[1], A.shape[0]), Image.BILINEAR), np.float32)
                    inp = torch.from_numpy(np.stack([A, B])[None] / 255.).float().to(dev)
                    out = m(inp, torch.tensor([tid], device=dev))[0] if is_moe else m(inp)
                    F = out.squeeze().clamp(0, 1).cpu().numpy() * 255.0
                    r = M.compute_all(A, B, F, include_diagnostic=False)
                    for k, v in r.items():
                        accum.setdefault(k, []).append(float(v))
            method_means[nm] = {k: float(np.nanmean(v)) for k, v in accum.items()}
        # average rank over RANK_METRICS for this task
        tbl = M.average_rank_table(method_means, metrics=list(M.RANK_METRICS))
        print(f"\n=== task {task} (n={len(pairs)}) — avg rank (lower=better) ===")
        ranked = sorted(args.methods, key=lambda n: float(tbl.loc[n, "AvgRank"]))
        for nm in ranked:
            mm = method_means[nm]
            print(f"  {nm:14s} avgRank={float(tbl.loc[nm,'AvgRank']):.2f} | "
                  f"SSIM_hm={mm['SSIM_hm']:.3f} Qabf_hm={mm['Qabf_hm']:.3f} "
                  f"VIF_hm={mm['VIF_hm']:.3f} MI_hm={mm['MI_hm']:.3f} "
                  f"SF={mm['SF']:.1f} SD={mm['SD']:.1f}")
            overall_rank[nm].append(float(tbl.loc[nm, "AvgRank"]))
    print("\n#### OVERALL mean avg-rank across tasks (lower=better) ####")
    for nm in sorted(args.methods, key=lambda n: np.mean(overall_rank[n])):
        print(f"  {nm:14s} {np.mean(overall_rank[nm]):.3f}")


if __name__ == "__main__":
    main()
