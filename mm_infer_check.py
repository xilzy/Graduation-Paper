"""Multi-task inference + S0 degradation self-check for the unified model.

Runs a trained checkpoint (dense net.MODEL or MoE net_moe.MODEL_MoE) over each
task's probe set, saves a few fused images (Y, and RGB if output_mode=rgb), and
reports the S0 degradation panel per task:
  * black/constant : mean & std of fused Y
  * copy-single-src: corr(fused, A) vs corr(fused, B) (a healthy fuse keeps both;
    a near-1 / near-0 split means it collapsed to one source)
  * NaN/Inf check

Usage:
  venv/bin/python mm_infer_check.py --ckpt models/mm_dense_v1/model_8.pth \
     --config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
     --out outputs/mm_dense_v1 --arch dense
  (--arch moe for net_moe; task_id = position in --config list)
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch

import mm_fusion_data as mfd
import ycbcr


def y01(path):
    im = Image.open(path)
    if im.mode in ("RGB", "RGBA", "P", "CMYK"):
        y = np.asarray(im.convert("YCbCr"), dtype=np.float32)[:, :, 0]
    else:
        y = np.asarray(im.convert("L"), dtype=np.float32)
    return y / 255.0


def corr(x, y):
    x, y = x.ravel(), y.ravel()
    x = x - x.mean(); y = y - y.mean()
    d = (np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / d) if d > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["dense", "moe"], default="dense")
    ap.add_argument("--out-channel", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--n-routed", type=int, default=4)
    ap.add_argument("--n-shared", type=int, default=1)
    ap.add_argument("--out-scale", action="store_true")
    ap.add_argument("--no-task-bias", action="store_true")
    ap.add_argument("--n-per-task", type=int, default=5)
    ap.add_argument("--save", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.arch == "moe":
        from Networks.net_moe import MODEL_MoE
        model = MODEL_MoE(in_channel=2, n_tasks=len(args.config),
                          out_channel=args.out_channel, depth=args.depth,
                          num_heads=args.num_heads, n_routed=args.n_routed,
                          n_shared=args.n_shared, out_scale=args.out_scale,
                          use_task_bias=not args.no_task_bias)
    else:
        from Networks.net import MODEL
        model = MODEL(in_channel=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval().to(dev)

    os.makedirs(args.out, exist_ok=True)
    print(f"=== degradation self-check [{args.arch}] {args.ckpt} ===")
    overall_ok = True
    for tid, cfgp in enumerate(args.config):
        cfg = mfd.load_config(cfgp)
        task = cfg["task"]
        mode = cfg.get("output_mode", "gray")
        probe = mfd.probe_stems(cfg)[: args.n_per_task]
        pairs = [p for p in mfd.list_pairs(cfg, "test") if p[0] in probe] or \
                mfd.list_pairs(cfg, "test")[: args.n_per_task]
        tdir = os.path.join(args.out, task); os.makedirs(tdir, exist_ok=True)
        rows = []
        with torch.no_grad():
            for i, (stem, ap_, bp_) in enumerate(pairs):
                A = y01(ap_); B = y01(bp_)
                if B.shape != A.shape:
                    B = np.asarray(Image.fromarray((B*255).astype("uint8")).resize(
                        (A.shape[1], A.shape[0]), Image.BILINEAR), np.float32)/255.0
                # ACM now size-correct (P4 fix) -> full-image inference, any HxW
                inp = torch.from_numpy(np.stack([A, B])[None]).float().to(dev)
                if args.arch == "moe":
                    out, _ = model(inp, torch.tensor([tid], device=dev))
                else:
                    out = model(inp)
                f = out.squeeze().clamp(0, 1).cpu().numpy()
                nan = bool(np.isnan(f).any() or np.isinf(f).any())
                rows.append((stem, f.mean(), f.std(), corr(f, A), corr(f, B), nan,
                             A.std(), B.std()))
                if i < args.save:
                    Image.fromarray((f*255).round().astype("uint8")).save(
                        os.path.join(tdir, stem + "_Y.png"))
                    if mode == "rgb":
                        ya, cba, cra = ycbcr.load_ycbcr(ap_)
                        yb, cbb, crb = ycbcr.load_ycbcr(bp_)
                        cbf, crf = ycbcr.fuse_chroma(cba, cbb), ycbcr.fuse_chroma(cra, crb)
                        Image.fromarray(ycbcr.ycbcr_to_rgb(f*255, cbf, crf)).save(
                            os.path.join(tdir, stem + "_rgb.png"))
        m = np.array([[r[1], r[2], r[3], r[4], r[6], r[7]] for r in rows])
        anynan = any(r[5] for r in rows)
        mean_, std_, cA, cB, sA, sB = m.mean(0)
        black = mean_ < 0.02 or std_ < 0.02
        # copy-collapse only meaningful when BOTH sources carry real luminance
        # structure (sA,sB>0.06); GFP fluorescence is near-black in Y (info in
        # chroma), so its low-Y-variance source is correctly exempt.
        both_lum = sA > 0.06 and sB > 0.06
        copy = both_lum and (
            (abs(cA) > 0.98 and abs(cB) < 0.4) or (abs(cB) > 0.98 and abs(cA) < 0.4))
        ok = not (anynan or black or copy)
        overall_ok &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {task:8s} n={len(rows)} mean={mean_:.3f} std={std_:.3f} "
              f"corrA={cA:.3f} corrB={cB:.3f} (srcStd A={sA:.3f} B={sB:.3f}) nan={anynan} "
              f"{'BLACK ' if black else ''}{'COPY ' if copy else ''}-> {tdir}")
    print(f"=== overall: {'ALL PASS' if overall_ok else 'SOME FAIL'} ===")


if __name__ == "__main__":
    main()
