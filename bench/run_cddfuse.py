"""CDDFuse fusion over the standardized benchmark inputs (all 3 tasks).

Uses the local pretrained weights:
  models/CDDFuse_IVF.pth  -> irvis
  models/CDDFuse_MIF.pth  -> medical, gfp_pc  (microscopy/medical-like)

A -> VIS slot, B -> IR slot (consistent). Output = min-max normalized fused gray.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn

CDD = "/ytech_m2v4_hdd/lizhongyin/code/ref/MMIF-CDDFuse"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
sys.path.insert(0, CDD)
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

TASK_CKPT = {"irvis": "CDDFuse_IVF.pth", "medical": "CDDFuse_MIF.pth", "gfp_pc": "CDDFuse_MIF.pth"}


def load_models(ckpt_path, device):
    Enc = nn.DataParallel(Restormer_Encoder()).to(device)
    Dec = nn.DataParallel(Restormer_Decoder()).to(device)
    BF = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DF = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    Enc.load_state_dict(ck['DIDF_Encoder']); Dec.load_state_dict(ck['DIDF_Decoder'])
    BF.load_state_dict(ck['BaseFuseLayer']); DF.load_state_dict(ck['DetailFuseLayer'])
    for m in (Enc, Dec, BF, DF): m.eval()
    return Enc, Dec, BF, DF


def fuse_one(a, b, models, device):
    Enc, Dec, BF, DF = models
    vis = torch.FloatTensor(a[None, None] / 255.0).to(device)
    ir = torch.FloatTensor(b[None, None] / 255.0).to(device)
    fVB, fVD, _ = Enc(vis); fIB, fID, _ = Enc(ir)
    fFB = BF(fVB + fIB); fFD = DF(fVD + fID)
    fuse, _ = Dec(vis, fFB, fFD)
    mn, mx = fuse.min(), fuse.max()
    fuse = (fuse - mn) / (mx - mn).clamp_min(1e-8)
    return np.squeeze((fuse * 255).detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["irvis", "medical", "gfp_pc"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    for task in args.tasks:
        ckpt = os.path.join(CDD, "models", TASK_CKPT[task])
        if ckpt not in cache:
            cache[ckpt] = load_models(ckpt, device)
        models = cache[ckpt]
        a_dir = os.path.join(BENCH, "inputs", task, "A")
        b_dir = os.path.join(BENCH, "inputs", task, "B")
        out = os.path.join(BENCH, "fused", "CDDFuse", task); os.makedirs(out, exist_ok=True)
        stems = [f[:-4] for f in os.listdir(a_dir) if f.endswith(".png")]
        with torch.no_grad():
            for s in stems:
                a = np.asarray(Image.open(os.path.join(a_dir, s + ".png")).convert("L"), float)
                b = np.asarray(Image.open(os.path.join(b_dir, s + ".png")).convert("L"), float)
                f = fuse_one(a, b, models, device)
                Image.fromarray(f.clip(0, 255).astype("uint8")).save(os.path.join(out, s + ".png"))
        print(f"[CDDFuse/{task}] wrote {len(stems)} -> {out}")


if __name__ == "__main__":
    main()
