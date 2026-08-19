"""DenseFuse runner for the fusion_bench standardized contract.

Uses the official hli1221/densefuse-pytorch net + pretrained gray weights
(models/densefuse_gray.model). Implements both 'add' and the paper's
L1-norm (soft) fusion strategy. Convention: pretrained gray model operates
on pixel values in [0,255] (matches repo's get_test_images for mode 'L').

Run from the repo dir so `import net` works.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/ytech_m2v4_hdd/lizhongyin/code/ref/DenseFuse-pytorch")
from net import DenseFuse_net

INPUTS = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs"
OUTROOT = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/DenseFuse"
WEIGHTS = "/ytech_m2v4_hdd/lizhongyin/code/ref/DenseFuse-pytorch/models/densefuse_gray.model"
EPS = 1e-10


def load_gray_255(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)  # [0,255]
    t = torch.from_numpy(a)[None, None]  # (1,1,H,W)
    return t


def l1_fusion(f1, f2):
    """Paper's l1-norm strategy: per-pixel l1 of channel features -> box-blur
    average -> soft weights -> weighted sum of feature maps."""
    # activity level: sum over channels of |feature|
    a1 = torch.sum(torch.abs(f1), dim=1, keepdim=True)
    a2 = torch.sum(torch.abs(f2), dim=1, keepdim=True)
    # local average (box blur, window=3) as in the paper (block-based l1)
    k = 3
    pad = k // 2
    box = torch.ones(1, 1, k, k, device=f1.device) / (k * k)
    a1 = F.conv2d(F.pad(a1, (pad, pad, pad, pad), mode="reflect"), box)
    a2 = F.conv2d(F.pad(a2, (pad, pad, pad, pad), mode="reflect"), box)
    w1 = a1 / (a1 + a2 + EPS)
    w2 = a2 / (a1 + a2 + EPS)
    return w1 * f1 + w2 * f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--strategy", default="l1", choices=["l1", "add"])
    args = ap.parse_args()

    dev = "cuda"
    model = DenseFuse_net(1, 1)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval().to(dev)

    a_dir = os.path.join(INPUTS, args.task, "A")
    b_dir = os.path.join(INPUTS, args.task, "B")
    out_dir = os.path.join(OUTROOT, args.task)
    os.makedirs(out_dir, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a_dir) if f.endswith(".png"))
    n = 0
    with torch.no_grad():
        for stem in stems:
            # A = visible/functional, B = IR/structure. DenseFuse symmetric autoencoder.
            tA = load_gray_255(os.path.join(a_dir, stem + ".png")).to(dev)
            tB = load_gray_255(os.path.join(b_dir, stem + ".png")).to(dev)
            enA = model.encoder(tA)[0]
            enB = model.encoder(tB)[0]
            if args.strategy == "add":
                f = (enA + enB) / 2.0
            else:
                f = l1_fusion(enA, enB)
            out = model.decoder([f])[0]
            img = out.clamp(0, 255)[0, 0].cpu().numpy().astype("uint8")
            Image.fromarray(img).save(os.path.join(out_dir, stem + ".png"))
            n += 1
    print(f"[{args.task}] strategy={args.strategy} wrote {n} images to {out_dir}")


if __name__ == "__main__":
    main()
