"""Driver: run PIAFusion (pytorch port pretrained) over fusion_bench inputs.
A = VIS-Y / functional-Y (gray), B = IR / structure (gray). Output gray fused Y.
"""
import os, sys, argparse
import numpy as np
import torch
from PIL import Image

REPO = "/ytech_m2v4_hdd/lizhongyin/code/ref/PIAFusion_pytorch"
sys.path.insert(0, REPO)
from models.fusion_model import PIAFusion
from models.common import clamp

INPUTS = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs"
OUTROOT = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/PIAFusion"


def load_gray(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None]  # 1,1,H,W
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--weights", default=os.path.join(REPO, "pretrained/fusion_model_epoch_29.pth"))
    args = ap.parse_args()

    dev = "cuda"
    model = PIAFusion().to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    a_dir = os.path.join(INPUTS, args.task, "A")
    b_dir = os.path.join(INPUTS, args.task, "B")
    out_dir = os.path.join(OUTROOT, args.task)
    os.makedirs(out_dir, exist_ok=True)

    stems = sorted(f[:-4] for f in os.listdir(a_dir) if f.endswith(".png"))
    n = 0
    with torch.no_grad():
        for stem in stems:
            vis_y = load_gray(os.path.join(a_dir, stem + ".png")).to(dev)  # A -> VIS slot
            ir = load_gray(os.path.join(b_dir, stem + ".png")).to(dev)     # B -> IR slot
            fused = clamp(model(vis_y, ir))  # 1,1,H,W in [0,1]
            out = (fused[0, 0].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(out, mode="L").save(os.path.join(out_dir, stem + ".png"))
            n += 1
    print(f"[{args.task}] wrote {n} fused images -> {out_dir}")


if __name__ == "__main__":
    main()
