"""Generate fused images from a trained MDFNet checkpoint (YCbCr-aware).

Pipeline (matches the network design):
  * network input  = the Y (luminance) channel of each source (RGB->YCbCr->Y;
    grayscale sources use the gray value directly, which equals Y).
  * network output = fused Y (tanh, clipped to [0,1]).
  * FINAL image:
      - output_mode "rgb"  (e.g. GFP-PC): recombine fused Y with the CbCr fused
        from the sources (color comes from GFP) -> inverse YCbCr -> RGB .png.
      - output_mode "gray" (e.g. IR-VIS): save the fused Y as grayscale .png.
  * additionally always saves the fused Y to <out>_Y/ so evaluation (Y-domain
    metrics) is identical regardless of output_mode.

output_mode is read from the task config ("output_mode": "rgb"|"gray"),
overridable via --output-mode.

Usage:
  venv/bin/python infer_fusion.py --ckpt models/10/model_10.pth \
      --config configs/gfp_pc.json --set all --out outputs/mdfnet_gfp_pc --name MDFNet
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch

from Networks.net import MODEL
import fusion_data
import ycbcr


def _y01(y):
    return torch.from_numpy(y.astype(np.float32) / 255.0)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="all", choices=["all", "probe"])
    ap.add_argument("--in-channel", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=1)
    ap.add_argument("--out-channel", type=int, default=16)
    ap.add_argument("--output-mode", default=None, choices=["rgb", "gray"])
    ap.add_argument("--name", default="method")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    cfg = fusion_data.load_config(args.config)
    mode = args.output_mode or cfg.get("output_mode", "gray")
    pairs = fusion_data.select(cfg, args.set)
    os.makedirs(args.out, exist_ok=True)
    out_y = args.out.rstrip("/") + "_Y"
    os.makedirs(out_y, exist_ok=True)

    model = MODEL(in_channel=args.in_channel, window_size=args.window_size,
                  out_channel=args.out_channel)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval().to(dev)

    print(f"[{args.name}] {len(pairs)} pairs  mode={mode} -> {args.out}")
    with torch.no_grad():
        for stem, a_path, b_path in pairs:
            ya, cba, cra = ycbcr.load_ycbcr(a_path)
            yb, cbb, crb = ycbcr.load_ycbcr(b_path)
            # align B (and its chroma) to A's size if needed (pre-registration fix)
            if yb.shape != ya.shape:
                def _rs(x):
                    return np.asarray(Image.fromarray(x.astype("uint8")).resize(
                        (ya.shape[1], ya.shape[0]), Image.BILINEAR), dtype=np.float64)
                yb, cbb, crb = _rs(yb), _rs(cbb), _rs(crb)

            inp = torch.cat((_y01(ya), _y01(yb)), dim=1).to(dev)   # (1,2,H,W)
            fused_y = model(inp).squeeze().clamp(0, 1).cpu().numpy() * 255.0

            Image.fromarray(fused_y.round().astype("uint8")).save(
                os.path.join(out_y, stem + ".png"))

            if mode == "rgb":
                cbf = ycbcr.fuse_chroma(cba, cbb)
                crf = ycbcr.fuse_chroma(cra, crb)
                rgb = ycbcr.ycbcr_to_rgb(fused_y, cbf, crf)
                Image.fromarray(rgb).save(os.path.join(args.out, stem + ".png"))
            else:
                Image.fromarray(fused_y.round().astype("uint8")).save(
                    os.path.join(args.out, stem + ".png"))
    print(f"[{args.name}] done (final={mode} in {args.out}, fused-Y in {out_y})")


if __name__ == "__main__":
    main()
