"""Clean PIL+torch driver for RFN-Nest (Li et al., InfFus 2021).

Reuses ONLY the net definitions (net.py) + bundled pretrained state_dicts:
  - nest autoencoder: models/nestfuse/nestfuse_gray_1e2.model
  - RFN fusion (two-stage): models/rfn_twostage/6.0/Final_epoch_2_alpha_700_wir_6.0_wvi_3.0_ssim_vi.model
Gray [0,255] convention (mode='L', no /255), clamp(0,255) -> uint8, like the repo.
Slot mapping per contract: A -> VIS (en_vi), B -> IR (en_ir).
"""
import os, sys, argparse
import numpy as np
import torch
from PIL import Image

REPO = "/ytech_m2v4_hdd/lizhongyin/code/ref/RFN-Nest"
sys.path.insert(0, REPO)
from net import NestFuse_light2_nodense, Fusion_network  # noqa

INPUTS = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/inputs"
OUTROOT = "/ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/RFN-Nest"
NB_FILTER = [64, 112, 160, 208, 256]
PATH_AUTO = os.path.join(REPO, "models/nestfuse/nestfuse_gray_1e2.model")
PATH_RFN = os.path.join(REPO, "models/rfn_twostage/6.0/Final_epoch_2_alpha_700_wir_6.0_wvi_3.0_ssim_vi.model")


def load_models(device):
    nest = NestFuse_light2_nodense(NB_FILTER, 1, 1, deepsupervision=False)
    nest.load_state_dict(torch.load(PATH_AUTO, map_location="cpu"))
    fusion = Fusion_network(NB_FILTER, "res")
    fusion.load_state_dict(torch.load(PATH_RFN, map_location="cpu"))
    nest.eval().to(device)
    fusion.eval().to(device)
    return nest, fusion


def to_tensor(path, device):
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)  # [0,255]
    t = torch.from_numpy(arr)[None, None].to(device)
    return t


def pad8(t):
    _, _, h, w = t.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


@torch.no_grad()
def fuse_one(nest, fusion, a_path, b_path, device):
    vi = to_tensor(a_path, device)   # A -> VIS
    ir = to_tensor(b_path, device)   # B -> IR
    vi, h, w = pad8(vi)
    ir, _, _ = pad8(ir)
    en_vi = nest.encoder(vi)
    en_ir = nest.encoder(ir)
    f = fusion(en_ir, en_vi)
    out = nest.decoder_eval(f)[0]    # deepsupervision=False -> single output
    out = out[0, 0, :h, :w].cpu().numpy()
    # repo convention (utils.save_image_test): per-image min-max -> [0,255], NOT clamp
    mn, mx = out.min(), out.max()
    out = (out - mn) / (mx - mn + 1e-5) * 255.0
    return out.round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    device = "cuda"
    nest, fusion = load_models(device)
    a_dir = os.path.join(INPUTS, args.task, "A")
    b_dir = os.path.join(INPUTS, args.task, "B")
    out_dir = os.path.join(OUTROOT, args.task)
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted(f for f in os.listdir(a_dir) if f.endswith(".png"))
    n = 0
    for s in stems:
        fused = fuse_one(nest, fusion, os.path.join(a_dir, s), os.path.join(b_dir, s), device)
        Image.fromarray(fused, mode="L").save(os.path.join(out_dir, s))
        n += 1
    print(f"[{args.task}] wrote {n} fused images to {out_dir}")


if __name__ == "__main__":
    main()
