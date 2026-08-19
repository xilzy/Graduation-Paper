"""Refactored MDFNet training: real batched forward + unified FusionDataset.

Key differences vs the original Train.py:
  * NO per-sample Python loop -> true mini-batch forward/backward (the original
    iterated one image at a time inside each batch, killing GPU parallelism).
  * Data via FusionDataset (on-the-fly overlapping patches) instead of a
    pre-baked h5 file; in_channel and window_size are parameters.
  * Same loss formulation (content = RMI + intensity, structure = SSIM + grad)
    so single-task GFP-PC results stay comparable to the released model.

Usage:
  venv/bin/python train_fusion.py --config configs/gfp_pc.json --name p0_gfp_pc \
      --epochs 10 --batch-size 64 --lr 1e-3
"""
import os
import argparse
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Networks.net import MODEL
from losses import ssim_ir, ssim_vi, RMI_ir, RMI_vi, intensity, joint_grad
from fusion_dataset import FusionDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True,
                    help="one or more task config json files")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight", type=float, nargs=7, default=[1, 5, 1, 2.5, 1, 1, 1],
                    help="ssim_a ssim_b rmi_a rmi_b int_a int_b grad")
    ap.add_argument("--alpha", type=float, default=2.0, help="structure weight")
    ap.add_argument("--lr-decay", type=float, default=0.8,
                    help="per-epoch LR attenuation (paper: 0.8); 1.0 disables")
    ap.add_argument("--in-channel", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=1)
    ap.add_argument("--patch", type=int, default=170)
    ap.add_argument("--stride", type=int, default=28)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(f"models/{args.name}", exist_ok=True)
    os.makedirs(f"logs/{args.name}", exist_ok=True)
    with open(f"models/{args.name}/args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    ds = FusionDataset(args.config, split="train",
                       patch=args.patch, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    print(f"train patches: {len(ds)}  batches/epoch: {len(dl)}  tasks: {ds.tasks}")

    model = MODEL(in_channel=args.in_channel, window_size=args.window_size).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    # paper: learning-rate attenuation 0.8 per epoch (StepLR step=1)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.lr_decay)
    w = args.weight
    log = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {k: 0.0 for k in ("loss", "content", "structure", "ssim", "rmi", "intensity", "grad")}
        nb = 0
        for batch in dl:
            a = batch["src_a"].to(dev, non_blocking=True)   # (B,1,H,W)
            b = batch["src_b"].to(dev, non_blocking=True)
            inp = torch.cat((a, b), dim=1)                  # (B,2,H,W)
            out = model(inp)                                # (B,1,H,W)

            loss_ssim = w[0] * ssim_ir(out, a) + w[1] * ssim_vi(out, b)
            # joint_grad returns a per-sample (B,1) L1 norm; reduce to scalar for
            # batched backward (the original per-sample loop had B=1 so numel==1)
            loss_grad = w[6] * joint_grad(a, b, out).mean()
            L_structure = loss_ssim + loss_grad

            loss_rmi = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
            loss_int = 0.5 * w[4] * intensity(a, out) + 0.5 * w[5] * intensity(b, out)
            L_content = loss_rmi + loss_int

            loss = L_content + args.alpha * L_structure

            opt.zero_grad()
            loss.backward()
            opt.step()

            agg["loss"] += loss.item(); agg["content"] += L_content.item()
            agg["structure"] += L_structure.item(); agg["ssim"] += loss_ssim.item()
            agg["rmi"] += loss_rmi.item(); agg["intensity"] += loss_int.item()
            agg["grad"] += loss_grad.item(); nb += 1

        for k in agg:
            agg[k] /= max(nb, 1)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{args.epochs}  lr {cur_lr:.2e}  loss {agg['loss']:.4f} "
              f"(content {agg['content']:.4f} structure {agg['structure']:.4f})  {dt:.1f}s")
        row = {"epoch": epoch + 1, "lr": cur_lr, **agg, "sec": dt}
        log.append(row)
        pd.DataFrame(log).to_csv(f"logs/{args.name}/log.csv", index=False)
        torch.save(model.state_dict(), f"models/{args.name}/model_{epoch+1}.pth")
        sched.step()

    print("training done")


if __name__ == "__main__":
    main()
