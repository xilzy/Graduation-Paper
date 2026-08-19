"""Unified multi-task MDFNet training (GFP-PC + IR-VIS + medical).

Same network + loss contract as train_fusion.py, but the data comes from
MMFusionDataset which mixes several task configs into one balanced stream of
170x170 luminance crops. A single shared-weight ("dense") model is trained on
all tasks at once -- this is the Phase-1 dense multi-task baseline the MoE work
will be compared against.

Per-task losses are also logged so task interference is visible.

Usage:
  venv/bin/python train_mm.py \
    --config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
    --name mm_dense --epochs 10 --batch-size 64 --crops-per-task 4000
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
from mm_fusion_dataset import MMFusionDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight", type=float, nargs=7, default=[1, 5, 1, 2.5, 1, 1, 1],
                    help="ssim_a ssim_b rmi_a rmi_b int_a int_b grad")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--in-channel", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=1)
    ap.add_argument("--patch", type=int, default=170)
    ap.add_argument("--crops-per-task", type=int, default=4000)
    ap.add_argument("--fixed-pool", action="store_true",
                    help="use fixed crop pool (default: on-the-fly random crops)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="if >0, stop after this many optimizer steps (smoke test)")
    return ap.parse_args()


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(f"models/{args.name}", exist_ok=True)
    os.makedirs(f"logs/{args.name}", exist_ok=True)
    with open(f"models/{args.name}/args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    ds = MMFusionDataset(args.config, split="train", patch=args.patch,
                         crops_per_task=args.crops_per_task,
                         random_crop=not args.fixed_pool)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    # report per-task crop counts
    counts = {}
    for tid, *_ in ds.index:
        counts[ds.tasks[tid]] = counts.get(ds.tasks[tid], 0) + 1
    print(f"train crops: {len(ds)}  batches/epoch: {len(dl)}  per-task: {counts}")

    model = MODEL(in_channel=args.in_channel, window_size=args.window_size).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.lr_decay)
    w = args.weight
    log = []
    gstep = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {k: 0.0 for k in ("loss", "content", "structure", "ssim", "rmi", "intensity", "grad")}
        per_task = {}
        nb = 0
        for batch in dl:
            a = batch["src_a"].to(dev, non_blocking=True)
            b = batch["src_b"].to(dev, non_blocking=True)
            inp = torch.cat((a, b), dim=1)
            out = model(inp)

            loss_ssim = w[0] * ssim_ir(out, a) + w[1] * ssim_vi(out, b)
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
            gstep += 1
            if args.max_steps and gstep >= args.max_steps:
                break

        for k in agg:
            agg[k] /= max(nb, 1)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{args.epochs}  lr {cur_lr:.2e}  loss {agg['loss']:.4f} "
              f"(content {agg['content']:.4f} structure {agg['structure']:.4f})  {dt:.1f}s")
        log.append({"epoch": epoch + 1, "lr": cur_lr, **agg, "sec": dt})
        pd.DataFrame(log).to_csv(f"logs/{args.name}/log.csv", index=False)
        torch.save(model.state_dict(), f"models/{args.name}/model_{epoch+1}.pth")
        sched.step()
        if args.max_steps and gstep >= args.max_steps:
            print("hit max-steps, stopping"); break

    print("training done")


if __name__ == "__main__":
    main()
