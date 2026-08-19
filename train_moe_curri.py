"""Curriculum-quota MoE training (Innovation A) — new file, leaves train_moe.py intact.

Stage 1 (warmup, first --warmup epochs): strict 1:1:1 balanced sampling
(crops_per_task identical for all tasks) so no modality dominates early.

Stage 2 (after warmup): per-task sampling quota is adapted from the per-task
TRAINING LOSS measured each epoch. The worse-trained task (higher loss) gets a
bit MORE data; the better-trained task a bit less; the shift is CAPPED so the
mix stays close to balanced (ratio within [1/cap, cap], total ~constant).
  quota_t = base * clip( (loss_t / mean_loss)^beta , 1/cap, cap ), renormalized.
(`--curri-direction low` flips it to upweight the LOWER-loss task instead.)

Per-task loss is obtained by computing the fusion loss on each task's slice of
every mixed batch (forward is shared/once; only the loss is split).

Reuses MODEL_MoE + the maxfuse loss (current best: blend head + window attn).
"""
import os, argparse, time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Networks.net_moe import MODEL_MoE
from losses import ssim_ir, ssim_vi, RMI_ir, RMI_vi, joint_grad, ssim_loss
from mm_fusion_dataset import MMFusionDataset
from train_moe import ms_max_grad_loss   # REF Addition B (multi-scale max-grad)

MAXT = {"irvis", "medical"}  # (unused for maxfuse; kept for parity)


def maxfuse_loss(out, a, b, w, alpha, ssim_target, ms_grad=0.0):
    if ssim_target == "max":
        loss_ssim = (w[0] + w[1]) * 0.5 * ssim_loss(out, a, b)
    else:
        loss_ssim = w[0] * ssim_ir(out, a) + w[1] * ssim_vi(out, b)
    loss_grad = w[6] * joint_grad(a, b, out).mean()
    if ms_grad > 0:
        loss_grad = loss_grad + ms_grad * ms_max_grad_loss(a, b, out)
    loss_rmi = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
    loss_int = w[4] * torch.mean((torch.maximum(a, b) - out) ** 2)
    return loss_rmi + loss_int + alpha * (loss_ssim + loss_grad)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6, help="epochs of strict 1:1:1")
    ap.add_argument("--cap", type=float, default=1.2, help="max per-task quota ratio (gentle)")
    ap.add_argument("--beta", type=float, default=1.0, help="loss->quota sensitivity")
    ap.add_argument("--ms-grad", type=float, default=0.0, help="REF Addition B weight")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight", type=float, nargs=7, default=[2, 2, 2, 2, 4, 0, 3])
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--crops-per-task", type=int, default=4000)
    ap.add_argument("--patch", type=int, default=170)
    ap.add_argument("--out-channel", type=int, default=64)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--n-routed", type=int, default=12)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-shared", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--ssim-target", default="max", choices=["max", "balanced"])
    ap.add_argument("--fusion-head", default="blend", choices=["direct", "blend"])
    ap.add_argument("--res-scale", type=float, default=0.0)
    ap.add_argument("--routing", default="softmax", choices=["softmax", "deepseek"])
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

    tasks = [__import__("mm_fusion_data").load_config(c)["task"] if isinstance(c, str) else c["task"]
             for c in args.config]
    model = MODEL_MoE(in_channel=2, n_tasks=len(args.config), out_channel=args.out_channel,
                      depth=3, num_heads=args.num_heads, window_size=args.window_size,
                      n_routed=args.n_routed, k=args.k, n_shared=args.n_shared,
                      task_cond=True, out_scale=True, fusion_head=args.fusion_head,
                      res_scale=args.res_scale, routing=args.routing).to(dev)
    print(f"MoE params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"routing={args.routing} tasks={tasks}")
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.lr_decay)
    w, log = args.weight, []
    quota = args.crops_per_task                          # int -> balanced 1:1:1
    ref_loss = {}
    base = args.crops_per_task

    for epoch in range(args.epochs):
        ds = MMFusionDataset(args.config, split="train", patch=args.patch,
                             crops_per_task=quota, random_crop=True, seed=epoch)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
        cnt = {}
        for tid, *_ in ds.index:
            cnt[tasks[tid]] = cnt.get(tasks[tid], 0) + 1
        model.train()
        t0 = time.time()
        tl_sum = {t: 0.0 for t in tasks}; tl_n = {t: 0 for t in tasks}
        agg = 0.0; nb = 0
        for batch in dl:
            a = batch["src_a"].to(dev, non_blocking=True)
            b = batch["src_b"].to(dev, non_blocking=True)
            tid = batch["task_id"].to(dev, non_blocking=True)
            out, aux = model(torch.cat((a, b), 1), tid)
            loss = 0.0
            for t in torch.unique(tid):
                m = tid == t
                lt = maxfuse_loss(out[m], a[m], b[m], w, args.alpha, args.ssim_target, args.ms_grad)
                frac = m.float().mean()
                loss = loss + lt * frac                  # weighted -> full-batch mean
                tn = tasks[int(t)]
                tl_sum[tn] += float(lt) * int(m.sum()); tl_n[tn] += int(m.sum())
            loss = loss + args.aux_weight * aux
            opt.zero_grad(); loss.backward(); opt.step()
            agg += float(loss); nb += 1
        # per-task mean loss this epoch
        tloss = {t: (tl_sum[t] / max(tl_n[t], 1)) for t in tasks}
        dt = time.time() - t0
        print(f"ep{epoch+1}/{args.epochs} lr{opt.param_groups[0]['lr']:.2e} "
              f"loss{agg/max(nb,1):.4f} per-task-loss{ {k:round(v,3) for k,v in tloss.items()} } "
              f"quota{cnt} {dt:.0f}s")
        log.append({"epoch": epoch + 1, "loss": agg / max(nb, 1),
                    **{f"loss_{t}": tloss[t] for t in tasks},
                    **{f"quota_{t}": cnt.get(t, 0) for t in tasks}})
        pd.DataFrame(log).to_csv(f"logs/{args.name}/log.csv", index=False)
        torch.save(model.state_dict(), f"models/{args.name}/model_{epoch+1}.pth")
        sched.step()

        # ---- record per-task reference loss at warmup end (for RELATIVE signal) ----
        if epoch + 1 == args.warmup:
            ref_loss = dict(tloss)                        # baseline per task
        # ---- after warmup: adapt next-epoch quota from RELATIVE loss progress ----
        # (fix for the absolute-loss bug: medical's loss is intrinsically larger,
        #  so use each task's loss RELATIVE to its own warmup-end value. A task
        #  whose loss has dropped LESS = still behind = gets a bit MORE data.)
        if epoch + 1 >= args.warmup and ref_loss:
            rel = {t: tloss[t] / max(ref_loss[t], 1e-6) for t in tasks}
            mean_r = sum(rel.values()) / len(rel)
            quota = {}
            for t in tasks:
                r = (rel[t] / mean_r) if mean_r > 0 else 1.0   # >1 = lagging -> more
                r = max(1.0 / args.cap, min(args.cap, r ** args.beta))
                quota[t] = int(base * r)

    print("training done")


if __name__ == "__main__":
    main()
