"""Multi-GPU (DDP) training for the v3 MoE — Track A training-time optimisation.

The v3 model is small (~4M), so the batched-dense MoE OOMs and bf16 is slower
(tiny-op/cast overhead + RMI cholesky) — profiling showed the win is (a) fewer
kernel launches via torch.compile and (b) data-parallelism. This runs 4-GPU DDP
+ optional torch.compile, cutting wall-clock ~4x for the same data.

Launch (4 GPUs 0-3):
  torchrun --nproc_per_node=4 --master_port=29501 train_moe_ddp.py \
    --config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
    --name v3_ddp --epochs 20 --batch-size 10 --out-channel 96 --depth 4 \
    --window-size 8 --n-routed 12 --compile
"""
import os, argparse, time
import torch, torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Networks.net_moe import MODEL_MoE
from losses import ssim_loss, RMI_ir, RMI_vi, joint_grad
from mm_fusion_dataset import MMFusionDataset


def maxfuse(out, a, b, w, alpha):
    ls = (w[0] + w[1]) * 0.5 * ssim_loss(out, a, b)
    lg = w[6] * joint_grad(a, b, out).mean()
    lr = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
    li = w[4] * torch.mean((torch.maximum(a, b) - out) ** 2)
    return lr + li + alpha * (ls + lg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=10, help="per-GPU batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--weight", type=float, nargs=7, default=[2, 2, 2, 2, 4, 0, 3])
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--crops-per-task", type=int, default=4000)
    ap.add_argument("--patch", type=int, default=170)
    ap.add_argument("--out-channel", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--n-routed", type=int, default=12)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--combine", default="sparse")
    ap.add_argument("--attn", default="vanilla", choices=["vanilla", "sdpa"])
    ap.add_argument("--cap-factor", type=float, default=1.25)
    ap.add_argument("--fused-adam", action="store_true", help="fused Adam kernel (fewer optimizer launches on many small params)")
    ap.add_argument("--bucket-cap", type=float, default=25.0, help="DDP gradient bucket size (MB); smaller overlaps comm earlier for tiny models")
    ap.add_argument("--grad-bucket-view", action="store_true", help="DDP gradient_as_bucket_view (saves a grad copy)")
    ap.add_argument("--touch-all", action="store_true", help="touch all params (0*sum) -> find_unused_parameters=False (faster DDP)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0, help=">0: timing test")
    ap.add_argument("--warmup", type=int, default=0, help="steps to skip before steady-state timing")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_); dev = f"cuda:{lr_}"
    if rank == 0:
        os.makedirs(f"models/{args.name}", exist_ok=True); os.makedirs(f"logs/{args.name}", exist_ok=True)

    ds = MMFusionDataset(args.config, "train", args.patch, args.crops_per_task, random_crop=True)
    samp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=samp,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    model = MODEL_MoE(in_channel=2, n_tasks=len(args.config), out_channel=args.out_channel,
                      depth=args.depth, window_size=args.window_size, n_routed=args.n_routed,
                      k=args.k, n_shared=1, out_scale=True, fusion_head="blend",
                      res_scale=0.0, attn_impl=args.attn).to(dev)
    model.set_combine(args.combine, cap_factor=args.cap_factor)
    if args.compile:
        model = torch.compile(model)
    # find_unused_parameters=True is the safe MoE default but its per-step graph
    # traversal caps DDP scaling (~2.6x). --touch-all instead adds a 0*sum(params)
    # term so every param gets a (zero) gradient -> no "unused" params -> we can
    # use the faster find_unused_parameters=False path. bucket_cap_mb + grad-bucket-view
    # tune the comm path that bounds scaling for this tiny (4M) model.
    ddp = DDP(model, device_ids=[lr_], find_unused_parameters=not args.touch_all,
              bucket_cap_mb=args.bucket_cap, gradient_as_bucket_view=args.grad_bucket_view)
    opt = optim.Adam(ddp.parameters(), lr=args.lr, fused=args.fused_adam)
    sch = optim.lr_scheduler.StepLR(opt, 1, gamma=args.lr_decay)
    w = args.weight

    for ep in range(args.epochs):
        samp.set_epoch(ep); ddp.train(); t0 = time.time(); nb = 0; tsteady = None; nsteady = 0
        for batch in dl:
            a = batch["src_a"].to(dev, non_blocking=True)
            b = batch["src_b"].to(dev, non_blocking=True)
            tid = batch["task_id"].to(dev, non_blocking=True)
            out, aux = ddp(torch.cat((a, b), 1), tid)
            loss = maxfuse(out, a, b, w, args.alpha) + args.aux_weight * aux
            if args.touch_all:
                loss = loss + 0.0 * sum(p.sum() for p in ddp.parameters())
            opt.zero_grad(); loss.backward(); opt.step(); nb += 1
            if args.warmup and nb == args.warmup:      # start steady-state clock after warmup
                torch.cuda.synchronize(); tsteady = time.time(); nsteady = nb
            if args.max_steps and nb >= args.max_steps:
                break
        torch.cuda.synchronize()
        if rank == 0:
            msg = (f"ep{ep+1}/{args.epochs} loss{float(loss):.4f} {nb} steps {time.time()-t0:.1f}s "
                   f"(world={world}, per-gpu bs{args.batch_size})")
            if tsteady is not None and nb > nsteady:
                ms = (time.time() - tsteady) / (nb - nsteady) * 1000
                msg += f"  steady {ms:.1f} ms/step (over {nb-nsteady} steps)"
            print(msg, flush=True)
            (model._orig_mod if hasattr(model, "_orig_mod") else model)
            torch.save((model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                       f"models/{args.name}/model_{ep+1}.pth")
        sch.step()
        if args.max_steps:
            break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
