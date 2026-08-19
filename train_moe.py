"""Train the MoE MDFNet (U-MDFNet) on the unified multi-task fusion stream.

Same data pipeline + base loss as train_mm.py, plus:
  * MODEL_MoE forward takes task_id and returns (out, aux); the load-balance aux
    loss is added with weight --aux-weight.
  * optional task-adaptive intensity (--task-adaptive): for IR-VIS / medical the
    fused luminance is pulled toward the per-pixel MAX of the two sources
    (preserve hot targets / functional bright regions); GFP-PC keeps the original
    mean-intensity (balanced) term. This addresses the cross-task intensity
    conflict noted in MASTER_PLAN.

Usage:
  venv/bin/python train_moe.py \
    --config configs/gfp_pc.json configs/irvis_msrs.json configs/medical_harvard.json \
    --name mm_moe_v1 --epochs 8 --batch-size 64 --crops-per-task 4000 \
    --n-routed 4 --k 2 --aux-weight 0.01 --task-adaptive
"""
import os
import argparse
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Networks.net_moe import MODEL_MoE
from losses import ssim_ir, ssim_vi, RMI_ir, RMI_vi, intensity, joint_grad, ssim_loss
from mm_fusion_dataset import MMFusionDataset

# tasks whose intensity target is per-pixel MAX (salient/functional preservation)
MAX_INTENSITY_TASKS = {"irvis", "medical"}


def intensity_max(a, b, out):
    return torch.mean((torch.maximum(a, b) - out) ** 2)


def _sobel_mag(x):
    import torch.nn.functional as F
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3) / 4
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def ms_max_grad_loss(a, b, out, scales=(1.0, 0.5)):
    """Multi-scale max-gradient (REF Addition B): push |grad(F)| toward the
    per-pixel MAX of the two sources' gradients at several scales -> raises edge
    transfer / detail (Qabf, SF, AG, EN). Drives the decision map to pick the
    sharper source at edges."""
    import torch.nn.functional as F
    loss = 0.0
    for s in scales:
        if s < 1.0:
            aa = F.avg_pool2d(a, 2); bb = F.avg_pool2d(b, 2); oo = F.avg_pool2d(out, 2)
        else:
            aa, bb, oo = a, b, out
        gmax = torch.maximum(_sobel_mag(aa), _sobel_mag(bb))
        loss = loss + (_sobel_mag(oo) - gmax).abs().mean()
    return loss / len(scales)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight", type=float, nargs=7, default=[1, 5, 1, 2.5, 1, 1, 1])
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--lr-decay", type=float, default=0.8)
    ap.add_argument("--in-channel", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=1)
    ap.add_argument("--patch", type=int, default=170)
    ap.add_argument("--crops-per-task", type=int, default=4000)
    ap.add_argument("--fixed-pool", action="store_true",
                    help="fixed crop pool (default: on-the-fly random crops)")
    ap.add_argument("--n-routed", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-shared", type=int, default=1)
    ap.add_argument("--out-channel", type=int, default=16,
                    help="backbone/expert feature width (capacity lever)")
    ap.add_argument("--depth", type=int, default=3,
                    help="transformer (MoE) blocks per scale")
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--fusion-head", default="direct", choices=["direct", "blend", "blenddetail", "inn"],
                    help="blend = MoE-predicted decision map F=w*A+(1-w)*B+res")
    ap.add_argument("--res-scale", type=float, default=0.2)
    ap.add_argument("--routing", default="softmax", choices=["softmax","deepseek"])
    ap.add_argument("--ms-grad", type=float, default=0.0,
                    help="weight of multi-scale max-gradient loss (REF Addition B; detail/Qabf/SF)")
    ap.add_argument("--aux-weight", type=float, default=0.01)
    ap.add_argument("--no-task-cond", action="store_true",
                    help="implicit (input-only) routing ablation (TITA-style)")
    ap.add_argument("--out-scale", action="store_true",
                    help="scale FFN output by 1/(n_shared+1) to control residual magnitude")
    ap.add_argument("--no-task-bias", action="store_true",
                    help="disable per-task stem channel bias")
    ap.add_argument("--task-adaptive", action="store_true",
                    help="task-adaptive max-intensity for irvis/medical")
    ap.add_argument("--loss-mode", default="orig", choices=["orig", "maxfuse"],
                    help="maxfuse = contrast-preserving (toward per-pixel max)")
    ap.add_argument("--ssim-target", default="max", choices=["max", "balanced"],
                    help="maxfuse SSIM target: 'max'=ssim(out,max(a,b)) (higher contrast), "
                         "'balanced'=ssim to both sources")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--init-from", default="", help="load weights from this ckpt before training (staged specialise)")
    ap.add_argument("--only-task", default="", help="train ONLY this task (per-task quota=only this)")
    return ap.parse_args()


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(f"models/{args.name}", exist_ok=True)
    os.makedirs(f"logs/{args.name}", exist_ok=True)
    with open(f"models/{args.name}/args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    cpt = args.crops_per_task
    if args.only_task:                                    # IR-VIS-only specialisation
        import mm_fusion_data as _mfd
        _tasks = [_mfd.load_config(c)["task"] for c in args.config]
        cpt = {t: (args.crops_per_task if t == args.only_task else 0) for t in _tasks}
    ds = MMFusionDataset(args.config, split="train", patch=args.patch,
                         crops_per_task=cpt, random_crop=not args.fixed_pool)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True)
    counts = {}
    for tid, *_ in ds.index:
        counts[ds.tasks[tid]] = counts.get(ds.tasks[tid], 0) + 1
    print(f"train crops: {len(ds)}  batches/epoch: {len(dl)}  per-task: {counts}")
    print(f"tasks (id order): {ds.tasks}  task_adaptive={args.task_adaptive}")

    model = MODEL_MoE(in_channel=args.in_channel, window_size=args.window_size,
                      out_channel=args.out_channel, depth=args.depth,
                      num_heads=args.num_heads,
                      n_tasks=len(ds.tasks), n_routed=args.n_routed, k=args.k,
                      n_shared=args.n_shared, task_cond=not args.no_task_cond,
                      out_scale=args.out_scale,
                      use_task_bias=not args.no_task_bias,
                      fusion_head=args.fusion_head, res_scale=args.res_scale,
                      routing=args.routing).to(dev)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"loaded init weights from {args.init_from}")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"MoE params: {nparam/1e6:.2f}M  routed={args.n_routed} k={args.k} "
          f"shared={args.n_shared} task_cond={not args.no_task_cond}")
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.lr_decay)
    w = args.weight
    log = []
    gstep = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {k: 0.0 for k in ("loss", "content", "structure", "ssim", "rmi",
                                 "intensity", "grad", "aux")}
        nb = 0
        for batch in dl:
            a = batch["src_a"].to(dev, non_blocking=True)
            b = batch["src_b"].to(dev, non_blocking=True)
            tid = batch["task_id"].to(dev, non_blocking=True)
            inp = torch.cat((a, b), dim=1)
            out, aux = model(inp, tid)

            if args.loss_mode == "maxfuse":
                # contrast-preserving: intensity toward per-pixel max(a,b) (salient
                # source wins each pixel -> preserves source dynamic range instead
                # of averaging it down); SSIM/RMI BALANCED to both sources (the
                # bench scores SSIM to both); gradient toward joint max edges. All
                # weights tunable via --weight so contrast/detail can be pushed.
                if args.ssim_target == "max":
                    loss_ssim = (w[0] + w[1]) * 0.5 * ssim_loss(out, a, b)   # ssim(out, max(a,b))
                else:
                    loss_ssim = w[0] * ssim_ir(out, a) + w[1] * ssim_vi(out, b)
                loss_grad = w[6] * joint_grad(a, b, out).mean()
                if args.ms_grad > 0:
                    loss_grad = loss_grad + args.ms_grad * ms_max_grad_loss(a, b, out)
                L_structure = loss_ssim + loss_grad
                loss_rmi = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
                loss_int = w[4] * torch.mean((torch.maximum(a, b) - out) ** 2)
                L_content = loss_rmi + loss_int
            else:
                loss_ssim = w[0] * ssim_ir(out, a) + w[1] * ssim_vi(out, b)
                loss_grad = w[6] * joint_grad(a, b, out).mean()
                L_structure = loss_ssim + loss_grad
                loss_rmi = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
                if args.task_adaptive:
                    is_max = torch.tensor(
                        [ds.tasks[t] in MAX_INTENSITY_TASKS for t in tid.tolist()],
                        device=dev).view(-1, 1, 1, 1)
                    mx = (torch.maximum(a, b) - out) ** 2
                    mn = 0.5 * ((a - out) ** 2 + (b - out) ** 2)
                    loss_int = torch.mean(torch.where(is_max, mx, mn))
                else:
                    loss_int = 0.5 * w[4] * intensity(a, out) + 0.5 * w[5] * intensity(b, out)
                L_content = loss_rmi + loss_int
            loss = L_content + args.alpha * L_structure + args.aux_weight * aux

            opt.zero_grad()
            loss.backward()
            opt.step()

            agg["loss"] += loss.item(); agg["content"] += L_content.item()
            agg["structure"] += L_structure.item(); agg["ssim"] += loss_ssim.item()
            agg["rmi"] += loss_rmi.item(); agg["intensity"] += float(loss_int)
            agg["grad"] += loss_grad.item(); agg["aux"] += float(aux); nb += 1
            gstep += 1
            if args.max_steps and gstep >= args.max_steps:
                break

        for k in agg:
            agg[k] /= max(nb, 1)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{args.epochs}  lr {cur_lr:.2e}  loss {agg['loss']:.4f} "
              f"(content {agg['content']:.4f} structure {agg['structure']:.4f} "
              f"aux {agg['aux']:.4f})  {dt:.1f}s")
        log.append({"epoch": epoch + 1, "lr": cur_lr, **agg, "sec": dt})
        pd.DataFrame(log).to_csv(f"logs/{args.name}/log.csv", index=False)
        torch.save(model.state_dict(), f"models/{args.name}/model_{epoch+1}.pth")
        sched.step()
        if args.max_steps and gstep >= args.max_steps:
            print("hit max-steps, stopping"); break

    print("training done")


if __name__ == "__main__":
    main()
