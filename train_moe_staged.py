"""Staged training with IR-VIS specialisation (user idea #1) — new file.

Phase 1 (epochs 1..--phase1): train the WHOLE model jointly on all 3 tasks
(per-task fusion heads). This is the shared model the other two modalities keep.

Phase 2 (epochs --phase1+1..--epochs): FREEZE the backbone + gfp/medical heads;
train ONLY the IR-VIS head on IR-VIS data, with detail/gradient emphasis. Because
everything the other two tasks use is frozen, medical/gfp outputs are byte-for-byte
unchanged (no forgetting); IR-VIS gets a dedicated, specialised head.

IR-VIS is task index 1 (config order: gfp_pc, irvis, medical). Phase-2 data is
IR-VIS-only via a per-task quota dict {gfp:0, irvis:N, medical:0} which keeps the
task_id = 1 consistent with the model's IR-VIS head.
"""
import os, argparse, time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Networks.net_moe import MODEL_MoE
from losses import ssim_ir, ssim_vi, RMI_ir, RMI_vi, joint_grad, ssim_loss
from train_moe import ms_max_grad_loss
from mm_fusion_dataset import MMFusionDataset
import mm_fusion_data as mfd


def maxfuse(out, a, b, w, alpha, ms_grad=0.0):
    loss_ssim = (w[0] + w[1]) * 0.5 * ssim_loss(out, a, b)
    loss_grad = w[6] * joint_grad(a, b, out).mean()
    if ms_grad > 0:
        loss_grad = loss_grad + ms_grad * ms_max_grad_loss(a, b, out)
    loss_rmi = w[2] * RMI_ir(out, a) + w[3] * RMI_vi(out, b)
    loss_int = w[4] * torch.mean((torch.maximum(a, b) - out) ** 2)
    return loss_rmi + loss_int + alpha * (loss_ssim + loss_grad)


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=26)
    p.add_argument("--phase1", type=int, default=18, help="joint epochs before IR-VIS specialisation")
    p.add_argument("--irvis-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=5e-4, help="phase-2 IR-VIS-head LR")
    p.add_argument("--lr-decay", type=float, default=0.8)
    p.add_argument("--weight", type=float, nargs=7, default=[2, 2, 2, 2, 4, 0, 3])
    p.add_argument("--weight2", type=float, nargs=7, default=[3, 3, 2, 2, 4, 0, 4],
                   help="phase-2 IR-VIS loss weights (ssim+grad emphasis)")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--ms-grad2", type=float, default=1.0, help="phase-2 IR-VIS ms-grad")
    p.add_argument("--crops-per-task", type=int, default=4000)
    p.add_argument("--patch", type=int, default=170)
    p.add_argument("--out-channel", type=int, default=64)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--n-routed", type=int, default=12)
    p.add_argument("--fusion-head", default="blenddetail")
    p.add_argument("--res-scale", type=float, default=0.3)
    p.add_argument("--routing", default="softmax")
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    a = args_()
    dev = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(f"models/{a.name}", exist_ok=True); os.makedirs(f"logs/{a.name}", exist_ok=True)
    with open(f"models/{a.name}/args.txt", "w") as f:
        for k, v in vars(a).items():
            f.write(f"{k}: {v}\n")
    tasks = [mfd.load_config(c)["task"] for c in a.config]
    model = MODEL_MoE(in_channel=2, n_tasks=len(a.config), out_channel=a.out_channel,
                      depth=3, num_heads=8, window_size=a.window_size, n_routed=a.n_routed,
                      k=2, n_shared=1, task_cond=True, out_scale=True,
                      fusion_head=a.fusion_head, res_scale=a.res_scale, routing=a.routing,
                      per_task_head=True).to(dev)
    print(f"staged MoE {sum(p.numel() for p in model.parameters())/1e6:.2f}M  tasks={tasks} "
          f"irvis=idx{a.irvis_task}")
    log = []

    def run_epoch(dl, opt, w, ms_grad):
        model.train(); tot = 0.0; n = 0
        for batch in dl:
            src_a = batch["src_a"].to(dev, non_blocking=True)
            src_b = batch["src_b"].to(dev, non_blocking=True)
            tid = batch["task_id"].to(dev, non_blocking=True)
            out, aux = model(torch.cat((src_a, src_b), 1), tid)
            loss = maxfuse(out, src_a, src_b, w, a.alpha, ms_grad) + a.aux_weight * aux
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        return tot / max(n, 1)

    # ---------- Phase 1: joint ----------
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sch = optim.lr_scheduler.StepLR(opt, 1, gamma=a.lr_decay)
    for ep in range(a.phase1):
        ds = MMFusionDataset(a.config, "train", a.patch, a.crops_per_task, random_crop=True, seed=ep)
        dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
        t0 = time.time(); l = run_epoch(dl, opt, a.weight, 0.0)
        print(f"[P1] ep{ep+1}/{a.phase1} loss{l:.4f} {time.time()-t0:.0f}s"); sch.step()
        log.append({"phase": 1, "epoch": ep + 1, "loss": l})
    torch.save(model.state_dict(), f"models/{a.name}/phase1.pth")

    # ---------- Phase 2: freeze all but IR-VIS head; IR-VIS-only fine-tune ----------
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(f"heads.{a.irvis_task}.")
    tr = [p for p in model.parameters() if p.requires_grad]
    print(f"[P2] trainable params (IR-VIS head): {sum(p.numel() for p in tr)/1e3:.1f}K")
    opt2 = optim.Adam(tr, lr=a.lr2)
    sch2 = optim.lr_scheduler.StepLR(opt2, 1, gamma=a.lr_decay)
    ir_quota = {t: (a.crops_per_task if i == a.irvis_task else 0) for i, t in enumerate(tasks)}
    for ep in range(a.phase1, a.epochs):
        ds = MMFusionDataset(a.config, "train", a.patch, ir_quota, random_crop=True, seed=ep)
        dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
        t0 = time.time(); l = run_epoch(dl, opt2, a.weight2, a.ms_grad2)
        print(f"[P2-irvis] ep{ep+1}/{a.epochs} loss{l:.4f} {time.time()-t0:.0f}s"); sch2.step()
        log.append({"phase": 2, "epoch": ep + 1, "loss": l})
        pd.DataFrame(log).to_csv(f"logs/{a.name}/log.csv", index=False)
        torch.save(model.state_dict(), f"models/{a.name}/model_{ep+1}.pth")
    print("training done")


if __name__ == "__main__":
    main()
