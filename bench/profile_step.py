"""Profile one v3 training step to find the runtime bottleneck (Track A).
Reports per-op CUDA time (top-15) + mean step time, for a chosen config, and
lets us A/B optimizations (bf16, compile, sdpa) via flags."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, torch
import torch.optim as optim
from Networks.net_moe import MODEL_MoE
from losses import ssim_loss, RMI_ir, RMI_vi, joint_grad

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=10)
ap.add_argument("--oc", type=int, default=96)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--nr", type=int, default=12)
ap.add_argument("--amp", action="store_true", help="bf16 autocast")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--profile", action="store_true")
ap.add_argument("--combine", default="sparse", choices=["sparse", "batched", "grouped"])
ap.add_argument("--attn", default="vanilla", choices=["vanilla", "sdpa"])
ap.add_argument("--cap-factor", type=float, default=1.25, help="grouped-MoE capacity factor")
ap.add_argument("--compile-mode", default="default",
                choices=["default", "reduce-overhead", "max-autotune"])
a = ap.parse_args()
dev = "cuda:0"

m = MODEL_MoE(in_channel=2, n_tasks=3, out_channel=a.oc, depth=a.depth, window_size=8,
              n_routed=a.nr, k=2, n_shared=1, out_scale=True, fusion_head="blend",
              res_scale=0.0, attn_impl=a.attn).to(dev)
m.set_combine(a.combine, cap_factor=a.cap_factor)
if a.compile:
    m = torch.compile(m, mode=a.compile_mode)
opt = optim.Adam(m.parameters(), lr=1e-3)
x = torch.rand(a.bs, 2, 170, 170, device=dev)
tid = torch.randint(0, 3, (a.bs,), device=dev)
adev, bdev = x[:, 0:1], x[:, 1:2]


def step():
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
        out, aux = m(x, tid)
        loss = (ssim_loss(out, adev, bdev) + RMI_ir(out, adev) + RMI_vi(out, bdev)
                + torch.mean((torch.maximum(adev, bdev) - out) ** 2)
                + joint_grad(adev, bdev, out).mean() + 0.01 * aux)
    loss.backward(); opt.step()
    return float(loss)


for _ in range(5):  # warmup (+compile)
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(a.steps):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / a.steps * 1000
print(f"[cfg bs{a.bs} oc{a.oc} d{a.depth} nr{a.nr} amp{a.amp} compile{a.compile}] "
      f"mean step {dt:.1f} ms  peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

if a.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(8):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
