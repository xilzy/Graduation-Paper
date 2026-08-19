"""Correctness checks for the Route-A (SDPA) and Route-C (grouped MoE) infra
optimisations: both must be numerically equivalent to the proven vanilla path
(SDPA exactly; grouped up to capacity drops -> use a large cap_factor so no drop).
Run on one GPU: CUDA_VISIBLE_DEVICES=<g> python bench/verify_infra.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from Networks.net_moe import MODEL_MoE

dev = "cuda:0"
torch.manual_seed(0)
cfg = dict(in_channel=2, n_tasks=3, out_channel=96, depth=4, window_size=8,
           n_routed=12, k=2, n_shared=1, out_scale=True, fusion_head="blend", res_scale=0.0)
x = torch.rand(4, 2, 170, 170, device=dev)
tid = torch.randint(0, 3, (4,), device=dev)


def run(model):
    model.eval()
    with torch.no_grad():
        out, aux = model(x, tid)
    return out


# ---- Route A: SDPA vs vanilla (weight-identical) ----
mv = MODEL_MoE(**cfg, attn_impl="vanilla").to(dev)
ms = MODEL_MoE(**cfg, attn_impl="sdpa").to(dev)
ms.load_state_dict(mv.state_dict())          # same weights
ov, osdpa = run(mv), run(ms)
d_attn = (ov - osdpa).abs().max().item()
print(f"[A] SDPA vs vanilla   max|Δ| = {d_attn:.2e}  ({'OK' if d_attn < 1e-4 else 'FAIL'})")

# ---- Route C: grouped (large capacity) vs sparse (same model) ----
mv.set_combine("sparse")
o_sparse = run(mv)
mv.set_combine("grouped", cap_factor=8.0)    # cap = 8*T*k/E = 1.33*T >= T -> zero drop
o_group = run(mv)
d_grp = (o_sparse - o_group).abs().max().item()
print(f"[C] grouped(cap8=no-drop) vs sparse   max|Δ| = {d_grp:.2e}  ({'OK' if d_grp < 1e-4 else 'FAIL'})")

# ---- Route C at production capacity (1.25) -> report drop rate & deviation ----
mv.set_combine("grouped", cap_factor=1.25)
o_g125 = run(mv)
d_125 = (o_sparse - o_g125).abs().max().item()
print(f"[C] grouped(cap1.25) vs sparse   max|Δ| = {d_125:.2e}  (small = few tokens dropped)")
