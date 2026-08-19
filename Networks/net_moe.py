"""MoE-enhanced MDFNet (U-MDFNet, Phase-2 main innovation).

Extends the released MDFNet backbone (multi-scale ACM + window-attention
Transformer) with a Mixture-of-Experts FFN, following the design in
knowledge/paper/SURVEY_AND_MOE_PLAN.md (TC-MoA + MoE-Fusion + DeepSeek-MoE):

  * The Transformer block's FFN (Mlp) -> MoE-FFN:
        1 SHARED expert (always-on, cross-task common fusion ability)
      + N ROUTED experts (task/modality specific), top-k gated.
  * Routing condition = token feature (+ task embedding) -> TC-MoA style
    task-conditioned routing. Set task_cond=False for the TITA-style
    "implicit, input-only" routing ablation.
  * Load-balance aux loss (Switch/GShard) prevents expert collapse on the small
    fusion datasets; the shared expert is the additional collapse safeguard.
  * A learned per-task channel bias is added after the stem so the dense path is
    also task-aware (fair comparison point for the routing ablation).

The network keeps the SAME I/O contract as net.MODEL: input (B,2,H,W) of two Y
maps, output (B,1,H,W) fused Y in [-1,1] via tanh. forward() additionally needs
a task_id tensor (B,) and returns (out, aux_loss).

Only net_moe.py is new; net.py / layers.py are untouched.
"""
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from .net import (Basic3x3, Basic1x1, WindowAttention,
                  window_partition, window_reverse, PatchEmbed)


class SDPAWindowAttention(WindowAttention):
    """Weight-identical drop-in for WindowAttention that runs the attention core
    via F.scaled_dot_product_attention (Route-A infra optimisation).

    The manual `q@k.T -> +bias -> softmax -> @v` (three bmm/softmax kernels,
    profiled at ~11.6% CUDA) is replaced by ONE fused SDPA call, which on H800
    dispatches to the FlashAttention / mem-efficient kernel. The relative-position
    bias (and the optional shift mask) is passed as SDPA's additive `attn_mask`.
    Numerically equivalent to the parent (SDPA applies the 1/sqrt(d) scale
    internally, so q is NOT pre-scaled here). Same params -> load either forward
    from the same checkpoint.
    """
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B_, nH, N, hd)
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()       # (nH, N, N)
        attn_bias = rpb.unsqueeze(0)                       # (1, nH, N, N) broadcast over B_
        if mask is not None:                              # shifted-window case (unused here)
            nW = mask.shape[0]
            attn_bias = (attn_bias.view(1, 1, self.num_heads, N, N)
                         + mask.view(nW, 1, 1, N, N)).expand(
                             nW, B_ // nW, self.num_heads, N, N).reshape(B_, self.num_heads, N, N)
        p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
        out = out.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


class AffineCoupling(nn.Module):
    """RealNVP-style invertible affine coupling (used for the INN detail branch).
    Splits channels in half; one half predicts an affine (scale,shift) for the
    other -> information-preserving (invertible) high-frequency transform, the
    CDDFuse detail-encoder trick that lifts EN/SD/SF at equal capacity."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c // 2, c, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c, c - c // 2 + c // 2, 3, padding=1))

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        s, t = self.net(x1).chunk(2, 1)
        return torch.cat([x1, x2 * torch.tanh(s).exp() + t], 1)


class INNDetail(nn.Module):
    """Invertible high-frequency detail branch: lift (A,B)->C ch, K coupling
    blocks, project to a 1-ch detail map (tanh). Added to the base blend so the
    fused image can exceed the convex-blend detail/contrast ceiling."""
    def __init__(self, in_ch=2, c=32, n=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, c, 3, padding=1)
        self.blocks = nn.ModuleList([AffineCoupling(c) for _ in range(n)])
        self.proj = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, ab):
        h = self.lift(ab)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.proj(h))


def _sobel_mag(x):
    """Per-pixel Sobel gradient magnitude (B,1,H,W) -> (B,1,H,W)."""
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3) / 4
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class Expert(nn.Module):
    """A single FFN expert (same shape as the original Mlp)."""
    def __init__(self, dim, hidden, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MoEFFN(nn.Module):
    """Shared-expert + top-k routed-expert FFN with load-balance aux loss.

    x: (B, N, C) tokens.  task_emb: (B, C) or None (broadcast over tokens).
    Returns y (B,N,C); the load-balance aux loss is stashed in self.aux.
    """
    def __init__(self, dim, mlp_ratio=4., n_routed=4, k=2, n_shared=1,
                 task_cond=True, drop=0., out_scale=False, routing="softmax",
                 bias_gamma=1e-3):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.n_routed, self.k = n_routed, min(k, max(n_routed, 1))
        self.task_cond = task_cond
        self.n_shared = n_shared
        self.routing = routing      # "softmax" (Switch-aux) | "deepseek" (aux-loss-free)
        self.bias_gamma = bias_gamma
        self.out_scale = out_scale
        self.combine = "sparse"     # "sparse" (per-expert loop) | "batched" (fused einsum) | "grouped" (capacity)
        self.cap_factor = 1.25      # grouped: per-expert capacity = cap_factor * T * k / E
        self.shared = nn.ModuleList([Expert(dim, hidden, drop=drop)
                                     for _ in range(n_shared)])
        self.experts = nn.ModuleList([Expert(dim, hidden, drop=drop)
                                      for _ in range(n_routed)])
        self.gate = nn.Linear(dim, n_routed, bias=False) if n_routed > 0 else None
        # DeepSeek-V3 aux-loss-free balancing: per-expert bias added to the
        # affinity for top-k SELECTION only (not the gate value); updated each
        # step by +/-gamma toward balanced load. Buffer (not a grad parameter).
        if routing == "deepseek" and n_routed > 0:
            self.register_buffer("ebias", torch.zeros(n_routed))
            self.register_buffer("cload", torch.full((n_routed,), 1.0 / n_routed))  # EMA load
        self.aux = torch.tensor(0.0)

    def forward(self, x, task_emb=None):
        B, N, C = x.shape
        flat = x.reshape(-1, C)                      # (B*N, C)
        out = flat.new_zeros(flat.shape)
        for s in self.shared:                        # always-on shared experts
            out = out + s(flat)

        if self.n_routed > 0:
            gate_in = flat
            if self.task_cond and task_emb is not None:
                te = task_emb[:, None, :].expand(B, N, C).reshape(-1, C)
                gate_in = flat + te                  # additive task conditioning
            logits = self.gate(gate_in)
            if self.routing == "deepseek":
                # DeepSeek-V3: sigmoid affinity; select top-k by (affinity+bias),
                # but gate value = normalized affinity of the selected experts.
                aff = torch.sigmoid(logits)                  # (B*N, n_routed)
                sel_score = aff + self.ebias                 # bias only steers selection
                _, topi = sel_score.topk(self.k, dim=-1)
                gsel = torch.gather(aff, 1, topi)
                gsel = gsel / (gsel.sum(-1, keepdim=True) + 1e-9)
                gates = torch.zeros_like(aff).scatter(1, topi, gsel)
            else:
                probs = F.softmax(logits, dim=-1)            # (B*N, n_routed)
                topv, topi = probs.topk(self.k, dim=-1)
                topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
                gates = torch.zeros_like(probs).scatter(1, topi, topv)
            if self.combine == "batched":
                # BATCHED expert compute (operator fusion, "sonic-moe" style):
                # stack the E experts' weights and run 2 batched GEMMs (einsum)
                # instead of a Python loop of E tiny linears + index_add. Fuses
                # thousands of tiny mm/index/copy ops (the profiled bottleneck)
                # into 2 big kernels; torch.compile fuses further. Dense over all
                # E experts but far fewer launches -> faster in practice here.
                W1 = torch.stack([e.fc1.weight for e in self.experts])   # (E,H,C)
                b1 = torch.stack([e.fc1.bias for e in self.experts])     # (E,H)
                W2 = torch.stack([e.fc2.weight for e in self.experts])   # (E,C,H)
                b2 = torch.stack([e.fc2.bias for e in self.experts])     # (E,C)
                h = torch.einsum('tc,ehc->eth', flat, W1) + b1[:, None, :]
                h = self.experts[0].act(h)
                y = torch.einsum('eth,ech->etc', h, W2) + b2[:, None, :]  # (E,T,C)
                out = out + torch.einsum('te,etc->tc', gates, y)
            elif self.combine == "grouped":
                # GROUPED capacity dispatch (Route-C infra innovation): keep the
                # SPARSE compute budget (each token hits only its top-k experts,
                # not all E), but replace the E-way Python loop of tiny linears +
                # nonzero + index_add (the profiled 45% mm / 18% index / 7% copy)
                # with a fixed-capacity, right-padded layout so ALL experts run in
                # TWO batched GEMMs. Overflow beyond capacity is dropped (GShard).
                T = flat.shape[0]
                E, kk = self.n_routed, self.k
                cap = max(1, int(self.cap_factor * T * kk / E))
                disp_t = torch.arange(T, device=flat.device).view(T, 1).expand(T, kk).reshape(-1)
                disp_e = topi.reshape(-1)                        # (D=T*k,) target expert
                disp_g = torch.gather(gates, 1, topi).reshape(-1)  # (D,) routing weight
                # position of each dispatch within its expert bucket, via a SORT
                # (group dispatches by expert) + a tiny per-expert offset cumsum.
                # Avoids the O(D*E) one-hot cumsum, which is pathological in eager.
                D = disp_e.shape[0]
                order = torch.argsort(disp_e)                    # dispatches grouped by expert
                counts = torch.bincount(disp_e, minlength=E)     # (E,) tokens per expert
                offsets = counts.cumsum(0) - counts              # (E,) group start in sorted order
                ar = torch.arange(D, device=flat.device)
                pos_sorted = ar - offsets[disp_e[order]]         # rank within expert (sorted order)
                pos_in_e = torch.empty(D, dtype=torch.long, device=flat.device)
                pos_in_e[order] = pos_sorted                     # scatter back to dispatch order
                keep = pos_in_e < cap                            # capacity mask
                dk = keep.nonzero(as_tuple=True)[0]
                ek, pk, tk = disp_e[dk], pos_in_e[dk], disp_t[dk]
                buf = flat.new_zeros(E, cap, C)
                buf[ek, pk] = flat[tk]                           # gather tokens -> (E,cap,C)
                W1 = torch.stack([e.fc1.weight for e in self.experts])   # (E,H,C)
                b1 = torch.stack([e.fc1.bias for e in self.experts])     # (E,H)
                W2 = torch.stack([e.fc2.weight for e in self.experts])   # (E,C,H)
                b2 = torch.stack([e.fc2.bias for e in self.experts])     # (E,C)
                h = self.experts[0].act(torch.bmm(buf, W1.transpose(1, 2)) + b1[:, None, :])
                y = torch.bmm(h, W2.transpose(1, 2)) + b2[:, None, :]    # (E,cap,C)
                routed = flat.new_zeros(flat.shape)
                routed.index_add_(0, tk, y[ek, pk] * disp_g[dk].unsqueeze(-1))
                out = out + routed
            else:
                # SPARSE top-k dispatch: each expert computes ONLY its routed
                # tokens (compute-light but many tiny ops + index_add overhead).
                routed = flat.new_zeros(flat.shape)
                for e in range(self.n_routed):
                    we = gates[:, e]
                    idx = torch.nonzero(we > 0, as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    ye = self.experts[e](flat[idx]) * we[idx].unsqueeze(-1)
                    routed.index_add_(0, idx, ye)
                out = out + routed
            if self.routing == "deepseek":
                # aux-loss-free: update per-expert bias toward balanced load
                # (no gradient term). overloaded -> bias down; underloaded -> up.
                if self.training:
                    with torch.no_grad():
                        load = torch.bincount(topi.reshape(-1), minlength=self.n_routed).float()
                        load = load / load.sum().clamp_min(1.0)
                        # EMA-smooth per-expert load (small-batch counts are noisy)
                        self.cload.mul_(0.9).add_(0.1 * load)
                        self.ebias += self.bias_gamma * torch.sign(self.cload.mean() - self.cload)
                # tiny complementary sequence-wise aux (DeepSeek-V3 alpha=1e-4)
                with torch.no_grad():
                    f = torch.bincount(topi[:, 0], minlength=self.n_routed).float()
                    f = f / f.sum().clamp_min(1.0)
                self.aux = self.n_routed * torch.sum(f * torch.sigmoid(logits).mean(0))
            else:
                # Switch-style load-balance aux: N * sum_i f_i * P_i
                with torch.no_grad():
                    f = torch.bincount(topi[:, 0], minlength=self.n_routed).float()
                    f = f / f.sum().clamp_min(1.0)
                self.aux = self.n_routed * torch.sum(f * probs.mean(0))
        else:
            self.aux = out.new_zeros(())

        if self.out_scale:
            out = out / float(self.n_shared + 1)
        return out.view(B, N, C)


class MoETransformerBlock(nn.Module):
    """TransformerBlock with FFN replaced by MoEFFN (window attention kept)."""
    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 n_routed=4, k=2, n_shared=1, task_cond=True, out_scale=False,
                 routing="softmax", attn_impl="vanilla",
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        AttnCls = SDPAWindowAttention if attn_impl == "sdpa" else WindowAttention
        self.attn = AttnCls(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MoEFFN(dim, mlp_ratio=mlp_ratio, n_routed=n_routed, k=k,
                          n_shared=n_shared, task_cond=task_cond, drop=drop,
                          out_scale=out_scale, routing=routing)
        self.register_buffer("attn_mask", None)

    def forward(self, x, x_size, task_emb=None):
        # view-consistent layout in/out (matches the proven dense backbone), with
        # REAL window attention: pad H,W up to a multiple of window_size, do
        # regular (non-shifted) window self-attention, crop back. window_size=1
        # degenerates to per-pixel linear; ws>1 gives genuine spatial context.
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        shortcut = x
        xn = self.norm1(x.reshape(H * W * B, C)).view(B, H, W, C)

        ws = self.window_size
        if ws > 1:
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h or pad_w:
                xn = F.pad(xn.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h),
                           mode="reflect").permute(0, 2, 3, 1).contiguous()
            Hp, Wp = H + pad_h, W + pad_w
            xw = window_partition(xn, ws).view(-1, ws * ws, C)
            aw = self.attn(xw, mask=None).view(-1, ws, ws, C)
            xr = window_reverse(aw, ws, Hp, Wp)
            if pad_h or pad_w:
                xr = xr[:, :H, :W, :].contiguous()
        else:
            xw = window_partition(xn, 1).view(-1, 1, C)
            xr = window_reverse(self.attn(xw, mask=None).view(-1, 1, 1, C), 1, H, W)

        x = shortcut + xr                                  # (B,H,W,C)
        y = self.norm2(x.reshape(B * H * W, C)).view(B, H, W, C)
        y = self.mlp(y.reshape(B, H * W, C), task_emb).view(B, H, W, C)
        x = x + y                                          # MoE FFN residual
        return x.view(B, C, H, W)                          # reinterpret (as original)


class MoEBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., n_routed=4, k=2, n_shared=1, task_cond=True,
                 out_scale=False, routing="softmax", attn_impl="vanilla",
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0,  # regular windows (no shift mask)
                mlp_ratio=mlp_ratio, n_routed=n_routed, k=k, n_shared=n_shared,
                task_cond=task_cond, out_scale=out_scale, routing=routing,
                attn_impl=attn_impl, norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size, task_emb=None):
        for blk in self.blocks:
            x = blk(x, x_size, task_emb)
        return x

    def aux_loss(self):
        return sum(b.mlp.aux for b in self.blocks)


class MODEL_MoE(nn.Module):
    def __init__(self, img_size=170, patch_size=4, embed_dim=96, num_heads=8,
                 window_size=1, in_channel=2, out_channel=16, output_channel=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 patch_norm=True, depth=3, n_tasks=3, n_routed=4, k=2, n_shared=1,
                 task_cond=True, out_scale=False, use_task_bias=True,
                 fusion_head="direct", res_scale=0.2, routing="softmax",
                 per_task_head=False, attn_impl="vanilla", norm_layer=nn.LayerNorm):
        super().__init__()
        self.convInput = Basic3x3(in_channel, out_channel)
        self.conv = Basic3x3(out_channel, out_channel)
        self.task_cond = task_cond
        self.use_task_bias = use_task_bias
        # fusion head:
        #   "direct" = original: out = tanh(1x1 conv(feat))  (free-form F)
        #   "blend"  = MoE-predicted decision map: F = w*A + (1-w)*B + res*tanh(r),
        #     w=sigmoid(.). F is anchored to a convex blend of the two sources, so
        #     it inherits their dynamic range (EN/SD) and linearly preserves both
        #     (MI/SCD/CC) while staying clean (Nabf); the residual adds detail.
        #   "blenddetail" = base/detail decomposition (CDDFuse/W-DUALMINE route):
        #     base  = convex blend  w*A+(1-w)*B  (info/intensity -> MI/VIF/SD/SSIM),
        #     detail= a learned high-freq residual, EDGE-GATED by the per-pixel
        #     source-gradient magnitude (a fixed prior) so it is injected ONLY at
        #     real edges -> raises SF/Qabf/AG without artifacts in flat regions
        #     (keeps Nabf/SSIM that the pure blend wins). Gradient-supervised via
        #     the ms-grad loss. Lets F exceed the convex-blend detail ceiling.
        #   per_task_head: give EACH task its own (richer) fusion head, so a task
        #     (e.g. IR-VIS) gets dedicated capacity and can be specialised in a
        #     later stage while the other tasks' heads (+ frozen backbone) stay
        #     fixed -> no forgetting. Each head = 3x3 conv block + 1x1 -> [w,detail].
        self.fusion_head = fusion_head
        self.res_scale = res_scale
        self.per_task_head = per_task_head
        if fusion_head == "inn":
            # base blend head (w) + invertible high-frequency detail branch
            self.head = nn.Conv2d(out_channel, 1, kernel_size=1)   # w_logit only
            self.inn = INNDetail(in_ch=in_channel, c=32, n=4)
        elif fusion_head in ("blend", "blenddetail"):
            if per_task_head:
                self.heads = nn.ModuleList([
                    nn.Sequential(Basic3x3(out_channel, out_channel),
                                  nn.Conv2d(out_channel, 2, 1))
                    for _ in range(n_tasks)])
            else:
                self.head = nn.Conv2d(out_channel, 2, kernel_size=1)   # [w_logit, detail/res]
        else:
            self.convolutional_out = Basic1x1(out_channel, output_channel)

        # task awareness: learned per-task channel bias (FiLM-lite) + router emb
        self.task_bias = nn.Embedding(n_tasks, out_channel)
        nn.init.zeros_(self.task_bias.weight)
        self.task_router_emb = nn.Embedding(n_tasks, out_channel)
        nn.init.normal_(self.task_router_emb.weight, std=0.02)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        pr = self.patch_embed.patches_resolution
        self.basicLayer = MoEBasicLayer(
            dim=out_channel, input_resolution=(pr[0], pr[1]), depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            n_routed=n_routed, k=k, n_shared=n_shared, task_cond=task_cond,
            out_scale=out_scale, routing=routing, attn_impl=attn_impl,
            norm_layer=norm_layer)

    def set_combine(self, mode, cap_factor=None):
        """Switch all MoE-FFN experts between 'sparse' loop, 'batched' fused einsum,
        and 'grouped' capacity dispatch (optionally set the grouped capacity factor)."""
        for blk in self.basicLayer.blocks:
            blk.mlp.combine = mode
            if cap_factor is not None:
                blk.mlp.cap_factor = cap_factor
        return self

    def forward(self, x, task_id):
        if not torch.is_tensor(task_id):
            task_id = torch.full((x.shape[0],), int(task_id), device=x.device,
                                 dtype=torch.long)
        c = self.convInput(x)
        if self.use_task_bias:
            c = c + self.task_bias(task_id)[:, :, None, None]  # task-aware stem
        te = self.task_router_emb(task_id) if self.task_cond else None

        aux = 0.0
        outs = []
        feat = c
        for _ in range(3):                                    # 3 pseudo-scales
            feat = self.conv(feat)
            t = self.basicLayer(feat, (feat.shape[2], feat.shape[3]), te)
            outs.append(self.conv(t))
            aux = aux + self.basicLayer.aux_loss()
        feat_sum = outs[0] + outs[1] + outs[2]
        if self.fusion_head == "inn":
            w = torch.sigmoid(self.head(feat_sum))            # base blend weight
            A, B = x[:, 0:1], x[:, 1:2]
            base = w * A + (1.0 - w) * B
            # edge-gated invertible detail (only at real edges -> keep flat clean)
            gmax = torch.maximum(_sobel_mag(A), _sobel_mag(B))
            gate = gmax / (gmax.amax(dim=(2, 3), keepdim=True) + 1e-6)
            out = (base + self.res_scale * gate * self.inn(x)).clamp(0.0, 1.0)
        elif self.fusion_head in ("blend", "blenddetail"):
            if self.per_task_head:
                h = 0
                for t in torch.unique(task_id):        # masked sum (autograd-safe)
                    mt = (task_id == t).view(-1, 1, 1, 1).float()
                    h = h + mt * self.heads[int(t)](feat_sum)
            else:
                h = self.head(feat_sum)                   # (B,2,H,W)
            w = torch.sigmoid(h[:, 0:1])                  # per-pixel A-weight
            A, B = x[:, 0:1], x[:, 1:2]
            base = w * A + (1.0 - w) * B
            detail = self.res_scale * torch.tanh(h[:, 1:2])
            if self.fusion_head == "blenddetail":
                # edge gate from source gradients (fixed prior in [0,1]) confines
                # the detail residual to real edges -> SF/Qabf up, flat regions
                # stay clean (Nabf/SSIM safe).
                gmax = torch.maximum(_sobel_mag(A), _sobel_mag(B))
                gate = gmax / (gmax.amax(dim=(2, 3), keepdim=True) + 1e-6)
                out = (base + gate * detail).clamp(0.0, 1.0)
            else:
                out = (base + detail).clamp(0.0, 1.0)
        else:
            out = self.convolutional_out(feat_sum)
        return out, aux
