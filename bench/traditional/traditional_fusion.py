#!/usr/bin/env python
"""Traditional (non-learning) image-fusion baselines for the Graduation-Paper benchmark.

Methods: LP, DWT, DTCWT, GTF, NSCT* (Laplacian+directional approximation).
All pure CPU (numpy/scipy/pywt/dtcwt). Inputs are 8-bit grayscale A/B pairs.

Usage:
  python traditional_fusion.py --method LP --task irvis
  python traditional_fusion.py --method ALL --task all
"""
import os
import argparse
import numpy as np
from PIL import Image

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
TASKS = ["irvis", "medical", "gfp_pc"]


# --------------------------------------------------------------------------- IO
def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def resize_to(X, ref):
    if X.shape != ref.shape:
        return np.asarray(
            Image.fromarray(np.clip(X, 0, 255).astype("uint8")).resize(
                (ref.shape[1], ref.shape[0]), Image.BILINEAR),
            dtype=np.float64)
    return X


def save_gray(F, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Fc = np.clip(np.round(F), 0, 255).astype("uint8")
    Image.fromarray(Fc).save(path)


# --------------------------------------------------------------------------- LP
def _gauss_kernel():
    k = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    k = k / k.sum()
    return np.outer(k, k)


def _conv2(img, ker):
    from scipy.ndimage import convolve
    return convolve(img, ker, mode="reflect")


def _pyr_down(img):
    g = _conv2(img, _gauss_kernel())
    return g[::2, ::2]


def _pyr_up(img, shape):
    h, w = shape
    up = np.zeros((h, w), dtype=np.float64)
    up[::2, ::2] = img
    # x4 to compensate for the 1/4 of pixels that are nonzero
    return _conv2(up, _gauss_kernel()) * 4.0


def laplacian_pyramid(img, levels):
    gp = [img]
    for _ in range(levels):
        gp.append(_pyr_down(gp[-1]))
    lp = []
    for i in range(levels):
        up = _pyr_up(gp[i + 1], gp[i].shape)
        lp.append(gp[i] - up)
    lp.append(gp[-1])  # top (residual low-pass)
    return lp


def reconstruct_pyramid(lp):
    cur = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        cur = lp[i] + _pyr_up(cur, lp[i].shape)
    return cur


def fuse_LP(A, B, levels=4):
    la = laplacian_pyramid(A, levels)
    lb = laplacian_pyramid(B, levels)
    fused = []
    for i in range(levels):
        # detail bands: max-abs
        sel = np.abs(la[i]) >= np.abs(lb[i])
        fused.append(np.where(sel, la[i], lb[i]))
    # top level: average
    fused.append(0.5 * (la[-1] + lb[-1]))
    return reconstruct_pyramid(fused)


# -------------------------------------------------------------------------- DWT
def fuse_DWT(A, B, wavelet="db2", levels=2):
    import pywt
    ca = pywt.wavedec2(A, wavelet, level=levels)
    cb = pywt.wavedec2(B, wavelet, level=levels)
    # approximation: mean
    fcoeffs = [0.5 * (ca[0] + cb[0])]
    for da, db in zip(ca[1:], cb[1:]):
        bands = []
        for sa, sb in zip(da, db):
            sel = np.abs(sa) >= np.abs(sb)
            bands.append(np.where(sel, sa, sb))
        fcoeffs.append(tuple(bands))
    F = pywt.waverec2(fcoeffs, wavelet)
    return F[:A.shape[0], :A.shape[1]]


# ------------------------------------------------------------------------ DTCWT
def fuse_DTCWT(A, B, levels=4):
    import dtcwt
    t = dtcwt.Transform2d()
    ta = t.forward(A, nlevels=levels)
    tb = t.forward(B, nlevels=levels)
    # lowpass: average
    lp = 0.5 * (ta.lowpass + tb.lowpass)
    # highpasses (complex, 6 directions each level): max-abs magnitude
    hps = []
    for ha, hb in zip(ta.highpasses, tb.highpasses):
        sel = np.abs(ha) >= np.abs(hb)
        hps.append(np.where(sel, ha, hb))
    from dtcwt import Pyramid
    fp = Pyramid(lp, tuple(hps))
    F = t.inverse(fp)
    return F[:A.shape[0], :A.shape[1]]


# -------------------------------------------------------------------------- GTF
def _div(px, py):
    # negative divergence (adjoint of forward-diff gradient)
    dx = np.zeros_like(px); dy = np.zeros_like(py)
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, 0] = px[:, 0]
    dx[:, -1] = -px[:, -2]
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[0, :] = py[0, :]
    dy[-1, :] = -py[-2, :]
    return dx + dy


def _shrink(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def fuse_GTF(A, B, lam=4.0, niter=20, rho=2.0):
    """Gradient Transfer Fusion (Ma et al., Inf. Fusion 2016).

    Maps to the contract's IR/VIS roles: B -> IR (intensity to preserve),
    A -> VIS (gradient/texture to transfer).
      min_F ||F - B||_1 + lam * ||grad F - grad A||_1
    Solved with an ADMM split: u = F - B, (wx,wy) = grad F - grad A.
    F-update is a Poisson-like solve done in the DCT (Neumann) domain.
    """
    from scipy.fft import dctn, idctn
    IR = B.astype(np.float64)   # intensity target
    VIS = A.astype(np.float64)  # gradient source
    h, w = IR.shape

    # precompute gradient of VIS
    ax = np.zeros_like(VIS); ay = np.zeros_like(VIS)
    ax[:, :-1] = VIS[:, 1:] - VIS[:, :-1]
    ay[:-1, :] = VIS[1:, :] - VIS[:-1, :]

    def fgrad(img):
        gx = np.zeros_like(img); gy = np.zeros_like(img)
        gx[:, :-1] = img[:, 1:] - img[:, :-1]
        gy[:-1, :] = img[1:, :] - img[:-1, :]
        return gx, gy

    F = IR.copy()
    # ADMM duals
    u = np.zeros_like(IR); bu = np.zeros_like(IR)
    wx = np.zeros_like(IR); wy = np.zeros_like(IR)
    bx = np.zeros_like(IR); by = np.zeros_like(IR)

    # Laplacian eigenvalues for DCT-Neumann Poisson solve
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    denom = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
    # system: (rho*I - rho*Lap) F = rho*(B+u-bu) - rho*div(...)
    sys_denom = rho - rho * denom
    sys_denom[sys_denom == 0] = 1e-8

    for _ in range(niter):
        # F-update: minimize rho/2||F-B-u+bu||^2 + rho/2||gradF-gradA-w+b||^2
        gx, gy = fgrad(F)
        rhs_data = rho * (IR + u - bu)
        rhs_grad = -rho * _div(wx + ax - bx, wy + ay - by)
        rhs = rhs_data + rhs_grad
        Fhat = dctn(rhs, norm="ortho")
        F = idctn(Fhat / sys_denom, norm="ortho")
        # u-update (data L1)
        Fmb = F - IR
        u = _shrink(Fmb + bu, 1.0 / rho)
        bu = bu + (Fmb - u)
        # w-update (gradient L1)
        gx, gy = fgrad(F)
        rx = gx - ax; ry = gy - ay
        wx = _shrink(rx + bx, lam / rho)
        wy = _shrink(ry + by, lam / rho)
        bx = bx + (rx - wx)
        by = by + (ry - wy)
    return F


# ------------------------------------------------------------------------- NSCT
def _directional_decomp(detail, ndir=4):
    """Cheap directional split of a detail (high-freq) band via oriented
    difference filters. Approximation of a directional filter bank."""
    from scipy.ndimage import convolve
    dirs = []
    # 4 oriented gradient-like kernels (0, 45, 90, 135 deg)
    kers = [
        np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], float),
        np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], float),
        np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], float),
        np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], float),
    ][:ndir]
    for k in kers:
        dirs.append(convolve(detail, k, mode="reflect"))
    return dirs


def fuse_NSCT(A, B, levels=4, ndir=4):
    """NSCT* — Laplacian-pyramid multiscale + directional decomposition of the
    detail bands; fusion = max directional-energy selection. Approximation of a
    true non-subsampled contourlet transform (labelled NSCT* in the report)."""
    la = laplacian_pyramid(A, levels)
    lb = laplacian_pyramid(B, levels)
    fused = []
    for i in range(levels):
        da_dirs = _directional_decomp(la[i], ndir)
        db_dirs = _directional_decomp(lb[i], ndir)
        # directional energy = sum of |coeff| across orientations
        ea = sum(np.abs(d) for d in da_dirs)
        eb = sum(np.abs(d) for d in db_dirs)
        sel = ea >= eb
        fused.append(np.where(sel, la[i], lb[i]))
    fused.append(0.5 * (la[-1] + lb[-1]))
    return reconstruct_pyramid(fused)


# --------------------------------------------------------------------------- run
FUSERS = {
    "LP": fuse_LP,
    "DWT": fuse_DWT,
    "DTCWT": fuse_DTCWT,
    "GTF": fuse_GTF,
    "NSCT": fuse_NSCT,  # written out as NSCT* in dir naming
}
# output dir name (NSCT approximation labelled NSCT*)
OUTNAME = {"NSCT": "NSCT*"}


def run_method_task(method, task):
    fuser = FUSERS[method]
    outname = OUTNAME.get(method, method)
    a_dir = os.path.join(BENCH, "inputs", task, "A")
    b_dir = os.path.join(BENCH, "inputs", task, "B")
    out_dir = os.path.join(BENCH, "fused", outname, task)
    stems = sorted(s[:-4] for s in os.listdir(a_dir) if s.endswith(".png"))
    n = 0
    for stem in stems:
        A = load_gray(os.path.join(a_dir, stem + ".png"))
        B = load_gray(os.path.join(b_dir, stem + ".png"))
        B = resize_to(B, A)
        F = fuser(A, B)
        save_gray(F, os.path.join(out_dir, stem + ".png"))
        n += 1
    print(f"[{method}] {task}: {n} images -> {out_dir}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    help="LP|DWT|DTCWT|GTF|NSCT|ALL")
    ap.add_argument("--task", default="all", help="irvis|medical|gfp_pc|all")
    args = ap.parse_args()
    methods = list(FUSERS) if args.method == "ALL" else [args.method]
    tasks = TASKS if args.task == "all" else [args.task]
    for m in methods:
        for t in tasks:
            run_method_task(m, t)


if __name__ == "__main__":
    main()
