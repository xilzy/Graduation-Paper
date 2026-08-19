"""Core-9 + diagnostic fusion metrics, ported from evaluation/*.m.

References (formulas verified against the MATLAB sources in evaluation/):
  EN        - MyEntroy.m / entropy
  SD        - SD.m / std2
  SF        - MySF.m / space_frequency.m
  AG        - AverageGradient.m / avg_gradient.m
  MI        - MI.m / mutual_info.m  (fusion MI = MI(A,F)+MI(B,F))
  SSIM      - ssim_index.m (Wang 2004)
  MS-SSIM   - msssim.m / MS_SSIM.m (Wang 2003)
  Qabf      - Qabf.m (Xydeas & Petrovic 2000)
  VIF       - vifvec.m / VIF (Sheikh & Bovik), pixel-domain (vifp)
  SCD       - analysis_SCD.m (Aslantas & Bendes 2015)
  Nabf      - analysis_nabf.m (fusion artifacts, lower better)
  CC        - my_cc.m (correlation coefficient)
  PSNR      - psnr.m

Inputs are 2-D float arrays in [0,255]. Helper _u8() clips/rounds to uint8 for
the histogram-based metrics (EN, MI), matching the MATLAB uint8 behaviour.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f(x):
    return np.asarray(x, dtype=np.float64)


def _u8(x):
    return np.clip(np.round(_f(x)), 0, 255).astype(np.uint8)


def _gaussian_window(size=11, sigma=1.5):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


# --------------------------------------------------------------------------- #
# no-reference metrics (function of F only)
# --------------------------------------------------------------------------- #
def en(F, **_):
    """Information entropy (higher better)."""
    f = _u8(F)
    hist = np.bincount(f.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sd(F, **_):
    """Standard deviation (higher better)."""
    return float(_f(F).std())


def sf(F, **_):
    """Spatial frequency (higher better)."""
    f = _f(F)
    rf = np.sqrt(np.mean((f[:, 1:] - f[:, :-1]) ** 2))
    cf = np.sqrt(np.mean((f[1:, :] - f[:-1, :]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def ag(F, **_):
    """Average gradient (higher better)."""
    f = _f(F)
    gx = f[:, 1:] - f[:, :-1]          # (H, W-1)
    gy = f[1:, :] - f[:-1, :]          # (H-1, W)
    gx = gx[:-1, :]                    # crop to common (H-1, W-1)
    gy = gy[:, :-1]
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))


# --------------------------------------------------------------------------- #
# mutual information
# --------------------------------------------------------------------------- #
def _mi_pair(X, Y):
    x, y = _u8(X).ravel(), _u8(Y).ravel()
    joint = np.histogram2d(x, y, bins=256, range=[[0, 256], [0, 256]])[0]
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    Hxy = -(joint[nz] * np.log2(joint[nz])).sum()
    px = px[px > 0]; py = py[py > 0]
    Hx = -(px * np.log2(px)).sum()
    Hy = -(py * np.log2(py)).sum()
    return float(Hx + Hy - Hxy)


def mi_split(A, B, F):
    """MI of fused with each source separately -> (MI_A, MI_B)."""
    return _mi_pair(A, F), _mi_pair(B, F)


def mi(A, B, F):
    """Fusion mutual information MI(A,F)+MI(B,F) (higher better)."""
    a, b = mi_split(A, B, F)
    return float(a + b)


# --------------------------------------------------------------------------- #
# SSIM / MS-SSIM
# --------------------------------------------------------------------------- #
def _ssim_map(x, y, win, data_range=255.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mode = "same"
    mu_x = convolve2d(x, win, mode=mode, boundary="symm")
    mu_y = convolve2d(y, win, mode=mode, boundary="symm")
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = convolve2d(x * x, win, mode=mode, boundary="symm") - mu_x2
    sy = convolve2d(y * y, win, mode=mode, boundary="symm") - mu_y2
    sxy = convolve2d(x * y, win, mode=mode, boundary="symm") - mu_xy
    cs = (2 * sxy + C2) / (sx + sy + C2)
    ssim = ((2 * mu_xy + C1) / (mu_x2 + mu_y2 + C1)) * cs
    return ssim.mean(), cs.mean()


def _ssim(x, y):
    win = _gaussian_window(11, 1.5)
    return _ssim_map(_f(x), _f(y), win)[0]


def ssim_split(A, B, F):
    return float(_ssim(F, A)), float(_ssim(F, B))


def ssim_pair(A, B, F):
    """Mean structural similarity of fused with the two sources (higher better)."""
    a, b = ssim_split(A, B, F)
    return float((a + b) / 2)


def _ms_ssim(x, y, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    x, y = _f(x), _f(y)
    win = _gaussian_window(11, 1.5)
    levels = len(weights)
    mssim, mcs = [], []
    for i in range(levels):
        s, cs = _ssim_map(x, y, win)
        mssim.append(s); mcs.append(cs)
        if i < levels - 1:
            x = gaussian_filter(x, 1.0)[::2, ::2]
            y = gaussian_filter(y, 1.0)[::2, ::2]
    mssim = np.clip(np.array(mssim), 1e-8, None)
    mcs = np.clip(np.array(mcs), 1e-8, None)
    w = np.array(weights)
    return float(np.prod(mcs[:-1] ** w[:-1]) * (mssim[-1] ** w[-1]))


def ms_ssim_pair(A, B, F):
    """Mean multi-scale SSIM of fused with the two sources (higher better)."""
    return float((_ms_ssim(F, A) + _ms_ssim(F, B)) / 2)


# --------------------------------------------------------------------------- #
# Qabf & Nabf (Xydeas-Petrovic gradient framework)
# --------------------------------------------------------------------------- #
def _sobel_grad(x):
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve2d(x, sx, mode="same", boundary="symm")
    gy = convolve2d(x, sy, mode="same", boundary="symm")
    g = np.sqrt(gx ** 2 + gy ** 2)
    a = np.arctan2(gy, gx)            # angle in (-pi, pi]
    return g, a


def _qabf_components(A, B, F):
    L = 1.0
    Tg, kg, Dg = 0.9994, -15.0, 0.5
    Ta, ka, Da = 0.9879, -22.0, 0.8
    gA, aA = _sobel_grad(_f(A))
    gB, aB = _sobel_grad(_f(B))
    gF, aF = _sobel_grad(_f(F))
    eps = 1e-10

    def qabf_one(g, a, gf, af):
        G = np.where(g > gf, gf / (g + eps), np.where(gf > g, g / (gf + eps), 1.0))
        Aa = 1.0 - np.abs(a - af) / (np.pi / 2)
        Qg = Tg / (1 + np.exp(kg * (G - Dg)))
        Qa = Ta / (1 + np.exp(ka * (Aa - Da)))
        return Qg * Qa

    QAF = qabf_one(gA, aA, gF, aF)
    QBF = qabf_one(gB, aB, gF, aF)
    wA = gA ** L
    wB = gB ** L
    return QAF, QBF, wA, wB, gA, gB, gF


def qabf(A, B, F):
    """Gradient-based fusion quality Qabf (higher better, in [0,1])."""
    QAF, QBF, wA, wB, *_ = _qabf_components(A, B, F)
    num = (QAF * wA + QBF * wB).sum()
    den = (wA + wB).sum() + 1e-10
    return float(num / den)


def nabf(A, B, F):
    """Fusion artifacts / noise added by fusion (lower better).

    Artifact locations are where the fused gradient exceeds both source
    gradients; their un-transferred information is accumulated.
    """
    QAF, QBF, wA, wB, gA, gB, gF = _qabf_components(A, B, F)
    AM = (gF > gA) & (gF > gB)
    num = (AM * ((1 - QAF) * wA + (1 - QBF) * wB)).sum()
    den = (wA + wB).sum() + 1e-10
    return float(num / den)


# --------------------------------------------------------------------------- #
# VIF (pixel-domain, Sheikh & Bovik) per-source averaged
# --------------------------------------------------------------------------- #
def _vifp(ref, dist):
    ref, dist = _f(ref), _f(dist)
    eps = 1e-10
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sigma = N / 5.0
        if scale > 1:
            ref = gaussian_filter(ref, sigma)[::2, ::2]
            dist = gaussian_filter(dist, sigma)[::2, ::2]
        mu1 = gaussian_filter(ref, sigma)
        mu2 = gaussian_filter(dist, sigma)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = gaussian_filter(ref * ref, sigma) - mu1_sq
        s2 = gaussian_filter(dist * dist, sigma) - mu2_sq
        s12 = gaussian_filter(ref * dist, sigma) - mu1_mu2
        s1 = np.maximum(s1, 0); s2 = np.maximum(s2, 0)
        g = s12 / (s1 + eps)
        sv_sq = s2 - g * s12
        g = np.where(s1 < eps, 0.0, g)
        sv_sq = np.where(s1 < eps, s2, sv_sq)
        s1 = np.where(s1 < eps, 0.0, s1)
        g = np.where(s2 < eps, 0.0, g)
        sv_sq = np.where(s2 < eps, 0.0, sv_sq)
        sv_sq = np.maximum(sv_sq, eps)
        num += np.sum(np.log10(1 + g ** 2 * s1 / (sv_sq + eps)))
        den += np.sum(np.log10(1 + s1 / eps))
    return float(num / (den + eps))


def vif_pair(A, B, F):
    """Visual information fidelity (pixel-domain), averaged over sources (higher better)."""
    return float((_vifp(A, F) + _vifp(B, F)) / 2)


# --------------------------------------------------------------------------- #
# SCD, CC, PSNR
# --------------------------------------------------------------------------- #
def _corr(x, y):
    x, y = _f(x).ravel(), _f(y).ravel()
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-10
    return float((x * y).sum() / d)


def scd(A, B, F):
    """Sum of correlations of differences (higher better)."""
    return float(_corr(F - B, A) + _corr(F - A, B))


def cc(A, B, F):
    """Correlation coefficient of fused with sources, averaged (higher better)."""
    return float((_corr(A, F) + _corr(B, F)) / 2)


def psnr_pair(A, B, F):
    """PSNR of fused vs the average of sources (higher better)."""
    a, b, f = _f(A), _f(B), _f(F)
    mse = np.mean((f - (a + b) / 2) ** 2) + 1e-10
    return float(10 * np.log10(255.0 ** 2 / mse))


# --------------------------------------------------------------------------- #
# balance-aware (per-source harmonic-mean) variants
#   A "copy one source" fuse scores high on summed/averaged fidelity but
#   collapses under the harmonic mean, which is dominated by the weaker source.
#   Rationale + evidence: knowledge/paper/EXP-0-02.
# --------------------------------------------------------------------------- #
def _hm(a, b):
    """Harmonic mean (soft-min); 0 if either side is non-positive."""
    if a <= 0 or b <= 0:
        return 0.0
    return float(2 * a * b / (a + b))


def qabf_split(A, B, F):
    QAF, QBF, wA, wB, *_ = _qabf_components(A, B, F)
    qa = float((QAF * wA).sum() / (wA.sum() + 1e-10))
    qb = float((QBF * wB).sum() / (wB.sum() + 1e-10))
    return qa, qb


def vif_split(A, B, F):
    return _vifp(A, F), _vifp(B, F)


def ms_ssim_split(A, B, F):
    return _ms_ssim(F, A), _ms_ssim(F, B)


def ssim_hm(A, B, F):
    a, b = ssim_split(A, B, F); return _hm(a, b)


def mi_hm(A, B, F):
    a, b = mi_split(A, B, F); return _hm(a, b)


def qabf_hm(A, B, F):
    a, b = qabf_split(A, B, F); return _hm(a, b)


def vif_hm(A, B, F):
    a, b = vif_split(A, B, F); return _hm(a, b)


def ms_ssim_hm(A, B, F):
    a, b = ms_ssim_split(A, B, F); return _hm(a, b)


def balance(A, B, F):
    """Modality balance in [0,1]: min(SSIM_A,SSIM_B)/max(...). 1 = perfectly balanced."""
    a, b = ssim_split(A, B, F)
    hi = max(a, b)
    return float(min(a, b) / hi) if hi > 0 else 0.0


# --------------------------------------------------------------------------- #
# FUNCTION axis: did the fusion preserve the functional source's SALIENT info?
#   Generic metrics can't tell whether GFP's (sparse, near-black) functional
#   signal was injected, because the structural source dominates. We restrict
#   the measurement to the functional source's salient region (its brightest
#   pixels: GFP fluorescence spots / IR thermal targets) and ask:
#     FuncCorr - does the fused luminance track the functional pattern there?
#     FuncSal  - do those regions stand out (are bright) in the fused image?
#   A copy-of-structure fuse (e.g. Max=PC) ignores GFP -> low on both.
# --------------------------------------------------------------------------- #
def _salient_mask(func_y, percentile=90.0):
    thr = np.percentile(func_y, percentile)
    m = func_y >= thr
    if m.sum() < 16:                      # degenerate (flat source) -> top pixels
        m = func_y >= np.sort(func_y.ravel())[-16]
    return m


def func_corr(func_y, fused_y, percentile=90.0):
    """Pearson corr between functional source and fused, within the salient mask."""
    m = _salient_mask(_f(func_y), percentile)
    x = _f(func_y)[m]; y = _f(fused_y)[m]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + 1e-10
    return float((x * y).sum() / d)


def func_sal(func_y, fused_y, percentile=90.0):
    """Saliency of the functional region in the fused image:
    (mean inside salient mask - mean outside) / global std. Higher = stands out."""
    fy = _f(func_y); fz = _f(fused_y)
    m = _salient_mask(fy, percentile)
    s = fz.std() + 1e-10
    return float((fz[m].mean() - fz[~m].mean()) / s)


# Function axis (task-specific, computed in eval with the configured func source).
FUNCTION_AXIS = ["FuncCorr", "FuncSal"]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
CORE_METRICS = {
    "EN": en, "MI": mi, "SD": sd, "SF": sf, "AG": ag,
    "SSIM": ssim_pair, "MS_SSIM": ms_ssim_pair, "Qabf": qabf, "VIF": vif_pair,
}
DIAGNOSTIC_METRICS = {
    "Nabf": nabf, "SCD": scd, "CC": cc, "PSNR": psnr_pair, "Balance": balance,
}
# Balance-aware ranking set for the S1 average-rank. Two axes that a real fusion
# must win simultaneously:
#   (i)  fidelity to BOTH sources -> harmonic-mean (soft-min) variants; kills the
#        "copy one source" cheat (e.g. naive Max).
#   (ii) no-reference image quality -> EN/SD/SF/AG; kills the "blur everything"
#        cheat (e.g. naive Avg).
RANK_METRICS = {
    "SSIM_hm": ssim_hm, "MS_SSIM_hm": ms_ssim_hm, "Qabf_hm": qabf_hm,
    "VIF_hm": vif_hm, "MI_hm": mi_hm,
    "EN": en, "SD": sd, "SF": sf, "AG": ag,
}
# Two evaluation axes (scored separately so metric-count imbalance can't bias a
# single scalar; a method should Pareto-improve, not trade one axis for the other).
#   FIDELITY_AXIS  - balance-aware fidelity to BOTH sources (kills copy/blur cheats)
#   QUALITY_AXIS   - no-reference image richness (sharpness/contrast/information)
# Caveat (GFP-PC): when one source is near-black (GFP), the fidelity axis is
# dominated by the other source, so trivial copy/blend tops it; the quality axis
# is the discriminator there. See knowledge/paper/EXP-0-02.
FIDELITY_AXIS = ["SSIM_hm", "MS_SSIM_hm", "Qabf_hm", "VIF_hm", "MI_hm"]
QUALITY_AXIS = ["EN", "SD", "SF", "AG"]
# True = higher is better, False = lower is better
HIGHER_IS_BETTER = {
    "EN": True, "MI": True, "SD": True, "SF": True, "AG": True,
    "SSIM": True, "MS_SSIM": True, "Qabf": True, "VIF": True,
    "Nabf": False, "SCD": True, "CC": True, "PSNR": True, "Balance": True,
    "SSIM_hm": True, "MS_SSIM_hm": True, "Qabf_hm": True, "VIF_hm": True,
    "MI_hm": True,
    "FuncCorr": True, "FuncSal": True,
}


def compute_all(A, B, F, include_diagnostic=True):
    """Return {metric: value} for one (A,B,F) triple (core + balance-aware + diag)."""
    out = {}
    pool = dict(CORE_METRICS)
    pool.update(RANK_METRICS)                         # adds the *_hm balance-aware metrics
    for name, fn in pool.items():
        try:
            out[name] = fn(A, B, F) if fn.__code__.co_argcount >= 3 else fn(F)
        except Exception:                            # never let one metric kill a run
            out[name] = float("nan")
    if include_diagnostic:
        for name, fn in DIAGNOSTIC_METRICS.items():
            try:
                out[name] = fn(A, B, F)
            except Exception:
                out[name] = float("nan")
        # per-source split diagnostics
        try:
            ma, mb = mi_split(A, B, F)
            out["MI_A"], out["MI_B"] = ma, mb
            sa, sb = ssim_split(A, B, F)
            out["SSIM_A"], out["SSIM_B"] = sa, sb
        except Exception:
            pass
    return out
