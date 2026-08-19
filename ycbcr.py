"""YCbCr helpers for the RGB-input fusion pipeline.

Per the network design: for color sources we convert RGB->YCbCr and fuse only the
Y (luminance) channel; the fused Y is then recombined with the source CbCr and
inverse-transformed back to RGB to form the FINAL fused image. Grayscale tasks
(e.g. IR-VIS) skip this entirely.

We use BT.601 (PIL 'YCbCr', == JPEG full-range): for a grayscale pixel R=G=B=g,
Y == g exactly, so the Y fed to the network matches the original Test.py
convert('L') behaviour, and metrics (computed on Y) are unchanged. CbCr fusion
follows the deviation-weighted rule in Utils/.../fusedY2RGB.m.
"""
import numpy as np
from PIL import Image


def load_ycbcr(path):
    """Return (Y, Cb, Cr) float arrays in [0,255]; grayscale -> Cb=Cr=128 (neutral)."""
    img = Image.open(path)
    if img.mode == "L":
        y = np.asarray(img, dtype=np.float64)
        neutral = np.full_like(y, 128.0)
        return y, neutral, neutral
    arr = np.asarray(img.convert("YCbCr"), dtype=np.float64)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def load_y(path):
    """Y (luminance) channel only — the network input / metric basis."""
    return load_ycbcr(path)[0]


def fuse_chroma(c1, c2):
    """Deviation-weighted CbCr fusion (fusedY2RGB.m rule).

    Pixels neutral in both -> 128; otherwise weight each source's chroma by how
    far it is from neutral. A grayscale source (c==128) contributes nothing, so
    for GFP-PC the result equals the color source's (GFP) chroma.
    """
    d1, d2 = np.abs(c1 - 128.0), np.abs(c2 - 128.0)
    denom = d1 + d2
    return np.where(denom == 0, 128.0, (c1 * d1 + c2 * d2) / np.where(denom == 0, 1.0, denom))


def ycbcr_to_rgb(y, cb, cr):
    """Inverse transform -> uint8 RGB array (H,W,3)."""
    arr = np.stack([y, cb, cr], axis=-1).clip(0, 255).astype("uint8")
    return np.asarray(Image.fromarray(arr, "YCbCr").convert("RGB"))
