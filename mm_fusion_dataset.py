"""Unified multi-modal fusion dataset (GFP-PC + IR-VIS + medical).

One DataLoader yields mixed-task batches. Every task is reduced to the SAME
network contract used by the released MDFNet: two single-channel luminance (Y)
maps in [0,1], concatenated to a (2,H,W) tensor. This works for all tasks:

  * gray source (IR, PC, MRI)        -> Y == the gray value.
  * color source (VIS, GFP, PET/SPECT) -> Y of BT.601 YCbCr (chroma is kept on
    disk and only recombined at INFERENCE time, see ycbcr.py / infer_fusion).

Design choices that keep this coupling correct and robust:
  * Uniform 170x170 crops for ALL tasks (the grad loss hardcodes /170^2), so
    mixed-resolution sources (358, 640x480, 256) are reconciled by cropping, not
    resizing — no scale distortion of the luminance statistics.
  * B is aligned to A's size (bilinear) before cropping when a pair is not
    pre-registered to identical shape (medical/GFP are; MSRS is).
  * Images smaller than the patch are reflect-padded up to 170.
  * Per-task balancing: each task contributes a comparable number of crops so a
    large set (MSRS, 1083) does not drown small ones (GFP-PC, 118). Controlled by
    `crops_per_task`; crops are drawn with a fixed seed for reproducibility.
  * Test/probe split returns FULL images (batch_size must be 1).
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import mm_fusion_data as mfd

PATCH = 170


def load_y01(path, ref_shape=None):
    """Luminance (Y) in [0,1]. Color -> BT.601 Y; gray -> itself."""
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA", "P", "CMYK"):
        img = img.convert("YCbCr")
        y = np.asarray(img, dtype=np.float32)[:, :, 0]
    else:
        y = np.asarray(img.convert("L"), dtype=np.float32)
    y = y / 255.0
    if ref_shape is not None and y.shape != ref_shape:
        y = np.asarray(
            Image.fromarray((y * 255).astype("uint8")).resize(
                (ref_shape[1], ref_shape[0]), Image.BILINEAR),
            dtype=np.float32) / 255.0
    return y


def _pad_to(y, p):
    H, W = y.shape
    if H >= p and W >= p:
        return y
    ph, pw = max(0, p - H), max(0, p - W)
    return np.pad(y, ((0, ph), (0, pw)), mode="reflect")


class MMFusionDataset(Dataset):
    def __init__(self, configs, split="train", patch=PATCH,
                 crops_per_task=4000, root=".", seed=0, random_crop=True):
        if isinstance(configs, (str, dict)):
            configs = [configs]
        self.split = split
        self.patch = patch
        self.root = root
        self.random_crop = random_crop
        self.tasks = []
        self.cfgs = []
        self.index = []          # train: (tid, pair_idx[, top, left]); test: (tid, pi, None, None)
        self._pairs = {}
        rng = random.Random(seed)

        for tid, cfg in enumerate(configs):
            if isinstance(cfg, str):
                cfg = mfd.load_config(cfg)
            self.cfgs.append(cfg)
            self.tasks.append(cfg["task"])
            pairs = mfd.list_pairs(cfg, split, root)
            self._pairs[tid] = pairs
            if not pairs:
                continue

            if split != "train":
                for pi in range(len(pairs)):
                    self.index.append((tid, pi, None, None))
                continue

            # ---- train: balance each task to ~crops_per_task samples ----
            # crops_per_task may be an int (same for all) OR a dict
            # {task_name: count} for loss-aware curriculum (per-task quota).
            cpt = crops_per_task[cfg["task"]] if isinstance(crops_per_task, dict) else crops_per_task
            per_pair = max(1, cpt // len(pairs))
            if random_crop:
                # on-the-fly: index only picks WHICH pair; the crop POSITION is
                # drawn fresh every __getitem__ (diversity uncapped, so
                # crops_per_task is just epoch length / per-task balance).
                items = [(tid, pi) for pi in range(len(pairs))
                         for _ in range(per_pair)]
                rng.shuffle(items)
                self.index.extend(items[:cpt])
            else:
                # fixed pool: crop positions baked once (reproducible, but caps
                # diversity at crops_per_task distinct crops).
                pool = []
                for pi, (stem, a, b) in enumerate(pairs):
                    with Image.open(a) as im:
                        W, H = im.size
                    H, W = max(H, patch), max(W, patch)
                    for _ in range(per_pair):
                        pool.append((tid, pi, rng.randint(0, H - patch),
                                     rng.randint(0, W - patch)))
                rng.shuffle(pool)
                self.index.extend(pool[:cpt])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        rec = self.index[i]
        tid, pi = rec[0], rec[1]
        stem, a_path, b_path = self._pairs[tid][pi]
        A = load_y01(a_path)
        B = load_y01(b_path, ref_shape=A.shape)
        p = self.patch
        if self.split == "train":
            A, B = _pad_to(A, p), _pad_to(B, p)
            H, W = A.shape
            if self.random_crop:
                # torch RNG is seeded per-worker per-epoch by the DataLoader, so
                # this gives fresh, non-duplicated crops (np.random would alias
                # across workers).
                top = int(torch.randint(0, H - p + 1, (1,)))
                left = int(torch.randint(0, W - p + 1, (1,)))
            else:
                top = min(rec[2], H - p)
                left = min(rec[3], W - p)
            A = A[top:top + p, left:left + p]
            B = B[top:top + p, left:left + p]
        a = torch.from_numpy(np.ascontiguousarray(A))[None]
        b = torch.from_numpy(np.ascontiguousarray(B))[None]
        return {"src_a": a, "src_b": b, "task_id": tid,
                "task": self.tasks[tid], "stem": stem}
