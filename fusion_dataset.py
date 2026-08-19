"""Unified multi-task fusion dataset (Phase-0 foundation).

Replaces the original h5 + per-sample Python loop with an on-the-fly overlapping
patch dataset that supports real batched training and is ready for the
multi-task extension: each item carries (src_a, src_b, task_id, task_name).

Train split uses deterministic overlapping crops (paper regime: 170x170,
stride 28). Test/probe split returns full images (batch_size must be 1).

Accepts one or more task configs (same schema as fusion_data); items from all
tasks are concatenated, so a single DataLoader yields mixed-task batches.
"""
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import fusion_data


def _split_pairs(cfg, split, root):
    pairs = fusion_data.list_pairs(cfg, root)
    n_test = cfg.get("n_test", 30)
    # deterministic split: last n_test (by sorted stem) are test
    train_pairs, test_pairs = pairs[:-n_test], pairs[-n_test:]
    return {"train": train_pairs, "test": test_pairs, "all": pairs}[split]


def _load_gray01(path, ref_shape=None):
    img = Image.open(path).convert("L")
    if ref_shape is not None and img.size != (ref_shape[1], ref_shape[0]):
        img = img.resize((ref_shape[1], ref_shape[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class FusionDataset(Dataset):
    def __init__(self, configs, split="train", patch=170, stride=28, root="."):
        if isinstance(configs, (str, dict)):
            configs = [configs]
        self.split = split
        self.patch = patch
        self.root = root
        self.tasks = []
        self.index = []   # (task_id, pair_idx, top, left)  (top/left None for full image)

        for tid, cfg in enumerate(configs):
            if isinstance(cfg, str):
                cfg = fusion_data.load_config(cfg)
            self.tasks.append(cfg["task"])
            pairs = _split_pairs(cfg, split, root)
            setattr(self, f"_pairs_{tid}", pairs)
            for pi, (stem, a, b) in enumerate(pairs):
                if split == "train":
                    H, W = 358, 358  # canonical (mismatched pairs are resized to A)
                    tops = list(range(0, max(1, H - patch + 1), stride))
                    lefts = list(range(0, max(1, W - patch + 1), stride))
                    for t in tops:
                        for l in lefts:
                            self.index.append((tid, pi, t, l))
                else:
                    self.index.append((tid, pi, None, None))

    def _pairs(self, tid):
        return getattr(self, f"_pairs_{tid}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        tid, pi, top, left = self.index[i]
        stem, a_path, b_path = self._pairs(tid)[pi]
        A = _load_gray01(a_path)
        B = _load_gray01(b_path, ref_shape=A.shape)
        if top is not None:
            p = self.patch
            # guard against pairs smaller than canonical
            top = min(top, max(0, A.shape[0] - p))
            left = min(left, max(0, A.shape[1] - p))
            A = A[top:top + p, left:left + p]
            B = B[top:top + p, left:left + p]
        a = torch.from_numpy(A)[None]   # (1,H,W)
        b = torch.from_numpy(B)[None]
        return {"src_a": a, "src_b": b, "task_id": tid,
                "task": self.tasks[tid], "stem": stem}
