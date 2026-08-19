"""Multi-modal fusion pairing (superset of fusion_data).

Supports two pairing styles so GFP-PC, IR-VIS (MSRS) and medical (Harvard)
share one code path:

  pairing="suffix"  (GFP-PC): one dir per source, stems matched by removing a
                    per-source suffix; train/test split = last `n_test` stems.
  pairing="folder"  (MSRS/medical): explicit train_/test_ dirs per source, the
                    two sources share the SAME filename (stem = filename minus
                    a common extension).

A task config (json) — superset, backward compatible with fusion_data:
  {
    "task": "irvis",
    "pairing": "folder",
    "train_a_dir": "...","train_b_dir": "...",
    "test_a_dir":  "...","test_b_dir":  "...",
    "a_suffix": ".png", "b_suffix": ".png",
    "color_a": true, "color_b": false,   # which source carries chroma
    "output_mode": "gray", "func_source": "b",
    "probe_size": 15
  }
"""
import os
import json


def load_config(path):
    with open(path) as f:
        return json.load(f)


def _stems_in(d, suf):
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(suf):
            out[f[: -len(suf)]] = os.path.join(d, f)
    return out


def _pair_dirs(a_dir, b_dir, asuf, bsuf):
    """Pairs from two dirs by shared stem (suffix removed)."""
    a = _stems_in(a_dir, asuf)
    b = _stems_in(b_dir, bsuf)
    stems = sorted(set(a) & set(b))
    return [(s, a[s], b[s]) for s in stems]


def list_pairs(cfg, split, root="."):
    """Return list of (stem, a_path, b_path) for the requested split.

    split in {"train","test","all"}.
    """
    pairing = cfg.get("pairing", "suffix")
    asuf, bsuf = cfg.get("a_suffix", ""), cfg.get("b_suffix", "")

    if pairing == "folder":
        def J(k):
            return os.path.join(root, cfg[k]) if cfg.get(k) else None
        tr = _pair_dirs(J("train_a_dir"), J("train_b_dir"), asuf, bsuf)
        te = _pair_dirs(J("test_a_dir"), J("test_b_dir"), asuf, bsuf)
        return {"train": tr, "test": te, "all": tr + te}[split]

    # suffix style (single dir per source) + last-n_test split
    a_dir = os.path.join(root, cfg["src_a_dir"])
    b_dir = os.path.join(root, cfg["src_b_dir"])
    pairs = _pair_dirs(a_dir, b_dir, asuf, bsuf)
    n_test = cfg.get("n_test", 30)
    tr, te = pairs[:-n_test], pairs[-n_test:]
    return {"train": tr, "test": te, "all": pairs}[split]


def probe_stems(cfg, root="."):
    """Deterministic evenly-spaced probe subset of the TEST split."""
    pairs = list_pairs(cfg, "test", root)
    if not pairs:
        pairs = list_pairs(cfg, "all", root)
    n = len(pairs)
    k = min(cfg.get("probe_size", 15), n)
    if n == 0:
        return []
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [pairs[i][0] for i in sorted(set(idx))]


def select(cfg, which="all", split="all", root="."):
    pairs = list_pairs(cfg, split, root)
    if which == "all":
        return pairs
    if which == "probe":
        keep = set(probe_stems(cfg, root))
        return [p for p in pairs if p[0] in keep]
    raise ValueError(which)
