"""Dataset pairing + frozen probe-set selection, shared by infer/eval/train.

A task config is a dict:
  {
    "task": "gfp_pc",
    "src_a_dir": "source images/GFP-PC/GFP",   # modality A (e.g. GFP)
    "src_b_dir": "source images/GFP-PC/PCI",   # modality B (e.g. PC)
    "a_suffix": "-g.jpg",
    "b_suffix": "-t.jpg",
    "probe_size": 15
  }
Pairing is by stem = filename with the suffix removed; A and B must share stems.
The probe subset is chosen deterministically (evenly spaced over sorted stems)
so S1 numbers are comparable across iterations.
"""
import os
import json


def load_config(path):
    with open(path) as f:
        return json.load(f)


def list_pairs(cfg, root="."):
    a_dir = os.path.join(root, cfg["src_a_dir"])
    b_dir = os.path.join(root, cfg["src_b_dir"])
    asuf, bsuf = cfg["a_suffix"], cfg["b_suffix"]
    a_stems = {f[: -len(asuf)] for f in os.listdir(a_dir) if f.endswith(asuf)}
    b_stems = {f[: -len(bsuf)] for f in os.listdir(b_dir) if f.endswith(bsuf)}
    stems = sorted(a_stems & b_stems)
    pairs = [
        (s, os.path.join(a_dir, s + asuf), os.path.join(b_dir, s + bsuf))
        for s in stems
    ]
    return pairs


def probe_stems(cfg, root="."):
    """Deterministic evenly-spaced probe subset of the full pair list."""
    pairs = list_pairs(cfg, root)
    n = len(pairs)
    k = min(cfg.get("probe_size", 15), n)
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    idx = sorted(set(idx))
    return [pairs[i][0] for i in idx]


def select(cfg, which="all", root="."):
    pairs = list_pairs(cfg, root)
    if which == "all":
        return pairs
    if which == "probe":
        keep = set(probe_stems(cfg, root))
        return [p for p in pairs if p[0] in keep]
    raise ValueError(which)
