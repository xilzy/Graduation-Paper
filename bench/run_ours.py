"""Run our trained model over the fusion_bench standardized inputs.

Writes fused luminance Y to fusion_bench/fused/<name>/<task>/<stem>.png for all
three tasks, so it plugs straight into the existing eval pipeline
(recombine_rescore -> eval_method -> consolidate), scored identically to the 18
comparison methods.

Task-id mapping = training config order (gfp_pc=0, irvis=1, medical=2).

Usage:
  venv/bin/python bench/run_ours.py --model h2h_moe_ta --name Ours-MoE
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
import torch

ROOT = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"
BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
sys.path.insert(0, ROOT)

TASK_ID = {"gfp_pc": 0, "irvis": 1, "medical": 2}


def read_args(name):
    d = {}
    p = f"{ROOT}/models/{name}/args.txt"
    if os.path.exists(p):
        for line in open(p):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def build(model_name, dev):
    a = read_args(model_name)
    is_moe = "n_routed" in a
    ck = sorted(glob.glob(f"{ROOT}/models/{model_name}/model_*.pth"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
    if is_moe:
        from Networks.net_moe import MODEL_MoE
        sd = torch.load(ck, map_location="cpu")
        pth = any(kk.startswith("heads.") for kk in sd)   # per-task heads?
        m = MODEL_MoE(in_channel=2, n_tasks=3, per_task_head=pth,
                      out_channel=int(a.get("out_channel", 16)),
                      depth=int(a.get("depth", 3)), num_heads=int(a.get("num_heads", 8)),
                      window_size=int(a.get("window_size", 1)),
                      n_routed=int(a.get("n_routed", 4)), n_shared=int(a.get("n_shared", 1)),
                      out_scale=a.get("out_scale", "False") == "True",
                      use_task_bias=a.get("no_task_bias", "False") != "True",
                      fusion_head=a.get("fusion_head", "direct"),
                      res_scale=float(a.get("res_scale", 0.2)),
                      routing=a.get("routing", "softmax"))
    else:
        from Networks.net import MODEL
        m = MODEL(in_channel=2)
    m.load_state_dict(sd if is_moe else torch.load(ck, map_location="cpu"))
    return m.eval().to(dev), is_moe, os.path.basename(ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="models/<model> dir name")
    ap.add_argument("--name", required=True, help="method name in fusion_bench")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    m, is_moe, ck = build(args.model, dev)
    print(f"model={args.model} ckpt={ck} arch={'moe' if is_moe else 'dense'} -> method '{args.name}'")

    for task, tid in TASK_ID.items():
        Adir = f"{BENCH}/inputs/{task}/A"
        outdir = f"{BENCH}/fused/{args.name}/{task}"
        os.makedirs(outdir, exist_ok=True)
        stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(Adir))]
        with torch.no_grad():
            for stem in stems:
                A = np.asarray(Image.open(f"{BENCH}/inputs/{task}/A/{stem}.png").convert("L"), np.float32)
                B = np.asarray(Image.open(f"{BENCH}/inputs/{task}/B/{stem}.png").convert("L"), np.float32)
                if B.shape != A.shape:
                    B = np.asarray(Image.fromarray(B.astype("uint8")).resize(
                        (A.shape[1], A.shape[0]), Image.BILINEAR), np.float32)
                inp = torch.from_numpy(np.stack([A, B])[None] / 255.).float().to(dev)
                out = m(inp, torch.tensor([tid], device=dev))[0] if is_moe else m(inp)
                F = out.squeeze().clamp(0, 1).cpu().numpy() * 255.0
                Image.fromarray(F.round().astype("uint8")).save(f"{outdir}/{stem}.png")
        print(f"  {task}: wrote {len(stems)} fused-Y -> {outdir}")


if __name__ == "__main__":
    main()
