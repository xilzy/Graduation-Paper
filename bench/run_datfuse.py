import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, '/ytech_m2v4_hdd/lizhongyin/code/ref/DATFuse')
from Networks.network import MODEL as net

ROOT = '/ytech_m2v4_hdd/lizhongyin/fusion_bench'
MODEL_PATH = '/ytech_m2v4_hdd/lizhongyin/code/ref/DATFuse/model_10.pth'

device = torch.device('cuda:0')
model = net(in_channel=2).to(device)
state = torch.load(MODEL_PATH, map_location=device)
missing, unexpected = model.load_state_dict(state, strict=True)
model.eval()
tran = transforms.ToTensor()

def fuse(a_path, b_path):
    # contract: A = visible/functional, B = IR/structure
    # DATFuse input order = cat((ir, vi)) => [B, A]
    vi = Image.open(a_path).convert('L')   # A -> VIS slot
    ir = Image.open(b_path).convert('L')   # B -> IR slot
    vi_t = tran(vi)
    ir_t = tran(ir)
    inp = torch.cat((ir_t, vi_t), 0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    res = out.squeeze().detach().cpu().numpy()
    res = np.clip(res, 0.0, 1.0)
    res = (res * 255.0).round().astype(np.uint8)
    return res

def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ['irvis', 'medical', 'gfp_pc']
    for task in tasks:
        a_dir = os.path.join(ROOT, 'inputs', task, 'A')
        b_dir = os.path.join(ROOT, 'inputs', task, 'B')
        out_dir = os.path.join(ROOT, 'fused', 'DATFuse', task)
        os.makedirs(out_dir, exist_ok=True)
        stems = sorted(f[:-4] for f in os.listdir(a_dir) if f.endswith('.png'))
        t0 = time.time()
        for stem in stems:
            res = fuse(os.path.join(a_dir, stem + '.png'), os.path.join(b_dir, stem + '.png'))
            Image.fromarray(res, mode='L').save(os.path.join(out_dir, stem + '.png'))
        print(f'{task}: {len(stems)} imgs in {time.time()-t0:.1f}s -> {out_dir}')

if __name__ == '__main__':
    main()
