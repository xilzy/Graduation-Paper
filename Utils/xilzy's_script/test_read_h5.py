# 验证制作的h5,读出来的图片是否成对且正确
import h5py
import numpy as np
from tqdm import tqdm
import imageio
imsave=imageio.imsave

path = "./GFP-PCI_120_modified2.h5"
output_path1="./source patches/GFP/"
output_path2="./source patches/PCI/"

source_data = h5py.File(path, 'r')
data = source_data['data'][:]
data = np.transpose(data, (0, 3, 2, 1))
print("data max: %s, min: %s" % (np.max(data), np.min(data)))
print("data shape:", data.shape)
num_imgs = data.shape[0]
mod = num_imgs % 64
n_batches = int(num_imgs // 64)
print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
if mod > 0:
    print('Train set has been trimmed %d samples...\n' % mod)
    source_imgs = data[:-mod]
else:
    source_imgs = data

for batch in tqdm(range(1)):
    #制作h5数据集的时候是先存的gfp，再存的pci
    gfp_batch = data[batch * 64:(batch * 64 + 64), :, :, 0]
    pci_batch = data[batch * 64:(batch * 64 + 64), :, :, 1]
    # pci_batch = np.expand_dims(data[batch * args.batch_size:(batch * args.batch_size + args.batch_size), :, :, 1], axis=-1)
    #依次取出每一对图像，归一化之后转化成tensor格式
    for index in range(10):
        gfp=gfp_batch[index,:,:]
        pci=pci_batch[index,:,:]
        imsave(output_path1 + str(index)+'.png', gfp)  # 结果名和第一张（第二张）的图片名一致
        imsave(output_path2 + str(index)+'.png', pci)  # 结果名和第一张（第二张）的图片名一致