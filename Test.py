from PIL import Image
import numpy as np
import os
import torch

import time
import imageio

import torchvision.transforms as transforms

from Networks.net import MODEL as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=2)
# model_path = "models/model_10.pth"   #原作者提供的模型
# model_path = "models/10/model_10.pth"  #用自己训练的模型
model_path = "E:/论文写作/杨老师/论文投稿/第二篇/实验改进/models/10/model_10.pth"  #用自己训练的模型
#/**-L_structure系数测定-**/
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify1/model_10.pth"  #用自己训练的模型
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify2/model_10.pth"  #用自己训练的模型
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify3/model_10.pth"  #用自己训练的模型
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify4/model_10.pth"  #用自己训练的模型
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify5/model_10.pth"  #用自己训练的模型
#/**-消融实验-**/
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify6/model_10.pth"  #用自己训练的模型
# model_path = "F:/论文实验复现/杨老师/刘羽/MATR-main/models/modify7/model_10.pth"  #用自己训练的模型
use_gpu = torch.cuda.is_available()

# path1 = 'E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/GFP/'
# path2 = 'E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/PCI/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/方法对比/gfp-pci/proposed/第三篇/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/方法对比/gfp-pci/MATR/第三篇/'
# path1 = 'E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-pet/pet/'
# path2 = 'E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-pet/mri/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/方法对比/mri-pet/MDFNet/第三篇/'
path1 = 'E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-spect/spect/'
path2 = 'E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-spect/mri/'
save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/方法对比/mri-spect/MDFNet/第三篇/'
#/**-L_structure系数测定-**/
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify1/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify2/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify3/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify4/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify5/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify0/'
#/**-L_structure系数测定-**/
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify6/'
# save_path = 'F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify7/'

if use_gpu:

    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))

else:

    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)


def fusion():
    print('\nBegin to generate pictures ...\n')
    t = []
    for root, dirs, files in os.walk(path1):
        print('files:', files)

    for i in range(len(files)):
        tic = time.time()
        file_name1 = path1 + files[i]
        file_name2 = path2 + files[i]
        img1 = Image.open(file_name1).convert('L')
        img2 = Image.open(file_name2).convert('L')


        img1_org = img1
        img2_org = img2

        tran = transforms.ToTensor()

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        print(file_name1,img1_org.shape,file_name2,img2_org.shape)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img

        model.eval()
        out = model(input_img)

        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        # imageio.imwrite('./fusion result/{}.bmp'.format(num+1),
        #                 result)
        # imageio.imwrite('E:/论文写作/杨老师/论文投稿/第二篇/实验改进/test result/{}.bmp'.format(num+1),
        #                 result)
        imageio.imsave(save_path + file_name1.split('/')[-1], result)


        toc = time.time()
        print('end  {}{}'.format(i // 10, i % 10), ', time:{}'.format(toc - tic))



if __name__ == '__main__':

    fusion()
