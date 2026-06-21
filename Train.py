import os
import argparse

from tqdm import tqdm
import pandas as pd
import joblib
import glob

from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.net import MODEL as net

from losses import ssim_ir, ssim_vi,RMI_ir,RMI_vi,intensity,joint_grad
import h5py
import numpy as np

device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

#保存日志相关函数
def append_to_dataframe(df, new_data):
    """
    将新数据添加到 DataFrame 中.

    参数:
    - df: 原始 DataFrame.
    - new_data: 要添加的新数据，可以是 DataFrame 或 Series.

    返回值:
    - 更新后的 DataFrame.
    """
    if isinstance(new_data, pd.Series):
        new_data = pd.DataFrame(new_data).transpose()
    return pd.concat([df, new_data], axis=0).reset_index(drop=True)


#预定义参数
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=10, type=int)  #10个epoch
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 5,1,2.5,1,1,1], type=float)  #损失函数的权重系数
    # parser.add_argument('--weight', default=[1, 5, 1, 2.5], type=float)  # 损失函数的权重系数,重新调整了pci的SSIM权重系数
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)') #网络层次每多一层，通道数增加alpha个
    args = parser.parse_args()

    return args





class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):

        # ir = '...'
        # vi = '...'
        ir,vi=self.imageFolderDataset[index]
        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')

        if self.transform is not None:
            tran = transforms.ToTensor() #先归一化，然后转化成tensor
            ir=tran(ir)

            vi= tran(vi)

            input = torch.cat((ir, vi), -3)


            return input, ir,vi

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ** 完成一个epoch的 Training **
def train(writer,args, source_imgs,data,n_batches, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi,criterion_Intensity,criterion_Grad,optimizer, epoch, scheduler=None):
# def train(args, source_imgs,data,n_batches, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi,optimizer, epoch, scheduler=None):
    global step

    #这里都要用均值对象来实例化，是因为每次只读一对，进行处理，在外部进行求均值
    losses = AverageMeter()
    losses_structure = AverageMeter()
    losses_content = AverageMeter()

    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_RMI_ir = AverageMeter()
    losses_RMI_vi= AverageMeter()
    losses_Intensity_gfp = AverageMeter()
    losses_Intensity_pci= AverageMeter()
    losses_Grad= AverageMeter()
    weight = args.weight
    model.train()

    np.random.shuffle(source_imgs)  # 每个epoch都打乱一次数据集顺序
    for batch in tqdm(range(n_batches)):
        #制作h5数据集的时候是先存的gfp，再存的pci
        gfp_batch = data[batch * args.batch_size:(batch * args.batch_size + args.batch_size), :, :, 0]
        pci_batch = data[batch * args.batch_size:(batch * args.batch_size + args.batch_size), :, :, 1]
        # pci_batch = np.expand_dims(data[batch * args.batch_size:(batch * args.batch_size + args.batch_size), :, :, 1], axis=-1)
        #依次取出每一对图像，归一化之后转化成tensor格式
        for index in range(args.batch_size):
            step+=1      #步数+1
            gfp=gfp_batch[index,:,:]
            pci=pci_batch[index,:,:]
            # print("不除255：")
            # print(gfp)
            # print("______________________________________")
            # print("除以255：")
            # print(gfp/255.0)
            # print(gfp.shape,pci.shape)
            #把小块扩充成四维tensor
            h1, w1 = gfp.shape
            h2, w2 = pci.shape
            # h2, w2,_ = pci.shape
            gfp = gfp.reshape([ 1,1,h1, w1])
            pci = pci.reshape([ 1,1,h2, w2])
            # print(gfp.shape,pci.shape)
            gfp = torch.from_numpy(gfp)
            pci = torch.from_numpy(pci)
            # tran = transforms.ToTensor()
            # gfp=tran(gfp)
            # pci = tran(pci)
            Input = torch.cat((gfp, pci), -3) 
            # print(Input.shape)

            if use_gpu:
                Input = Input.cuda()
                gfp=gfp.cuda()
                pci=pci.cuda()


            else:
                Input = Input
                gfp=gfp
                pci=pci
            out = model(Input)

#             # 将图像张量转换为 NCHW 格式，并使用 make_grid() 函数将多张图像合并成一张网格图像
#             gfp_grid = make_grid(gfp, nrow=3, normalize=True, scale_each=True)
#             pci_grid = make_grid(pci, nrow=3, normalize=True, scale_each=True)
#             fused_grid = make_grid(out, nrow=3, normalize=True, scale_each=True)

            # # 使用 add_image() 方法将图像写入到日志文件中
            # writer.add_image('gfp', gfp_grid, global_step=step)
            # writer.add_image('pci', pci_grid, global_step=step)
            # writer.add_image('fused', fused_grid, global_step=step)

            #加权计算损失函数值
            #/*-- L_structure --*/
            # loss_SSIM
            loss_ssim_ir= weight[0] * criterion_ssim_ir(out, gfp)
            loss_ssim_vi= weight[1] * criterion_ssim_vi(out, pci)
            loss_ssim=loss_ssim_ir+loss_ssim_vi
            # loss_Grad
            loss_grad=weight[6]*criterion_Grad(gfp,pci,out)
            L_structure=loss_ssim+loss_grad

            # /*-- L_content --*/
            #loss_RMI
            loss_RMI_ir= weight[2] * criterion_RMI_ir(out,gfp)
            loss_RMI_vi = weight[3] * criterion_RMI_vi(out,pci)
            loss_RMI=loss_RMI_ir+loss_RMI_vi
            # loss_Intensity
            loss_Intensity_gfp=weight[4] * criterion_Intensity(gfp,out)
            loss_Intensity_pci = weight[5] * criterion_Intensity(pci, out)
            loss_Intensity=0.5*loss_Intensity_gfp+0.5*loss_Intensity_pci
            L_content=loss_RMI+loss_Intensity

            # /*-- L_total --*/
            loss = L_content+L_structure
            # loss = L_content+1.25*L_structure
            # loss = L_content+1.5*L_structure
            # loss = L_content+1.75*L_structure
            # loss = L_content+2.0*L_structure

            #使用 add_scalar() 方法将损失函数写入到日志文件中
            writer.add_scalar('loss_ssim_gfp', loss_ssim_ir.item(), global_step=step)
            writer.add_scalar('loss_ssim_pci', loss_ssim_vi.item(), global_step=step)
            writer.add_scalar('loss_ssim', loss_ssim.item(), global_step=step)
            writer.add_scalar('loss_RMI_gfp', loss_RMI_ir.item(), global_step=step)
            writer.add_scalar('loss_RMI_pci', loss_RMI_vi.item(), global_step=step)
            writer.add_scalar('loss_RMI', loss_RMI.item(), global_step=step)
            writer.add_scalar('loss_intensity_gfp', loss_Intensity_gfp.item(), global_step=step)
            writer.add_scalar('loss_intensity_pci', loss_Intensity_pci.item(), global_step=step)
            writer.add_scalar('loss_intensity', loss_Intensity.item(), global_step=step)
            writer.add_scalar('loss_grad', loss_grad.item(), global_step=step)
            writer.add_scalar('loss', loss.item(), global_step=step)

            losses.update(loss.item(), Input.size(0))
            losses_content.update(L_content.item(), Input.size(0))
            losses_structure.update(L_structure.item(), Input.size(0))
            losses_ssim_ir.update(loss_ssim_ir.item(), Input.size(0))
            losses_ssim_vi.update(loss_ssim_vi.item(), Input.size(0))
            losses_RMI_ir.update(loss_RMI_ir.item(), Input.size(0))
            losses_RMI_vi.update(loss_RMI_vi.item(), Input.size(0))
            losses_Intensity_gfp.update(loss_Intensity_gfp.item(), Input.size(0))
            losses_Intensity_pci.update(loss_Intensity_pci.item(), Input.size(0))
            losses_Grad.update(loss_grad.item(), Input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_content', losses_content.avg),
        ('loss_structure', losses_structure.avg),
        ('loss_ssim_gfp', losses_ssim_ir.avg),
        ('loss_ssim_pci', losses_ssim_vi.avg),
        ('loss_RMI_gfp', losses_RMI_ir.avg),
        ('loss_RMI_pci', losses_RMI_vi.avg),
        ('loss_intensity_gfp', losses_Intensity_gfp.avg),
        ('loss_intensity_pci', losses_Intensity_pci.avg),
        ('loss_grad', losses_Grad.avg),
    ])



    return log


step = 0  # 用来记录全局步长
def main():
    args = parse_args()
   #如果不存在日志和模型文件夹则创建
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)
    if not os.path.exists('logs/%s' %args.name):
        os.makedirs('logs/%s' %args.name)
    if not os.path.exists('./tensorboard_logs' ):
        os.makedirs('./tensorboard_logs')

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # path="./GFP-PCI_120.h5"
    # path = "./GFP-PCI_120_modified.h5"
    path = "./GFP-PCI_170_modified2.h5"
    source_data = h5py.File(path, 'r')
    data = source_data['data'][:]
    data = np.transpose(data, (0, 3, 2, 1))
    print("data max: %s, min: %s" % (np.max(data), np.min(data)))
    print("data shape:", data.shape)
    num_imgs = data.shape[0]
    mod = num_imgs % args.batch_size
    n_batches = int(num_imgs // args.batch_size)
    print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source_imgs = data[:-mod]
    else:
        source_imgs = data

    # training_dir_ir = "./source images/GFP-PC/GFP/"
    # folder_dataset_train_ir = glob.glob(training_dir_ir+ "*.bmp" )
    # training_dir_vi = "./source images/GFP-PC/PCI/"
    # folder_dataset_train_vi= glob.glob(training_dir_vi+ "*.bmp" )
    # folder_dataset_train_both=[]  #用来保存成对路径
    # for i in range(len(folder_dataset_train_vi)):
    #     temp=[folder_dataset_train_ir[i],folder_dataset_train_vi[i]]
    #     folder_dataset_train_both.append(temp)

    # transform_train = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Normalize((0.485, 0.456, 0.406),
    #                                                            (0.229, 0.224, 0.225))
    #                                       ])

    # dataset_train_ir = GetDataset(imageFolderDataset=folder_dataset_train_both,
    #                                               transform=transform_train)
    # dataset_train_vi = GetDataset(imageFolderDataset=folder_dataset_train_both,
    #                               transform=transform_train)
    # train_loader_ir = DataLoader(dataset_train_ir,
    #                           shuffle=True,
    #                           batch_size=args.batch_size)
    # train_loader_vi = DataLoader(dataset_train_vi,
    #                              shuffle=True,
    #                              batch_size=args.batch_size)

    #控制是否使用gpu
    model = net(in_channel=2)
    if use_gpu:
        model = model.cuda()
        model.cuda()
    else:
        model = model

    criterion_ssim_ir = ssim_ir
    criterion_ssim_vi = ssim_vi
    criterion_RMI_ir = RMI_ir
    criterion_RMI_vi=RMI_vi
    criterion_Intensity = intensity
    criterion_Grad=joint_grad


    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_content',
                                'loss_structure',
                                'loss_ssim_gfp',
                                'loss_ssim_pci',
                                'loss_RMI_gfp',
                                'loss_RMI_pci',
                                'loss_intensity_gfp',
                                'loss_intensity_pci',
                                'loss_grad',
                                ])

    # 创建 SummaryWriter 对象，指定日志文件保存路径
    writer = SummaryWriter('./tensorboard_logs')

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(writer,args, source_imgs,data,n_batches, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi,criterion_Intensity,criterion_Grad, optimizer, epoch)     # 训练集
        # train_log = train(args, source_imgs,data,n_batches, model, criterion_ssim_ir, criterion_ssim_vi,criterion_RMI_ir,criterion_RMI_vi, optimizer, epoch)     # 训练集

        print('loss: %.4f -- loss_content: %.4f -- loss_structure: %.4f-- loss_ssim_gfp: %.4f -- loss_ssim_pci: %.4f -- loss_RMI_gfp: %.4f -- loss_RMI_pci: %.4f-- loss_intensity_gfp: %.4f -- loss_intensity_pci: %.4f -- loss_grad: %.4f '
              % (train_log['loss'],
                 train_log['loss_content'],
                 train_log['loss_structure'],
                 train_log['loss_ssim_gfp'],
                 train_log['loss_ssim_pci'],
                 train_log['loss_RMI_gfp'],
                 train_log['loss_RMI_pci'],
                 train_log['loss_intensity_gfp'],
                 train_log['loss_intensity_pci'],
                 train_log['loss_grad'],
                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_content'],
            train_log['loss_structure'],
            train_log['loss_ssim_gfp'],
            train_log['loss_ssim_pci'],
            train_log['loss_RMI_gfp'],
            train_log['loss_RMI_pci'],
            train_log['loss_intensity_gfp'],
            train_log['loss_intensity_pci'],
            train_log['loss_grad'],

        ], index=['epoch', 'loss','loss_content', 'loss_structure','loss_ssim_gfp', 'loss_ssim_pci', 'loss_RMI_gfp', 'loss_RMI_pci','loss_intensity_gfp','loss_intensity_pci','loss_grad'])

        # log = log.append(tmp, ignore_index=True)#保存日志的，有问题先注释
        log = append_to_dataframe(log, tmp)  #因为pd.DataFrame对象不支持数据改变，所以自己写了一个函数来实现日志的覆盖保存
        log.to_csv('logs/%s/log.csv' %args.name, index=False)  #每一个epoch都会覆盖保存一次

        if (epoch+1) % 1 == 0:  #每一个epoch完了保存一次模型
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)
    # # 在训练结束后关闭 SummaryWriter
    writer.close()

if __name__ == '__main__':
    main()


