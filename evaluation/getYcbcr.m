%给一张RGB图片，生成保存三张Y,Cb,Cr通道图（用于画框架图）
clc;
clear;

%读取输入
input_path='F:\论文整理构思\评价指标\傅羽\test_imgs\mri-pet\pet\200.png';
RGB=imread(input_path);
YCbCr=rgb2ycbcr(RGB);     %直接使用matlab内置转换函数，自己实现那个有点问题
%imshow(RGB);
%imshow(YCbCr);
%imshow(YCbCr(:,:,1));

% %当前文件夹下保存图片
imwrite(YCbCr(:,:,1),'pet_Y.png');
imwrite(YCbCr(:,:,2),'pet_Cb.png');
imwrite(YCbCr(:,:,3),'pet_Cr.png');