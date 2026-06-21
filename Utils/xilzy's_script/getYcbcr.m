%给一张RGB图片，生成保存三张Y,Cb,Cr通道图（用于画框架图）
clc;
clear;

%读取输入
% input_path='F:\论文整理构思\评价指标\傅羽\test_imgs\mri-pet\pet\200.png';
% input_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-spect\spect\2.png';
% input_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\图表\框架图\3_GFP.jpg';
input_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\图表\框架图\3_fused_color.jpg';
RGB=imread(input_path);
YCbCr=rgb2ycbcr(RGB);     %直接使用matlab内置转换函数，自己实现那个有点问题
%imshow(RGB);
%imshow(YCbCr);
%imshow(YCbCr(:,:,1));

% %当前文件夹下保存图片
% imwrite(YCbCr(:,:,1),'pet_Y.png');
% imwrite(YCbCr(:,:,2),'pet_Cb.png');
% imwrite(YCbCr(:,:,3),'pet_Cr.png');
% imwrite(YCbCr(:,:,1),'spect_Y.png');
% imwrite(YCbCr(:,:,2),'spect_Cb.png');
% imwrite(YCbCr(:,:,3),'spect_Cr.png');
% imwrite(YCbCr(:,:,1),'GFP_Y.jpg');
% imwrite(YCbCr(:,:,2),'GFP_Cb.jpg');
% imwrite(YCbCr(:,:,3),'GFP_Cr.jpg');
imwrite(YCbCr(:,:,1),'3_fused.jpg');
