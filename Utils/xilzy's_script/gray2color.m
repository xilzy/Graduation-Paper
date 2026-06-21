%%% 把那些融合成灰度图像的转换成彩色(用YCbCr转换之后)
tic;
clc;
clear;
nums=30;     %要修复的数量
for i=1:nums
    %拼接图片路径
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\LES\第二篇\';
%     color_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\spect\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\LES\第二篇\';

%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\MSTSR\第二篇\';
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-spect\spect\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\MSTSR\第二篇\';

%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\NSCT\第二篇\';
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-spect\spect\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\NSCT\第二篇\';

%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\SR\第二篇\';
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-spect\spect\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\SR\第二篇\';
% 
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\DWT\第二篇\';
%     color_path='E:\论文写作\杨老师\论文投稿\CCF(EI)\第二篇文章\材料\检验图片\mri-spect\spect\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\DWT\第二篇\';

%     color_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\pet\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\MDFNet\第三篇\';
    color_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\spect\';
    fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\MDFNet\第三篇\';
    %读取图片
    color_path=strcat(color_path,num2str(i),'.png');
    fused_path=strcat(fused_path,num2str(i),'.png');
    Source=imread(color_path);
    Fuse=imread(fused_path);
    %转换处理
    YCbCr=rgb2ycbcr(Source);
    Fuse_color=cat(3,Fuse,YCbCr(:,:,2),YCbCr(:,:,3));
    Fuse_color=ycbcr2rgb(Fuse_color);
    %保存图片
    if exist(fused_path)==0 %%判断文件夹是否存在
      mkdir(fused_path);  
    end    
    imwrite(Fuse_color,fused_path);
    
end
toc;


% tic;
% clc;
% clear;
% nums=30;     %要修复的数量
% for i=1:nums
%     %拼接图片路径
%     color_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\gfp-pci\GFP\';
%     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\proposed\第三篇\';
% %     fused_path='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\NSCT\第三篇\';
%     %L_structure系数测定
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify1/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify2/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify3/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify4/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify5/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify0/';
%         %消融实验
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify6/';
% %     fused_path='F:/论文整理构思/评价指标/傅羽/DATA/MATR/modify7/';
% 
%     %读取图片
%     color_path=strcat(color_path,num2str(i),'.jpg');
%     fused_path=strcat(fused_path,num2str(i),'.jpg');
%     Source=imread(color_path);
%     Fuse=imread(fused_path);
%     %转换处理
%     YCbCr=rgb2ycbcr(Source);
%     Fuse_color=cat(3,Fuse,YCbCr(:,:,2),YCbCr(:,:,3));
%     Fuse_color=ycbcr2rgb(Fuse_color);
%     %保存图片
%     if exist(fused_path)==0 %%判断文件夹是否存在
%       mkdir(fused_path);  
%     end    
%     imwrite(Fuse_color,fused_path);
%     
% end
% toc;