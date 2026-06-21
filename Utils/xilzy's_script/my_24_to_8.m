%批量将24位图转化为8位图
close all;
clear;
clc;
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced1\AnisGF\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced2\AnisGF\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced1\AnisGF\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced2\AnisGF\';% 图像文件夹路径

% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced2\WGIF\';% 图像文件夹路径
file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced2\WGIF\';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有png格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名
        I=image(:, :, 1) ; %以R通道代替其他通道
%         save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced1\AnisGF\8\';
%         save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced2\AnisGF\8\';
%         save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced1\AnisGF\8\';
%         save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced2\AnisGF\8\';
%         save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\enhanced2\WGIF\8\';
        save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\enhanced2\WGIF\8\';
        save_name=strcat(save_path,image_name);
        imwrite(I,save_name); %覆盖原来的图像


    end
end

