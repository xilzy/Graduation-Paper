% %%% 两张图相减，体现差值(一般是大减小，负数结果舍入到零，所以有时候需要调换位置，看看谁减谁效果更好)
% close all;
% clear;
% clc;
% 
% %%%预定义参数变量
% %寻找差异最大mri
% % nums=47;
% % source_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\mri\';
% % enhanced_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\mri\enhanced2\WGIF\';
% % save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\difference\mri\';
% 
% %寻找差异最大ct
% % nums=47;
% % source_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\ct\';
% % enhanced_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\ct\enhanced2\WGIF\';
% % save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\difference\ct\';
% 
% %寻找差异最大pet
% % nums=20;
% % source_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\pet\';
% % enhanced_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\pet\enhanced2\WGIF\';
% % save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\difference\pet\';
% 
% %寻找差异最大spect
% nums=46;
% source_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\spect\';
% enhanced_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\spect\enhanced2\WGIF\';
% save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\difference\spect\';
% 
% %最终选定的4张
% % nums=4;
% % source_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\source\';
% % enhanced_path= 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\enhanced\';
% % save_path='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\源图像细节增强预处理说明图\difference\';
% 
% for i =1:1:nums
%   % 读取第一张图片
%   source=[source_path,num2str(i),'.png'];
%   image1 = imread(source);
%  
%   % 读取第二张图片
%   enhanced=[enhanced_path,num2str(i),'.png'];
%   image2 = imread(enhanced);
%  
%   % 将两张图片进行减法运算得到差异图像，并保存
% %   differenceImage = imsubtract(image1, image2);
%   differenceImage = imsubtract(image2, image1);  %Z = imsubtract(X,Y) 从数组 X 中的每个元素中减去数组 Y 中的对应元素，并在输出数组 Z 的对应元素中返回差。
%   save_name=strcat(save_path,num2str(i),'.png');
%   imwrite(differenceImage,save_name);
%  
% %   % 显示原始图片及其差异图像
% %   figure;
% %   subplot(1,3,1), imshow(image1), title('Original Image 1');
% %   subplot(1,3,2), imshow(image2), title('Original Image 2');
% %   subplot(1,3,3), imshow(differenceImage), title('Difference Image');
% end


%%% 两张图相减，体现差值(一般是大减小，负数结果舍入到零，所以有时候需要调换位置，看看谁减谁效果更好)
close all;
clear;
clc;

%%%预定义参数变量
nums=30;
% source_path= 'E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\差值图\DPCN\';
% enhanced_path= 'E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\差值图\MDFNet\';
% save_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\差值图\';
% source_path= 'F:\论文整理构思\评价指标\傅羽\DATA\MATR\modify6\';  %只添加L_content
source_path= 'F:\论文整理构思\评价指标\傅羽\DATA\MATR\modify7\';  %只添加L_structure
enhanced_path= 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\proposed\第三篇\';
% save_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\主观评价图\拓展实验\结构损失函数差值图\';
save_path='E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\主观评价图\拓展实验\内容损失函数差值图\';


for i =1:1:nums
  % 读取第一张图片
  source=[source_path,num2str(i),'.jpg'];
  image1 = imread(source);
 
  % 读取第二张图片
  enhanced=[enhanced_path,num2str(i),'.jpg'];
  image2 = imread(enhanced);
 
  % 将两张图片进行减法运算得到差异图像，并保存
%   differenceImage = imsubtract(image1, image2);
  differenceImage = imsubtract(image2, image1);  %Z = imsubtract(X,Y) 从数组 X 中的每个元素中减去数组 Y 中的对应元素，并在输出数组 Z 的对应元素中返回差。
  save_name=strcat(save_path,num2str(i),'.jpg');
  imwrite(differenceImage,save_name);
 
% %   % 显示原始图片及其差异图像
% %   figure;
%   subplot(1,3,1), imshow(image1), title('Original Image 1');
%   subplot(1,3,2), imshow(image2), title('Original Image 2');
%   subplot(1,3,3), imshow(differenceImage), title('Difference Image');
end