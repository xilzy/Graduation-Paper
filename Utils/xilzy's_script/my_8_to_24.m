%批量(方法1到3原理都是把源图像值同时作为三通道的值)
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-ct\24\';
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-pet\24\';
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\mri\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\ct\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\';% 图像文件夹路径
% file_path = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\D2FNet\第一篇\';% 图像文件夹路径
% file_path = 'C:\Users\HUAWEI\Desktop\图片\1\';% 图像文件夹路径
file_path = 'C:\Users\HUAWEI\Desktop\图片\2\';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有png格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名
%方法一：
%         [rows,cols]=size(image);
%          r=zeros(rows,cols);
%          g=zeros(rows,cols);
%          b=zeros(rows,cols);
%          r=double(image);
%          g=double(image);
%          b=double(image);
%          rgb=cat(3,r,g,b);
%         x=uint8(rgb);
%         %方法二：
%         x(:,:,1) = image;
%         x(:,:,2) = image;
%         x(:,:,3) = image;
%方法三：(使用repmat方法填充)
x = repmat(image,[1,1,3]);%将单通道图片转换为三通道图片
%         file_path2 = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-ct\24\';
%         file_path2 = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-pet\24\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\mri\24\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\ct\24\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri\24\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri\24\';
%         file_path2='F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\D2FNet\第一篇\24\';
%           file_path2 = 'C:\Users\HUAWEI\Desktop\图片\1\';
          file_path2 = 'C:\Users\HUAWEI\Desktop\图片\2\';
        image_name2=strcat(file_path2,image_name);
        imwrite(x,image_name2);
    end
end

