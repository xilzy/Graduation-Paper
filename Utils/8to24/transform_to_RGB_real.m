%%%%8位栅格图转化为24位真彩色(彩色版)
clc;
clear all;
%批量
%file_path = 'F:\论文实验复现\杨老师\马佳义团队\2021\EMFusion-main\PET-MRI\test_imgs\pet\';% 图像文件夹路径
file_path ='F:\论文实验复现\杨老师\马佳义团队\2021\EMFusion-main\SPECT-MRI\test_imgs\spect\';
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        [image,map] = imread(strcat(file_path,image_name));%读取栅格图
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名
        RGB=ind2rgb(image,map);

        file_path2='F:\论文实验复现\杨老师\马佳义团队\2021\EMFusion-main\Transform_RGB_results\';
        image_name2=strcat(file_path2,image_name);
        imwrite(RGB,image_name2);
    end
end

