% 由rgb图得到Y通道图
clc;
clear;
close all;

%批量读取图片处理保存
file_path ='F:\论文实验复现\杨老师\马佳义团队\2020\FusionDN-master\test_imgs\pet_mri\pet\';
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名和原图一致
        RGB= imread(strcat(file_path,image_name));%读取RGB图
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名
        YCbCr=rgb2YCbCr(RGB);
        Y=YCbCr(:,:,1);
        Y=uint8(Y);

        file_path2='F:\论文实验复现\杨老师\马佳义团队\2020\FusionDN-master\test_imgs\pet_mri\pet_Y\';
        image_name2=strcat(file_path2,image_name);
        imwrite(Y,image_name2);
    end
end

