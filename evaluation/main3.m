close all;
clear;
clc;
delete(gcp('nocreate'));
p = parpool('local',4) ;
currentFolder = pwd;


%%% 消融实验(什么都不加和两个都加的情况再对比实验里面)
% fusion_name={'modify6','modify7'};
% nums=30;

%%% L_structure系数实验
% fusion_name={'modify0'};
% fusion_name={'modify1','modify2','modify3','modify4','modify5'};
% nums=30;

%%%拓展实验
fusion_name={'LES','NSCT','MSTSR','CNN','DSAGAN','IFCNN','EMFusion','MDFNet'};
nums=20;
% nums=30;


n=length(fusion_name(:));                    %外层循环的个数 
for fm=1:n               %最外层控制不同方法的文件夹
name = fusion_name{fm};
disp(name);


%拼接路径
% fused_path = strcat('./DATA/MATR/',name,'/');
% fused_path = strcat('./DATA/方法对比/mri-pet/',name,'/第三篇/');
fused_path = strcat('./DATA/方法对比/mri-spect/',name,'/第三篇/');

    for i =1:1:nums
        %读取源图像对
%         source_ir  = ['E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/GFP/',num2str(i),'.jpg'];
%         source_vis = ['E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/PCI/',num2str(i),'.jpg'];
%         source_ir  = ['E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/GFP/',num2str(i),'.jpg'];
%         source_vis = ['E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/PCI/',num2str(i),'.jpg'];
%         source_ir  = ['E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-pet/mri/',num2str(i),'.png'];
%         source_vis = ['E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-pet/pet/',num2str(i),'.png'];
        source_ir  = ['E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-spect/mri/',num2str(i),'.png'];
        source_vis = ['E:/论文写作/杨老师/论文投稿/BSPC/第一篇文章/材料/检验图片/mri-spect/spect/',num2str(i),'.png'];

        
        image1 = imread(source_ir);
        image2 = imread(source_vis);
        %这里把所有彩色图都转换成灰度图来进行比较了
        if size(image1, 3) == 3
            image1 = rgb2gray(image1);
        end
        if size(image2, 3) == 3
            image2 = rgb2gray(image2);
        end

%             fused   = strcat(fused_path,num2str(i),'.jpg'); %拼接融合图像路径
            fused   = strcat(fused_path,num2str(i),'.png'); %拼接融合图像路径
            fused_image   = imread(fused);   %读取融合图像
        %end
        disp(fused);
        if size(fused_image, 3) == 3
            fused_image = rgb2gray(fused_image);
        end
        %这里将源图像和融合图像配准，再评估
        image1 = imresize(image1,size(fused_image));
        image2 = imresize(image2,size(fused_image));
        result = evalution(image1,image2,fused_image);
        a(i,:) = result;  %a.mat保存的是一个文件夹下一张图的信息
    end
    
    b(:) = sum(a(:,:))/nums;   %b.mat保存的是一个文件夹下所有图片指标的平均值
    c(fm,:) = b;            %c.mat保存的是几个文件夹的平均值
    save_path = strcat(fused_path,'/每张图片指标.xlsx');
    %构建表格内容
%     column_name0={'边缘强度 EI','空间频率 SF','信息熵EN','SSIM','MI','标准差 SD','清晰度 DF','平均梯度 AG','相关系数 CC','视觉保真度 VIFF','PSNR','Qabf','QNCIE','QCB','QCV','OCE','SCD','FMI_w','FMI_dct','MSSSIM','FMI_pixel','QSF','QMI','QS','Nabf','QG','QP','QW','QE'};
    column_name0={'边缘强度 EI','空间频率 SF','信息熵EN','MI','标准差 SD','清晰度 DF','平均梯度 AG','相关系数 CC','视觉保真度 VIFF','Nabf'};
    m=length(column_name0(:)); 
    data0=cell(nums+1,m+1);
    row_name0=1:nums;
    row_name0=num2cell(row_name0);
    data0(1,2:end)=column_name0;
    data0(2:end,1)=row_name0;
    data0(2:end,2:end)=num2cell(a);
    xlswrite(save_path,data0) ;
end
%在最后一种方法的文件夹下保存所有的平均值
row_name=fusion_name;
% column_name={'编号','边缘强度 EI','空间频率 SF','信息熵EN','SSIM','MI','标准差 SD','清晰度 DF','平均梯度 AG','相关系数 CC','视觉保真度 VIFF','PSNR','Qabf','QNCIE','QCB','QCV','OCE','SCD','FMI_w','FMI_dct','MSSSIM','FMI_pixel','QSF','QMI','QS','Nabf','QG','QP','QW','QE'};
column_name={'编号','边缘强度 EI','空间频率 SF','信息熵EN','MI','标准差 SD','清晰度 DF','平均梯度 AG','相关系数 CC','视觉保真度 VIFF','Nabf'};
m=length(column_name(:));
data=cell(fm+1,m+1);    %开始构建表格内容
data(1,2:end)=column_name;      %列抬头
data(2:end,1)=row_name;         %行抬头

row_name0=1:n;
row_name0=num2cell(row_name0);
data(2:end,2)=row_name0;         %行编号，用于辅助排序的
data(2:end,3:end)=num2cell(c);
save_path2=strcat(fused_path,'/平均值比较.xlsx');
xlswrite(save_path2,data);

delete(gcp('nocreate'));

