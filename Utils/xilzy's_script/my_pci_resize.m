%%% 作用：将图像分辨率大小批量调整为指定大小

tic;
clc;
clear;
% nums=148;     %要修复的数量
nums=3;     %要修复的数量

%路径
% path='E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/GFP/';
% path='E:/论文写作/杨老师/论文投稿/第二篇/材料/检验图片/gfp-pci/PCI/';
path='C:\Users\HUAWEI\Desktop\毕业毕设\论文初稿\图像加密算法\ImageEncrypt-main\Code\source\';

%想要转换的大小
target_size=[256,256];

for i=1:nums
    %读取图片
    file_name=strcat(path,num2str(i+9),'.jpg');
    source=imread(file_name);

    %转换处理
    if(size(source,[1,2])==target_size)
        continue
    else
        disp(file_name);  %显示被转换图片的文件名
        % 调整图像大小为 358x358，并使用双三次插值法进行像素插值
        I = imresize(source, target_size, 'bicubic');

        %保存图片  
        imwrite(I,file_name);
    end


    
end
toc;