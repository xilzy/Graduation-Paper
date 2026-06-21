%%% create_h5_file文件整体测试
clc;
clear;
close all;

%%% 预定义常量
savepath='F:\论文实验复现\杨老师\马佳义团队\2020\MGMDcGAN-master\code\Medical_dataset.h5';
size_input=192; %必须要小于等于读取图像的大小
stride=200;
Format='*.png';
img_ch=1; 
ratio=1; 

% initialization  初始化
data=zeros(size_input,size_input,2*img_ch,1); %192x192x2x1的全零矩阵,这四个参数分别控制WHCN

%这里应该是两个源图像对应的文件夹
folder1='F:\论文实验复现\杨老师\马佳义团队\2020\MGMDcGAN-master\code\medical_test_imgs\MR-T1';
filepath1=dir(fullfile(folder1,Format));
folder2='F:\论文实验复现\杨老师\马佳义团队\2020\MGMDcGAN-master\code\medical_test_imgs\PET-I';
filepath2=dir(fullfile(folder2,Format));

count=0;
L1=length(filepath1);
L2=length(filepath2);
L=min([L1,L2]);   %取图片数更少的文件夹为基准，因为要成对融合

tic  %保存当前时间
for i=1:L
    %读取一对源图像
    img1=imread(fullfile(folder1,filepath1(i).name));
    img2=imread(fullfile(folder2,filepath2(i).name));
    
    %读取图像的矩阵大小
    [height,width,channel]=size(img1);
    %这里是在调整图片的大小
    for c=1:channel
        img1_double(:,:,c)=im2double(img1(:,:,c));
        img2_double(:,:,c)=im2double(img2(:,:,c));    
        i1(:,:,c)=imresize(img1_double(:,:,c),[fix(height*ratio),fix(width*ratio)]);
        i2(:,:,c)=imresize(img2_double(:,:,c),[fix(height*ratio),fix(width*ratio)]);      
    end
    clear img1 img2

    %记录缩放之后的新图像矩阵大小
    [height,width,channel]=size(i1);
   
    %这里是把图像数据写入data
    for x=1:stride:(height-size_input+1)
        for y=1:stride:(width-size_input+1)    
            sub_img1(:,:,:)=i1(x:x+size_input-1,y:y+size_input-1,:);
            sub_img2(:,:,:)=i2(x:x+size_input-1,y:y+size_input-1,:);
            count=count+1;
            data(:,:,1:1+img_ch-1,count)=sub_img1(:,:,:);
            data(:,:,1+img_ch:1+2*img_ch-1,count)=sub_img2(:,:,:);
              
            %这里有画图显示
            figure(1)
            A=data(:,:,1:1+img_ch-1,count);
            B=data(:,:,1+img_ch:1+2*img_ch-1,count);
            subplot(221),imshow(A);
            subplot(222),imshow(B);
            subplot(223),imshow(sub_img1);
            subplot(224),imshow(sub_img2);
        end
    end
      
    if mod(i,10)==0   %每10次循环(10对图像)就记录一次程序运行时间
        i
        toc
    end
    clear i1 i2 img1_double img2_double  %每一次循环完了清除这4个变量，以便下一次读取新的
end

order=randperm(count);  %1-count的随机排列行向量
data=data(:,:,:,order);

chunksz=10;    %每一块几张图片
totalct=0;
create_flag=true;

for batch_num=1:floor(count/chunksz) %依次写入每一块数据
    lastread=(batch_num-1)*chunksz;
    batch_data=data(:,:,:,lastread+1:lastread+chunksz);
    startloc=struct('dat',[1,1,1,totalct+1]);
    if batch_num~=1
        create_flag=false;     %除了第一块以外都是追加
    end
    curr_dat_sz=store2hdf5(savepath,batch_data,create_flag,startloc,chunksz); %这里调用了自定义的函数store2hdf5
    totalct=curr_dat_sz(end);
end
