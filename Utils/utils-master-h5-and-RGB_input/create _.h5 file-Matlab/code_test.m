%%% create_h5_file文件细节测试
clc;
clear;
%%预定义常量
savepath='****.h5';
size_input=192;
stride=200;
Format='*.m';
img_ch=1; % image channel: 1 for gray, 3 for RGB
ratio=1;

data=zeros(size_input,size_input,2*img_ch,1);
%data1=zeros(192,192,2*3,1);
folder1='.';
filepath1=dir(fullfile(folder1,Format));
folder2='.';
filepath2=dir(fullfile(folder2,Format));

for i=1:20
if mod(i,10)==0
   i
end
end

for x=1:stride:(192-size_input+1)
    x
end