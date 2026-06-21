close all;clear all;clc;
array=zeros(1,256);

%%读取图像,并做概率统计
%I1=imread('E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\表格\消融实验\mri-ct\24\35_both.png');
%I1=imread('E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\表格\消融实验\mri-pet\142_both.png')
I1=imread('E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\表格\消融实验\mri-pet\142_grad.png')
%img1=I1(:,:,1);
[m,n,h]=size(I1);
pixel = zeros(256,h);
for k=1:h
  for i=1:m  %统计灰度像素出现个数
     for j=1:n
       pixel(I1(i,j)+1,k) = pixel(I1(i,j)+1,k)+1;
     end
  end
end
subplot(1,1,1);
%imhist(img1), axis tight, title('直方图')
%% 初步绘制直方图
histogram=bar(pixel);
set(gca,'XLim',[0 256]);%X轴的数据显示范围

set(gca,'XTick');%设置要显示坐标刻度

set(gca,'XTickLabel');%给坐标加标签 

%设置y轴范围和刻度

set(gca,'YLim',[0 500]);%X轴的数据显示范围

set(gca,'YTick');%设置要显示坐标刻度

set(gca,'YTickLabel');%给坐标加标签 

%% 美化设置
% 设置颜色
set(histogram(1),'FaceColor','r'); 
set(histogram(2),'FaceColor','g');
set(histogram(3),'FaceColor','b');