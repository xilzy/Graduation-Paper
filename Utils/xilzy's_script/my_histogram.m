%matlab热力学着色直方分布图（必须是24位图以上适用，但是直接用8_to_24会出现离散线段图）

close all;clear all;clc;
array=zeros(1,256);

%% 参数(通过X,Y轴范围控制显示区域)
%color='cool';  %冷色调
color='parula'; %暖色调
Y_lim=450;  %Y轴的范围
X_lim=[0,256];  %X轴的范围

%% 读取图像,并做概率统计
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-ct\24\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-pet\';% 图像文件夹路径
% file_path = 'E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-spect\';% 图像文件夹路径
% file_path = 'C:\Users\HUAWEI\Desktop\图片\1\';% 图像文件夹路径
file_path = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\主观评价图\消融实验\enlarge\';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有png格式的图像
img_num = length(img_path_list);%获取图像总数量

%I1=imread('E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\表格\消融实验\mri-ct\24\35_both.png');
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));% 显示正在处理的图像名
        [m,n,h]=size(image);
        pixel = zeros(256,h);
        for k=1:h
           for i=1:m  %统计灰度像素出现个数
               for j=1:n
                  pixel(image(i,j)+1,k) = pixel(image(i,j)+1,k)+1;
               end
           end
        end
        %% 分区间着色(热力学着色)
        %画直方图
        fig=figure;
        h = bar(pixel(:,2),1);
        %设置用于着色的CData
        h.CData = pixel(:,2);
        %将直方图的FaceColor属性设置为flat，这样就可以用CDate里的数据进行热力图着色
        h.FaceColor = 'flat';
        %设置热力图的数值范围
        caxis([0 400]);
        % %显示色标
        % colorbar
        %设置热力图的模式
        colormap(color); 
        set(gca,'XLim',X_lim);%X轴的数据显示范围
        set(gca,'XTick');%设置要显示坐标刻度

        set(gca,'XTickLabel');%给坐标加标签 

        %设置y轴范围和刻度

        set(gca,'YLim',[0 Y_lim]);%X轴的数据显示范围

        set(gca,'YTick');%设置要显示坐标刻度

        set(gca,'YTickLabel');%给坐标加标签 
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-ct\直方图\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-pet\直方图\';
%         file_path2='E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\升级一区工作\WGIF_detail_enhancement2\自增强\表格\消融实验\mri-spect\直方图\';
%           file_path2 = 'C:\Users\HUAWEI\Desktop\直方图\1\';
          file_path2 = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\主观评价图\消融实验\enlarge\直方图\';
          image_name2=strcat(file_path2,image_name);
          % 调整背景为透明
          set(gcf, 'Color', 'none');       % figure 背景透明
          set(gca, 'Color', 'none');       % 坐标轴背景透明    
          % 保存为透明 PNG
          exportgraphics(gca, strcat(file_path2, image_name), 'BackgroundColor', 'none', 'ContentType', 'image');
    end
end



% %将pixel转置，配合stacked模式
% pixel_=pixel.';
% subplot(1,1,1);
% %y2=randn(2008,3);
% %imhist(I1), axis tight, title('直方图')
% %% 初步绘制直方图
% hold on;
% histogram1=bar(pixel);
% hold off;
%set(gca,'XLim',[0 256]);%X轴的数据显示范围

% set(gca,'XTick');%设置要显示坐标刻度
% 
% set(gca,'XTickLabel');%给坐标加标签 
% 
% %设置y轴范围和刻度
% 
% set(gca,'YLim',[0 600]);%X轴的数据显示范围
% 
% set(gca,'YTick');%设置要显示坐标刻度
% 
% set(gca,'YTickLabel');%给坐标加标签 

% %% 美化设置
% % 设置颜色
% set(histogram(1),'FaceColor','r'); 
% set(histogram(2),'FaceColor','g');
% set(histogram(3),'FaceColor','b');
    
% %% 增加边界轮廓(实际上就是用plot画折线作为边界)
%     hold on
%     plot(X,Y(1,:),'Color',[1 0.1882 0.1882]);  %加上边界轮廓
%     plot(X,Y(2,:),'Color',[0.5 1 0]);
%     plot(X,Y(3,:),'Color',[0 0.5 1]);
%     hold off

% %% 使用hist函数
% hist(pixel,1000);