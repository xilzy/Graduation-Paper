clear all;
close all;
clc;
nums=30;    %处理的图片数，也是横坐标
%% 导入excel数据
NSCT_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\NSCT\第三篇\每张图片指标.xlsx','range','B1:AD31'); %要把标题行读进去，不然会把第一行当数据
LES_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\LES\第三篇\每张图片指标.xlsx','range','B1:AD31');
MSTSR_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\MSTSR\第三篇\每张图片指标.xlsx','range','B1:AD31');
CNN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\CNN\第三篇\每张图片指标.xlsx','range','B1:AD31');
IFCNN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\IFCNN\第三篇\每张图片指标.xlsx','range','B1:AD31');
EMFusion_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\EMFusion\第三篇\每张图片指标.xlsx','range','B1:AD31');
MURF_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\MURF\第三篇\每张图片指标.xlsx','range','B1:AD31');
MATR_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\MATR\第三篇\每张图片指标.xlsx','range','B1:AD31');
DPCN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\DPCN\第三篇\每张图片指标.xlsx','range','B1:AD31');
MDFNet_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\proposed\第三篇\每张图片指标.xlsx','range','B1:AD31');


% 转换表数据为double矩阵
NSCT=table2array(NSCT_t);
LES=table2array(LES_t);
MSTSR=table2array(MSTSR_t);
CNN=table2array(CNN_t);
IFCNN=table2array(IFCNN_t);
EMFusion=table2array(EMFusion_t);
MURF=table2array(MURF_t);
MATR=table2array(MATR_t);
DPCN=table2array(DPCN_t);
MDFNet=table2array(MDFNet_t);

% 提取X轴和Y轴(先从表中读出对应列，再旋转90度变成行向量作为Y值)

%公共横坐标
X=1:nums; 

% EI_Y 
NSCT_EI=NSCT(:,1);
NSCT_EI=rot90(NSCT_EI);
LES_EI=LES(:,1);
LES_EI=rot90(LES_EI);
MSTSR_EI=MSTSR(:,1);
MSTSR_EI=rot90(MSTSR_EI);
CNN_EI=CNN(:,1);
CNN_EI=rot90(CNN_EI);
IFCNN_EI=IFCNN(:,1);
IFCNN_EI=rot90(IFCNN_EI);
EMFusion_EI=EMFusion(:,1);
EMFusion_EI=rot90(EMFusion_EI);
MURF_EI=MURF(:,1);
MURF_EI=rot90(MURF_EI);
MATR_EI=MATR(:,1);
MATR_EI=rot90(MATR_EI);
DPCN_EI=DPCN(:,1);
DPCN_EI=rot90(DPCN_EI);
MDFNet_EI=MDFNet(:,1);
MDFNet_EI=rot90(MDFNet_EI);

% SF_Y 
NSCT_SF=NSCT(:,2);
NSCT_SF=rot90(NSCT_SF);
LES_SF=LES(:,2);
LES_SF=rot90(LES_SF);
MSTSR_SF=MSTSR(:,2);
MSTSR_SF=rot90(MSTSR_SF);
CNN_SF=CNN(:,2);
CNN_SF=rot90(CNN_SF);
IFCNN_SF=IFCNN(:,2);
IFCNN_SF=rot90(IFCNN_SF);
EMFusion_SF=EMFusion(:,2);
EMFusion_SF=rot90(EMFusion_SF);
MURF_SF=MURF(:,2);
MURF_SF=rot90(MURF_SF);
MATR_SF=MATR(:,2);
MATR_SF=rot90(MATR_SF);
DPCN_SF=DPCN(:,2);
DPCN_SF=rot90(DPCN_SF);
MDFNet_SF=MDFNet(:,2);
MDFNet_SF=rot90(MDFNet_SF);

% EN_Y 
NSCT_EN=NSCT(:,3);
NSCT_EN=rot90(NSCT_EN);
LES_EN=LES(:,3);
LES_EN=rot90(LES_EN);
MSTSR_EN=MSTSR(:,3);
MSTSR_EN=rot90(MSTSR_EN);
CNN_EN=CNN(:,3);
CNN_EN=rot90(CNN_EN);
IFCNN_EN=IFCNN(:,3);
IFCNN_EN=rot90(IFCNN_EN);
EMFusion_EN=EMFusion(:,3);
EMFusion_EN=rot90(EMFusion_EN);
MURF_EN=MURF(:,3);
MURF_EN=rot90(MURF_EN);
MATR_EN=MATR(:,3);
MATR_EN=rot90(MATR_EN);
DPCN_EN=DPCN(:,3);
DPCN_EN=rot90(DPCN_EN);
MDFNet_EN=MDFNet(:,3);
MDFNet_EN=rot90(MDFNet_EN);

% MI_Y 
NSCT_MI=NSCT(:,5);
NSCT_MI=rot90(NSCT_MI);
LES_MI=LES(:,5);
LES_MI=rot90(LES_MI);
MSTSR_MI=MSTSR(:,5);
MSTSR_MI=rot90(MSTSR_MI);
CNN_MI=CNN(:,5);
CNN_MI=rot90(CNN_MI);
IFCNN_MI=IFCNN(:,5);
IFCNN_MI=rot90(IFCNN_MI);
EMFusion_MI=EMFusion(:,5);
EMFusion_MI=rot90(EMFusion_MI);
MURF_MI=MURF(:,5);
MURF_MI=rot90(MURF_MI);
MATR_MI=MATR(:,5);
MATR_MI=rot90(MATR_MI);
DPCN_MI=DPCN(:,5);
DPCN_MI=rot90(DPCN_MI);
MDFNet_MI=MDFNet(:,5);
MDFNet_MI=rot90(MDFNet_MI);

% SD_Y 
NSCT_SD=NSCT(:,6);
NSCT_SD=rot90(NSCT_SD);
LES_SD=LES(:,6);
LES_SD=rot90(LES_SD);
MSTSR_SD=MSTSR(:,6);
MSTSR_SD=rot90(MSTSR_SD);
CNN_SD=CNN(:,6);
CNN_SD=rot90(CNN_SD);
IFCNN_SD=IFCNN(:,6);
IFCNN_SD=rot90(IFCNN_SD);
EMFusion_SD=EMFusion(:,6);
EMFusion_SD=rot90(EMFusion_SD);
MURF_SD=MURF(:,6);
MURF_SD=rot90(MURF_SD);
MATR_SD=MATR(:,6);
MATR_SD=rot90(MATR_SD);
DPCN_SD=DPCN(:,6);
DPCN_SD=rot90(DPCN_SD);
MDFNet_SD=MDFNet(:,6);
MDFNet_SD=rot90(MDFNet_SD);

% DF_Y 
NSCT_DF=NSCT(:,7);
NSCT_DF=rot90(NSCT_DF);
LES_DF=LES(:,7);
LES_DF=rot90(LES_DF);
MSTSR_DF=MSTSR(:,7);
MSTSR_DF=rot90(MSTSR_DF);
CNN_DF=CNN(:,7);
CNN_DF=rot90(CNN_DF);
IFCNN_DF=IFCNN(:,7);
IFCNN_DF=rot90(IFCNN_DF);
EMFusion_DF=EMFusion(:,7);
EMFusion_DF=rot90(EMFusion_DF);
MURF_DF=MURF(:,7);
MURF_DF=rot90(MURF_DF);
MATR_DF=MATR(:,7);
MATR_DF=rot90(MATR_DF);
DPCN_DF=DPCN(:,7);
DPCN_DF=rot90(DPCN_DF);
MDFNet_DF=MDFNet(:,7);
MDFNet_DF=rot90(MDFNet_DF);


% AG_Y 
NSCT_AG=NSCT(:,8);
NSCT_AG=rot90(NSCT_AG);
LES_AG=LES(:,8);
LES_AG=rot90(LES_AG);
MSTSR_AG=MSTSR(:,8);
MSTSR_AG=rot90(MSTSR_AG);
CNN_AG=CNN(:,8);
CNN_AG=rot90(CNN_AG);
IFCNN_AG=IFCNN(:,8);
IFCNN_AG=rot90(IFCNN_AG);
EMFusion_AG=EMFusion(:,8);
EMFusion_AG=rot90(EMFusion_AG);
MURF_AG=MURF(:,8);
MURF_AG=rot90(MURF_AG);
MATR_AG=MATR(:,8);
MATR_AG=rot90(MATR_AG);
DPCN_AG=DPCN(:,8);
DPCN_AG=rot90(DPCN_AG);
MDFNet_AG=MDFNet(:,8);
MDFNet_AG=rot90(MDFNet_AG);

% CC_Y 
NSCT_CC=NSCT(:,9);
NSCT_CC=rot90(NSCT_CC);
LES_CC=LES(:,9);
LES_CC=rot90(LES_CC);
MSTSR_CC=MSTSR(:,9);
MSTSR_CC=rot90(MSTSR_CC);
CNN_CC=CNN(:,9);
CNN_CC=rot90(CNN_CC);
IFCNN_CC=IFCNN(:,9);
IFCNN_CC=rot90(IFCNN_CC);
EMFusion_CC=EMFusion(:,9);
EMFusion_CC=rot90(EMFusion_CC);
MURF_CC=MURF(:,9);
MURF_CC=rot90(MURF_CC);
MATR_CC=MATR(:,9);
MATR_CC=rot90(MATR_CC);
DPCN_CC=DPCN(:,9);
DPCN_CC=rot90(DPCN_CC);
MDFNet_CC=MDFNet(:,9);
MDFNet_CC=rot90(MDFNet_CC);

% VIFF_Y 
NSCT_VIFF=NSCT(:,10);
NSCT_VIFF=rot90(NSCT_VIFF);
LES_VIFF=LES(:,10);
LES_VIFF=rot90(LES_VIFF);
MSTSR_VIFF=MSTSR(:,10);
MSTSR_VIFF=rot90(MSTSR_VIFF);
CNN_VIFF=CNN(:,10);
CNN_VIFF=rot90(CNN_VIFF);
IFCNN_VIFF=IFCNN(:,10);
IFCNN_VIFF=rot90(IFCNN_VIFF);
EMFusion_VIFF=EMFusion(:,10);
EMFusion_VIFF=rot90(EMFusion_VIFF);
MURF_VIFF=MURF(:,10);
MURF_VIFF=rot90(MURF_VIFF);
MATR_VIFF=MATR(:,10);
MATR_VIFF=rot90(MATR_VIFF);
DPCN_VIFF=DPCN(:,10);
DPCN_VIFF=rot90(DPCN_VIFF);
MDFNet_VIFF=MDFNet(:,10);
MDFNet_VIFF=rot90(MDFNet_VIFF);




%% 画图
figure;                       %开启画布
t = tiledlayout(3,3,'TileSpacing','Compact','Padding','Compact');  %代替subplot，可以让图片紧凑，要最小化绘图之间的空间，请将 'TileSpacing' 名称-值对组参数设置为 'compact'。要使布局周围的空间最小化，请将 'Padding' 名称-值对组参数设置为 'compact'。
%EI
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
silver_gray='#bfbfbf';
melon_orange='#fbc093';
% disp(silver_gray);
% disp(melon_orange);
plot(X,MDFNet_EI,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_EI,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_EI,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_EI,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_EI,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_EI,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_EI,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_EI,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_EI,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_EI,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('EI','FontWeight','bold');
legend('MDFNet:76.1308','NSCT:50.5177','LES:49.2361','MSTSR:54.3511','CNN:53.4944','IFCNN:52.4307','EMFusion:40.3589','MURF:67.0447','MATR:43.7159','DPCN:40.9084','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[20,160]);     
set(gca,'ytick',[20:20:160]);
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%SF
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_SF,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_SF,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_SF,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_SF,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_SF,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_SF,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_SF,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_SF,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_SF,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_SF,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('SF','FontWeight','bold');
legend('MDFNet:16.4667','NSCT:11.7134','LES:12.4381','MSTSR:12.0269','CNN:11.6278','IFCNN:012.4748','EMFusion:8.8703','MURF:13.7836','MATR:9.1587','DPCN:8.6517','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[3,35]);     
set(gca,'ytick',[5:5:35]);
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%EN
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_EN,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_EN,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_EN,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_EN,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_EN,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_EN,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_EN,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_EN,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_EN,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_EN,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('EN','FontWeight','bold');
legend('MDFNet:7.3574','NSCT:6.4270','LES:6.7055','MSTSR:7.2435','CNN:6.6314','IFCNN:6.4366','EMFusion:6.3135','MURF:6.8332','MATR:6.5307','DPCN:6.4917','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[5,9.5]);     
set(gca,'ytick',[5:0.5:9.5]);
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%MI
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_MI,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_MI,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_MI,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_MI,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_MI,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_MI,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_MI,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_MI,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_MI,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_MI,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('MI','FontWeight','bold');
legend('MDFNet:14.7147','NSCT:12.8541','LES:13.4110','MSTSR:14.4870','CNN:13.2629','IFCNN:12.8732','EMFusion:12.6269','MURF:13.6664','MATR:13.0613','DPCN:12.9833','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[10,19]);     
set(gca,'ytick',[10:1:19]);
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%SD
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_SD,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_SD,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_SD,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_SD,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_SD,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_SD,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_SD,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_SD,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_SD,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_SD,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('SD','FontWeight','bold');
legend('MDFNet:44.9804','NSCT:24.4048','LES:27.8743','MSTSR:41.1376','CNN:27.3428','IFCNN:23.5416','EMFusion:21.6866','MURF:34.6379','MATR:24.9451','DPCN:23.0824','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[0,100]);     
set(gca,'ytick',[0:20:100]);
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%DF
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_DF,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_DF,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_DF,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_DF,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_DF,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_DF,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_DF,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_DF,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_DF,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_DF,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('DF','FontWeight','bold'); 
legend('MDFNet:8.3482','NSCT:5.7426','LES:6.0147','MSTSR:6.0418','CNN:5.8375','IFCNN:6.1217','EMFusion:4.4369','MURF:7.0306','MATR:4.6152','DPCN:4.4069','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[2,17]);     
set(gca,'ytick',[2:2:16]); 
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%AG
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_AG,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_AG,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_AG,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_AG,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_AG,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_AG,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_AG,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_AG,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_AG,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_AG,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('AG','FontWeight','bold');   
legend('MDFNet:7.2692','NSCT:4.8620','LES:4.8345','MSTSR:5.1920','CNN:5.1301','IFCNN:5.0848','EMFusion:3.8401','MURF:6.4163','MATR:4.1420','DPCN:3.8792','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[2,14]);     
set(gca,'ytick',[2:2:14]); 
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%CC
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_CC,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_CC,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_CC,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_CC,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_CC,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_CC,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_CC,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_CC,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_CC,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_CC,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('CC','FontWeight','bold');
legend('MDFNet:0.7066','NSCT:0.7149','LES:0.6591','MSTSR:0.5571','CNN:0.7120','IFCNN:0.7042','EMFusion:0.6404','MURF:0.7099','MATR:0.6478','DPCN:0.6488','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[0.3,1.1]);     
set(gca,'ytick',[0.3:0.1:1.1]); 
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

%VIFF
nexttile
%已经使用的颜色有红，黄，蓝，绿，黑，青，紫，自定义蓝，银灰，甜瓜橙
plot(X,MDFNet_VIFF,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_VIFF,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_VIFF,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_VIFF,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_VIFF,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_VIFF,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,EMFusion_VIFF,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
plot(X,MURF_VIFF,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,MATR_VIFF,'-. h ','color',silver_gray,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',silver_gray);
plot(X,DPCN_VIFF,'-. h ','color',melon_orange,'LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor',melon_orange);
hold off;
title('VIFF','FontWeight','bold');
legend('MDFNet:1.0787','NSCT:0.6447','LES:0.5335','MSTSR:0.6576','CNN:0.6074','IFCNN:0.5322','EMFusion:0.3643','MURF:0.6925','MATR:0.4272','DPCN:0.4079','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[0,3.2]);     
set(gca,'ytick',[0:0.5:3]); 
xlabel('Image Pairs', 'FontWeight', 'bold'); %添加横坐标

% 假设前面已经绘制完9张子图
% --- 布局设置 ---
set(gcf, 'Position', [0, 0, 1500, 1200]); % 控制图像大小

% --- 导出为高清 PDF ---
exportgraphics(gcf, 'FusionMetrics_Full.pdf', ...
    'ContentType', 'vector', 'Resolution', 300);

 
