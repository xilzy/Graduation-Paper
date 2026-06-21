clear all;
close all;
clc;
nums=30;    %处理的图片数，也是横坐标
%% 导入excel数据
NSCT_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\NSCT\第一篇\每张图片指标.xlsx','range','B1:AD31'); %要把标题行读进去，不然会把第一行当数据
LES_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\LES\第一篇\每张图片指标.xlsx','range','B1:AD31');
MSTSR_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\MSTSR\第一篇\每张图片指标.xlsx','range','B1:AD31');
CNN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\CNN\第一篇\每张图片指标.xlsx','range','B1:AD31');
IFCNN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\IFCNN\第一篇\每张图片指标.xlsx','range','B1:AD31');
DSAGAN_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\DSAGAN\第一篇\每张图片指标.xlsx','range','B1:AD31');
EMFusion_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\EMFusion\第一篇\每张图片指标.xlsx','range','B1:AD31');
% D2FNet_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\D2FNet\第一篇\每张图片指标.xlsx','range','B1:AD31');
D2FNet_t=readtable('F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\wgif-D2FNet\第一篇\每张图片指标.xlsx','range','B1:AD31');


% 转换表数据为double矩阵
NSCT=table2array(NSCT_t);
LES=table2array(LES_t);
MSTSR=table2array(MSTSR_t);
CNN=table2array(CNN_t);
IFCNN=table2array(IFCNN_t);
DSAGAN=table2array(DSAGAN_t);
EMFusion=table2array(EMFusion_t);
D2FNet=table2array(D2FNet_t);

% 提取X轴和Y轴(先从表中读出对应列，再旋转90度变成行向量作为Y值)

%公共横坐标
X=1:nums; 

% SSIM_Y 
NSCT_SSIM=NSCT(:,4);
NSCT_SSIM=rot90(NSCT_SSIM);
LES_SSIM=LES(:,4);
LES_SSIM=rot90(LES_SSIM);
MSTSR_SSIM=MSTSR(:,4);
MSTSR_SSIM=rot90(MSTSR_SSIM);
CNN_SSIM=CNN(:,4);
CNN_SSIM=rot90(CNN_SSIM);
IFCNN_SSIM=IFCNN(:,4);
IFCNN_SSIM=rot90(IFCNN_SSIM);
DSAGAN_SSIM=DSAGAN(:,4);
DSAGAN_SSIM=rot90(DSAGAN_SSIM);
EMFusion_SSIM=EMFusion(:,4);
EMFusion_SSIM=rot90(EMFusion_SSIM);
D2FNet_SSIM=D2FNet(:,4);
D2FNet_SSIM=rot90(D2FNet_SSIM);

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
DSAGAN_CC=DSAGAN(:,9);
DSAGAN_CC=rot90(DSAGAN_CC);
EMFusion_CC=EMFusion(:,9);
EMFusion_CC=rot90(EMFusion_CC);
D2FNet_CC=D2FNet(:,9);
D2FNet_CC=rot90(D2FNet_CC);

% PSNR_Y
NSCT_PSNR=NSCT(:,11);
NSCT_PSNR=rot90(NSCT_PSNR);
LES_PSNR=LES(:,11);
LES_PSNR=rot90(LES_PSNR);
MSTSR_PSNR=MSTSR(:,11);
MSTSR_PSNR=rot90(MSTSR_PSNR);
CNN_PSNR=CNN(:,11);
CNN_PSNR=rot90(CNN_PSNR);
IFCNN_PSNR=IFCNN(:,11);
IFCNN_PSNR=rot90(IFCNN_PSNR);
DSAGAN_PSNR=DSAGAN(:,11);
DSAGAN_PSNR=rot90(DSAGAN_PSNR);
EMFusion_PSNR=EMFusion(:,11);
EMFusion_PSNR=rot90(EMFusion_PSNR);
D2FNet_PSNR=D2FNet(:,11);
D2FNet_PSNR=rot90(D2FNet_PSNR);

% QCB_Y
NSCT_QCB=NSCT(:,14);
NSCT_QCB=rot90(NSCT_QCB);
LES_QCB=LES(:,14);
LES_QCB=rot90(LES_QCB);
MSTSR_QCB=MSTSR(:,14);
MSTSR_QCB=rot90(MSTSR_QCB);
CNN_QCB=CNN(:,14);
CNN_QCB=rot90(CNN_QCB);
IFCNN_QCB=IFCNN(:,14);
IFCNN_QCB=rot90(IFCNN_QCB);
DSAGAN_QCB=DSAGAN(:,14);
DSAGAN_QCB=rot90(DSAGAN_QCB);
EMFusion_QCB=EMFusion(:,14);
EMFusion_QCB=rot90(EMFusion_QCB);
D2FNet_QCB=D2FNet(:,14);
D2FNet_QCB=rot90(D2FNet_QCB);

% Qs_Y
NSCT_QS=NSCT(:,24);
NSCT_QS=rot90(NSCT_QS);
LES_QS=LES(:,24);
LES_QS=rot90(LES_QS);
MSTSR_QS=MSTSR(:,24);
MSTSR_QS=rot90(MSTSR_QS);
CNN_QS=CNN(:,24);
CNN_QS=rot90(CNN_QS);
IFCNN_QS=IFCNN(:,24);
IFCNN_QS=rot90(IFCNN_QS);
DSAGAN_QS=DSAGAN(:,24);
DSAGAN_QS=rot90(DSAGAN_QS);
EMFusion_QS=EMFusion(:,24);
EMFusion_QS=rot90(EMFusion_QS);
D2FNet_QS=D2FNet(:,24);
D2FNet_QS=rot90(D2FNet_QS);

% Qp_Y
NSCT_QP=NSCT(:,27);
NSCT_QP=rot90(NSCT_QP);
LES_QP=LES(:,27);
LES_QP=rot90(LES_QP);
MSTSR_QP=MSTSR(:,27);
MSTSR_QP=rot90(MSTSR_QP);
CNN_QP=CNN(:,27);
CNN_QP=rot90(CNN_QP);
IFCNN_QP=IFCNN(:,27);
IFCNN_QP=rot90(IFCNN_QP);
DSAGAN_QP=DSAGAN(:,27);
DSAGAN_QP=rot90(DSAGAN_QP);
EMFusion_QP=EMFusion(:,27);
EMFusion_QP=rot90(EMFusion_QP);
D2FNet_QP=D2FNet(:,27);
D2FNet_QP=rot90(D2FNet_QP);



%% 画图
figure;                       %开启画布
t = tiledlayout(2,3,'TileSpacing','Compact','Padding','Compact');  %代替subplot，可以让图片紧凑，要最小化绘图之间的空间，请将 'TileSpacing' 名称-值对组参数设置为 'compact'。要使布局周围的空间最小化，请将 'Padding' 名称-值对组参数设置为 'compact'。
%SSIM
nexttile
plot(X,D2FNet_SSIM,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;   %这个会自动调整横纵坐标，所以需要用gca来全局覆盖，手动设置
plot(X,NSCT_SSIM,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_SSIM,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_SSIM,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_SSIM,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_SSIM,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_SSIM,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_SSIM,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('SSIM','FontWeight','bold');
legend('D2FNet:0.7589','NSCT:0.7305','LES:0.7292','MSTSR:0.7327','CNN:0.6737','IFCNN:0.7525','DSAGAN:0.4352','EMFusion:0.7557','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');                        %图例与画线的先后顺序一致,这里顺便把均值加上去
set(gca,'YLim',[0.1,1.2]);     
set(gca,'ytick',[0.1:0.1:1.2]);

%CC
nexttile
plot(X,D2FNet_CC,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;
plot(X,NSCT_CC,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_CC,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_CC,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_CC,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_CC,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_CC,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_CC,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('CC','FontWeight','bold');
legend('D2FNet:0.9495','NSCT:0.9319','LES:0.9462','MSTSR:0.9319','CNN:0.9454','IFCNN:0.9505','DSAGAN:0.9463','EMFusion:0.9502','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off'); 
set(gca,'YLim',[0.86,1.04]);
set(gca,'ytick',[0.86:0.02:1.04]);

% PSNR
nexttile
plot(X,D2FNet_PSNR,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;
plot(X,NSCT_PSNR,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_PSNR,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_PSNR,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_PSNR,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_PSNR,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_PSNR,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_PSNR,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('PSNR','FontWeight','bold');
legend('D2FNet:21.3157','NSCT:18.8314','LES:18.9692','MSTSR:19.1436','CNN:19.6602','IFCNN:19.5353','DSAGAN:20.0968','EMFusion:21.8668','Location','north','Orientation','horizontal','NumColumns',3,'Box','off'); 
set(gca,'YLim',[14.0,29.0]);
set(gca,'ytick',[14.0:1.5:29.0]);

% QCB
nexttile
plot(X,D2FNet_QCB,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;
plot(X,NSCT_QCB,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_QCB,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_QCB,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_QCB,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_QCB,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_QCB,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_QCB,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('Q_{CB}','FontWeight','bold');     %matlab字符串支持识别LaTeX语法，这里可以用语法打出一些带上下标的字符和数学符号
legend('D2FNet:0.6864','NSCT:0.6833','LES:0.6533','MSTSR:0.6874','CNN:0.5282','IFCNN:0.6648','DSAGAN:0.3926','EMFusion:0.6615','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');
set(gca,'YLim',[0.2,1.1]);
set(gca,'ytick',[0.2:0.1:1.1]);

%QS
nexttile
plot(X,D2FNet_QS,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;
plot(X,NSCT_QS,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_QS,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_QS,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_QS,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_QS,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_QS,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_QS,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('Q_{S}','FontWeight','bold');     %matlab字符串支持识别LaTeX语法，这里可以用语法打出一些带上下标的字符和数学符号
legend('D2FNet:0.8440','NSCT:0.7850','LES:0.7887','MSTSR:0.7885','CNN:0.7723','IFCNN:0.8341','DSAGAN:0.5194','EMFusion:0.8138','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');
set(gca,'YLim',[0.2,1.2]);
set(gca,'ytick',[0.2:0.1:1.2]);

%QP
nexttile
plot(X,D2FNet_QP,': o r','LineWidth',1.5,'MarkerSize',6,'MarkerFaceColor','r');     %最后一个参数是面填充色
hold on;
plot(X,NSCT_QP,'-. + g','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','g');
plot(X,LES_QP,': * b','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','b');
plot(X,MSTSR_QP,'-. v y','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','y');
plot(X,CNN_QP,': d c','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','c');
plot(X,IFCNN_QP,'-. s m','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','m');
plot(X,DSAGAN_QP,': p k','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','k');
plot(X,EMFusion_QP,'-. h ','color','[0.3 0.8 0.9]','LineWidth',1.5,'MarkerSize',3,'MarkerFaceColor','[0.3 0.8 0.9]');
hold off;
title('Q_{P}','FontWeight','bold');     %matlab字符串支持识别LaTeX语法，这里可以用语法打出一些带上下标的字符和数学符号
legend('D2FNet:0.5304','NSCT:0.4068','LES:0.3821','MSTSR:0.4482','CNN:0.4842','IFCNN:0.4400','DSAGAN:0.3303','EMFusion:0.5343','Location','northeast','Orientation','horizontal','NumColumns',3,'Box','off');
set(gca,'YLim',[0.2,0.9]);
set(gca,'ytick',[0.2:0.1:0.9]); 
 
