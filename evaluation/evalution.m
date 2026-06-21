function [evals] = evalution(imgA, imgB, imgF)
% %% 采用多种评价指标对融合效果进行评估
% evals(1) = Edge_Intensity(imgF); % 边缘强度 EI
% evals(2) = MySF(imgF);    % 空间频率【x,y】 SF2
% evals(3) = entropy(imgF); % EN
% evals(4) = analysis_ssim(imgA,imgB,imgF); %ssim
% evals(5) = analysis_MI(imgA,imgB,imgF);%MI
% evals(6) = std2(imgF);       % SD   matlab内置函数矩阵元素的标准差 
% evals(7) = Definition(imgF); % 清晰度 DF
% evals(8) = avg_gradient(imgF);    %平均梯度 AG
% evals(9) = my_cc(imgA, imgB, imgF);     % CC
% evals(10) = VIFF_Public(imgA, imgB, imgF);  %VIFF/VIF
% evals(11) = (psnr(imgA,imgF)+psnr(imgB,imgF))/2;    % 峰值信噪比 PSNR(要输入两个图像，这里不知道输入啥)
% evals(12) = analysis_Qabf(imgA,imgB,imgF); %Qabf
% evals(13) = metricWang(imgA, imgB, double(imgF));  % QNCIE
% evals(14) = metricChenBlum(imgA, imgB, imgF);      % QCB
% evals(15) = metricChen(imgA, imgB, imgF);      % QCV
% evals(16) = OverallCrossEntropy(imgA, imgB, imgF); % 总体平均交叉熵（OCE）
% evals(17) = analysis_SCD(imgA,imgB,imgF); %SCD
% evals(18) = analysis_fmi(imgA,imgB,imgF,'wavelet');%FMI_w
% evals(19) = analysis_fmi(imgA,imgB,imgF,'dct');%FMI_dct
% evals(20) = analysis_MSSSIM(imgA,imgB,imgF); %MSSSIM
% evals(21) = analysis_fmi(imgA,imgB,imgF);%FMI_pixel
% evals(22) = metricZheng(imgA, imgB,imgF); % QSF
% evals(23) = metricMI(imgA,imgB,imgF,1);   % QMI
% evals(24) = metricPeilla(imgA, imgB, imgF, 1);      % QS
% evals(25) = analysis_nabf(imgF,imgA,imgB); %Nabf
% evals(26) = metricXydeas(imgA, imgB, imgF);      % QG
% evals(27) = metricZhao(imgA, imgB, imgF);      % QP
% evals(28) = metricPeilla(imgA, imgB, imgF, 2);      % QW
% evals(29) = metricPeilla(imgA, imgB, imgF, 3);      % QE


% %最终选定的六个
% evals(1) = analysis_ssim(imgA,imgB,imgF); %ssim
% evals(2) = my_cc(imgA, imgB, imgF);     % CC
% evals(3) = (psnr(imgA,imgF)+psnr(imgB,imgF))/2;    % 峰值信噪比 PSNR(要输入两个图像，这里不知道输入啥)
% evals(4) = metricChenBlum(imgA, imgB, imgF);      % QCB
% evals(5) = metricPeilla(imgA, imgB, imgF, 1);      % QS
% evals(6) = metricZhao(imgA, imgB, imgF);      % QP


% %最终选定的10个
evals(1) = Edge_Intensity(imgF); % 边缘强度 EI
evals(2) = MySF(imgF);    % 空间频率【x,y】 SF2
evals(3) = entropy(imgF); % EN
evals(4) = analysis_MI(imgA,imgB,imgF);%MI
evals(5) = std2(imgF);       % SD   matlab内置函数矩阵元素的标准差 
evals(6) = Definition(imgF); % 清晰度 DF
evals(7) = avg_gradient(imgF);    %平均梯度 AG
evals(8) = my_cc(imgA, imgB, imgF);     % CC
evals(9) = VIFF_Public(imgA, imgB, imgF);  %VIFF/VIF
evals(10) = analysis_nabf(imgF,imgA,imgB); %Nabf

%重复指标，或者跑不起的
% evals(26) = MIabf(imgA, imgB, uint8(imgF)); % MI2]
% evals(24) = Qabf(imgA, imgB, imgF); 				% Q^{AB/F}
% evals(25)= metricYang(imgA, imgB, imgF);      % QY
% evals(26) = metricCvejic(imgA, imgB, imgF,2);      % QC
%evals(23)  = (vifvec(im2double(imgA),im2double(imgF))+vifvec(im2double(imgB),im2double(imgF)))/2;%vif




end
