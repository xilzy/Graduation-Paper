function [img_YCbCr] = rgb2YCbCr(img)
[m,n,dim]=size(img);
img=double(img); %转换成浮点型，要进行矩阵乘法，不能是完全的整型矩阵，所以要转换
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
matrix=[0.257,0.564,0.098;
        -0.148,-0.291,-0.439;
        0.439,-0.368,-0.071];
for i=1:m
   for j=1:n 
        tmp=matrix*[R(i,j),G(i,j),B(i,j)]'; %这里有一个转置操作，右上角
        Y(i,j)=tmp(1)+16;
        Cb(i,j)=tmp(2)+128;
        Cr(i,j)=tmp(3)+128;
   end
end
img_YCbCr(:,:,1)=Y;
img_YCbCr(:,:,2)=Cb;
img_YCbCr(:,:,3)=Cr;
end
