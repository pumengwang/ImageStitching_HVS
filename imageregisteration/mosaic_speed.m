%only for RGB image homography
 clc;
clear all;
close all

tic;

f = 'lena';
ext = 'jpg';
img1 = imread([f '1.' ext]);
%img1=img1(20:end-20,20:end-20,:);
img2 = imread([f '2.' ext]);
%img2=img2(20:end-20,20:end-20,:);
if size(img1,3)==1%to find whether input is RGB image
fprintf('error,only for RGB images\n');
end

img1Dup=rgb2gray(img1);%duplicate img1
%  img1Dup=img1Dup(20:end-20,20:end-20);
img2Dup=rgb2gray(img2);%duplicate img2
%  img2Dup=img2Dup(20:end-20,20:end-20);


edge1 = edge(img1Dup,'log');%图1canny算法检测到的边界
edge2 = edge(img2Dup,'log');%图2仿射变换后canny算法检测到的边界

img1Dup=double(img1Dup);
img2Dup=double(img2Dup);

% use Harris in both images to find corner.

[locs1] =Harrisv2(img1Dup);
[locs2] = Harrisv2(img2Dup);


%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%%%delete useless corner%%%

num_corn1 = size([locs1],1);
num_corn2 = size([locs2],1);

num_corn = min(num_corn1,num_corn2);
%%%%%%%%%%分割%%%%%
%对检测到的边界进行膨胀
se1 = strel('disk',4);
% 创建6*6的正方形
se2 = strel('line',10,45);
% 创建直线长度10，角度45
se3 = strel('disk',7);
% 创建圆盘半径
se4 = strel('disk',5);
% 创建椭圆体，半径15，高度5
%%%
% I12 = imdilate(edge1,se1);
% I13 = imdilate(edge1,se2);
I14 = imdilate(edge1,se3);

%腐蚀
area11 = imerode(I14,se3);%%%%腐蚀轮廓已对齐

area11 = imerode(area11,se4);
area12 = imerode(area11,se1);
area13 = imerode(area12,se2);%%%%%腐蚀的越多删掉的角点越少。速度提升越不明显
% area4 = imerode(area3,se2);

%figure,imshow(add);

%edge_add1 = add1+edge1;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界


%%%
% I2 = imdilate(edge2,se1);
% I3 = imdilate(edge2,se2);
I4 = imdilate(edge2,se3);
%I5 = imdilate(edge,se4);

%腐蚀
area1 = imerode(I4,se3);

area1 = imerode(area1,se4);
area2 = imerode(area1,se1);
area3 = imerode(area2,se2);

% area4 = imerode(area3,se2);

%%%%%%%
if num_corn <500
    uarea1 = area13;
    uarea2 = area3;
elseif num_corn >= 500 && num_corn < 1000
    uarea1 = area12;
    uarea2 = area2;
elseif num_corn >= 1000
    uarea1 = area11;
    uarea2 = area1;
end

add1 = (I14+uarea1)/2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域
add2 = (I4+uarea2)/2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域
%figure,imshow(add);

%edge_add2 = add2+edge2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界
% 
% figure,imshow(edge2);
% figure,imshow(I4);
% figure,imshow(area4);


%%%%%删除useless角点%%%
[row,col] = find(add1 == 1);
useless_area = [row,col];

[locs1] = setdiff(locs1,useless_area,'rows');

[row,col] = find(add2 == 1);
useless_area = [row,col];

[locs2] = setdiff(locs2,useless_area,'rows');



A=locs1;
           img1Dup=uint8(img1Dup);
           figure,imshow(img1Dup);
           hold on
           plot(A(1:end,2),A(1:end,1),'r+');
           
B=locs2;
           img2Dup=uint8(img2Dup);
           figure,imshow(img2Dup);
           hold on
           plot(B(1:end,2),B(1:end,1),'r+');
           
%figure,imshow(edge_add1),title('区域分割');


toc
% runtime = toc



%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%using NCC to find coorespondence between two images
[matchLoc1 matchLoc2] =  findCorr(img1Dup,img2Dup,locs1, locs2);

% use RANSAC to find homography matrix
[H inlierIdx] = findHomography1(matchLoc1',matchLoc2');
 H  %#ok
tform = maketform('affine',H);
img21 = imtransform(img2,tform); % reproject img2
%figure(3),imshow(img21)
% adjust color or grayscale linearly, using corresponding infomation
[M1 N1 dim] = size(img1);
[M2 N2 dim2] = size(img2);
% do the mosaic
pt = zeros(3,3);
pt(1,:) = [1 1 1]*H;
pt(2,:) =[N2 1 1]* H;
pt(3,:) = [N2 M2 1]*H;
pt(4,:) = [1 M2 1]*H;%上下左右四个顶点位置。
x2 = pt(:,1)./pt(:,3);
y2 = pt(:,2)./pt(:,3);


up = round(min(y2));
Yoffset = 0;
if up <= 0
	Yoffset = -up+1;
	up = 1;
end

left = round(min(x2));
Xoffset = 0;
if left<=0
	Xoffset = -left+1;
	left = 1;
end

[M3 N3 dim3] = size(img21);
imgout(up:up+M3-1,left:left+N3-1,:) = img21;
imgout(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;     

figure,imshow(uint8(imgout));

toc;
runtime = toc