%only for RGB image homography
 clc;
clear all;
close all

f = 'scale_light';
ext = 'jpg';
img1 = imread([f '1.' ext]);
img2 = imread([f '2.' ext]);

% H = [    0.9590    0.0137         0
%    -0.0318    0.9686         0
%   145.9796   -4.1725    1.0000];%% jian 12仿射矩阵

H = [0.9549    0.0105         0
   -0.0377    0.9623         0
   73.7390   -1.4277    1.0000];%%scale 12仿射矩阵


 
img2_change = rgb2gray(img2) + 10;
     
tform = maketform('affine',H);
img21 = imtransform(img2,tform);    %%变形后图2

img22 = imtransform(img2_change,tform);%%保证有效区域无0值

[M1 N1 dim1] = size(img1);
[M2 N2 dim2] = size(img2);
% do the mosaic
pt = zeros(4,3);
pt(1,:) = [1 1 1]*H;
pt(2,:) =[N2 1 1]* H;
pt(3,:) = [N2 M2 1]*H;
pt(4,:) = [1 M2 1]*H;%四个顶点位置,从左上顺时针。

k12 = (pt(1,2)-pt(2,2))/(pt(1,1)-pt(2,1));%%重合区域边界斜率
k23 = (pt(3,2)-pt(2,2))/(pt(3,1)-pt(2,1));
k34 = (pt(4,2)-pt(3,2))/(pt(4,1)-pt(3,1));
k14 = (pt(1,2)-pt(4,2))/(pt(1,1)-pt(4,2));


gray1= rgb2gray(img1);
gray2= rgb2gray(img21);

% figure,
% imshow(gray2);

edge1 = edge(gray1,'canny');%图1canny算法检测到的边界
edge2 = edge(gray2,'canny');%图2仿射变换后canny算法检测到的边界

% figure,
% imshow(edge2);

double_gray1 = double(gray1);
double_gray2 = double(gray2);



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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%补偿%%%%%%%%%%%%%%%%%%%%%%%%%%%
img1(:,:,1) = img1(:,:,1) - 1.32;
img1(:,:,2) = img1(:,:,2) + 0.5;
img1(:,:,3) = img1(:,:,3) + 2.2;
img1 = uint8(img1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[M3 N3 dim3] = size(img21);
imgout(up:up+M3-1,left:left+N3-1,:) = img21;
imgout(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;

imgout = uint8(imgout);

figure,imshow(imgout), title('直接融合');