%only for RGB image homography
 clc;
clear all;
close all

tic;
f = 'jian';
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

img1Dup=double(img1Dup);
img2Dup=double(img2Dup);

% use Harris in both images to find corner.
[locs1] =Harrisv2(img1Dup);
[locs2] = Harrisv2(img2Dup);

% % % use sift in both images to find corner.
% [image1, descriptors1, locs1] = sift(img1Dup);
% [image2, descriptors2, locs2] = sift(img1Dup);

%using NCC to find coorespondence between two images
[matchLoc1, matchLoc2] =  findCorr(img1Dup,img2Dup,locs1, locs2);

% use RANSAC to find homography matrix
[H, inlierIdx] = findHomography1(matchLoc1',matchLoc2');
 H  %#ok
tform = maketform('affine',H);
img21 = imtransform(img2,tform); % reproject img2
%figure(3),imshow(img21)
% adjust color or grayscale linearly, using corresponding infomation
[M1, N1, dim] = size(img1);
[M2, N2, dim2] = size(img2);
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

[M3, N3, dim3] = size(img21);
imgout(up:up+M3-1,left:left+N3-1,:) = img21;
imgout(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;     

figure,imshow(uint8(imgout));
toc