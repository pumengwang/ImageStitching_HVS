%only for RGB image homography
 clc;
clear all;
close all

f = 'xiao';
ext = 'jpg';
img1 = imread([f '1.' ext]);
img2 = imread([f '2.' ext]);

H = [     0.9494    0.0043         0
         -0.0379    0.9681         0
         263.5857   -6.3415    1.0000];%%仿射矩阵

 
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

[M3 N3 dim3] = size(img21);
imgout(up:up+M3-1,left:left+N3-1,:) = img21;
imgout(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;

imgout = uint8(imgout);

figure,imshow(imgout), title('直接融合');

gray_imgout = rgb2gray(imgout);

%%求重合区域坐标

[m,n] = size(gray_imgout);%%最后拼接图像的大小

ref_img1 = zeros(m,n);
ref_img21 = zeros(m,n);

up_o = round(min(y2));
left_o = round(min(x2));


ref_img1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1) = 1;%%图1对应的坐标为1

img22(find(img22>0)) = 1;

ref_img21(up:up+M3-1,left:left+N3-1,:) = img22;

% for i = 1:M3
%     for j = 1:N3
%         if img22 > 0
%             ref_img21(up+i-1,left+j-1) = 1;
%         end
% 
%     end
% end


ref_overlap = ref_img1 + ref_img21;%%%参考图像，2代表重合区域，1代表有像素的区域
ref_overlap = uint8(ref_overlap);%%%%%%%%%%%参考图像%%%%%%%%%%%%%%%%
figure,imshow(ref_overlap),title('重合区域坐标参考');





%%%%%%%%%%分割%%%%%
%对检测到的边界进行膨胀
se1 = strel('square',6);
% 创建6*6的正方形
se2 = strel('line',10,45);
% 创建直线长度10，角度45
se3 = strel('disk',7);
% 创建圆盘半径
se4 = strel('ball',15,5);
% 创建椭圆体，半径15，高度5

I2 = imdilate(edge1,se1);
I3 = imdilate(edge1,se2);
I4 = imdilate(edge1,se3);
%I5 = imdilate(edge,se4);

%腐蚀
area1 = imerode(I4,se3);
area2 = imerode(area1,se3);
area3 = imerode(area2,se2);

add = (I4+area3)/2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域
%figure,imshow(add);

edge_add = add+edge1;
figure,imshow(edge_add),title('区域分割');%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界
%%%以上对图1完成了分割%%%

%%%%%%%%%%%%%%%%%%%%%%%%掩蔽特性%%%%%%%%%%%%%%%%%%%
%%此步前可做高斯滤波处理灰度图像
jicha = rangefilt(gray1);%%局部极差，极差越小代表越平滑
jicha = double(jicha);
%biaozhuncha = stdfilt(gray); %局部标准差

%熵
entropy_map = Normalize(entropyfilt(gray1));
% entropy_map = uint8(entropy_map);
%

%%在edge_add图中，平滑区域减去k1倍的极差值，混乱区域加上k2倍的熵值
k1 = 1;
k2 = 0.8;
masking_img1 = zeros(M1,N1);
masking_img1 = double(masking_img1);

for i = 1:M1
    for j = 1:N1
        switch(edge_add(i,j))
            case(0)
                masking_img1(i,j) = k1*(255 -jicha(i,j));
            case(1)
                masking_img1(i,j) = k2*entropy_map(i,j);
            case(2)
                masking_img1(i,j) = k2*entropy_map(i,j);
                                
            otherwise
                masking_img1(i,j) = 0;
        end
            
    end
 
end

masking_img1 = Normalize(masking_img1);
masking_img1 = uint8(masking_img1);

for i = 1:M1
    for j =1:N1
        if edge_add(i,j) == 0.5 || edge_add(i,j) == 1.5
            masking_img1(i,j) = 0;
            
        end
    
    end 
end    

masking = zeros(m,n);

masking(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1) = masking_img1;
masking = uint8(masking);%%%%%%%%%%%%%%%%%掩蔽特性%%%%%%%%%%%%%%%%

figure,
imshow(masking),title('掩蔽特性');




%%%%%%%%%%%%%%%%%%求其余映射图%%%%%%%%%%%%%%%%


%%%%%%%%%%%非线性映射图求解%%%%
average = mean(gray1(:));

nonlinear = log(double_gray1+1)+log(average);%%log（average）是公式S = K*LnL + K0中的K0，与图像的平均亮度有关
Nor_nonlinear = uint8(Normalize(nonlinear));%%非线性映射图
%nonlinear = Normalize(nonlinear);

Nor_nonlinear_map = zeros(m,n);

Nor_nonlinear_map(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1) = Nor_nonlinear;
Nor_nonlinear_map = uint8(Nor_nonlinear_map);
figure,
imshow(Nor_nonlinear_map),title('灰度对数归一化');



%%%%%%%%显著性映射图%%%%%%%%%%%

saliency_map1 = SDSP(img1);
saliency_map2 = SDSP(img21);

% saliency_map = uint8(0.5*saliency_map1 + 0.5*saliency_map2);
% figure,
% imshow(saliency_map2),title('显著性');

ref_saliency1 = zeros(m,n);
ref_saliency2 = zeros(m,n);


ref_saliency1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1) = saliency_map1;
ref_saliency2(up:up+M3-1,left:left+N3-1) = saliency_map2;

saliency = ref_saliency1 + ref_saliency2;

saliency(find(ref_overlap == 0)) = 0;
saliency(find(ref_overlap == 1)) = 100;
saliency(find(ref_overlap == 2)) = 0.5*saliency(find(ref_overlap == 2));
%%灰色表示原图位置，黑色无数据
saliency = uint8(saliency); %%%%%%%%%%%%%%%%%掩蔽特性%%%%%%%%%%%%%%%%
figure,
imshow(saliency),title('全图显著性');


%%%%%%%%%%%%%%%%%缝合路径权值图%%%%%%%%%%%%%%%%

cost_map = 0.75*(255-saliency) + 0.15*masking + 0.1*Nor_nonlinear_map;

cost_map = uint8(cost_map);
figure,
imshow(cost_map),title('权值图');


%%%%%%%%%%%%%%%%%边缘信息参考%%%%%%%%%%%%%%%%

ref_edge1 = zeros(m,n);
ref_edge2 = zeros(m,n);


ref_edge1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = edge1;
ref_edge2(up:up+M3-1,left:left+N3-1,:) = edge2;

%%若重合边界过少，可以做很小范围的膨胀
ref_edge = ref_edge1 + ref_edge2;

ref_edge = uint8(ref_edge);%%2代表重合的边界，可以穿过； 1表示不重合的边界
figure,
imshow(ref_edge*100),title('边缘信息参考');

figure,
subplot(121);imshow(cost_map);title('权值图');
subplot(122);imshow(ref_edge*100);title('边缘信息参考');

%%edge_add中0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界