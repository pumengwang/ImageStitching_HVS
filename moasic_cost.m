%only for RGB image homography
 clc;
clear all;
close all

f = 'jian';
ext = 'jpg';
img1 = imread([f '1.' ext]);
img2 = imread([f '2.' ext]);


% H = [    1.0354    0.0241         0
%    -0.0058    1.0538         0
%    40.9300  -55.5326    1.0000];%%%tu1
% H = [    0.9973   -0.0116         0
%     0.0229    0.9754         0
%    50.0256  -24.0130    1.0000];%%%dh原始矩阵
% 
% H = [    0.9973   -0.0116         0
%     0.0229    0.9854         0
%    50.0256  -26.0130    1.0000];%%%dh改参数
%  H = [ 1.0395   -0.0026         0
%    -0.0340    1.0181         0
%   126.4352  -38.7236    1.0000];%%%hg

%  H = [ 0.9856   -0.0296         0
%     0.0323    0.9988         0
%    12.8007  -36.2664    1.0000];%%%9lou
% 
%  H = [ 1.1760    0.0155         0
%     0.0200    1.1694         0
%    12.7816  -66.2034    1.0000];%%%build
% H = [ 1.0687    0.0112         0
%    -0.0180    1.0826         0
%    90.7917  -27.4238    1.0000];%%%dl
% H = [    0.9932    0.0002         0
%     0.0197    1.0146         0
%    48.1505  -29.0500    1.0000];%%%tj
% H = [   0.9489    0.0039         0
%    -0.0358    0.9666         0
%   263.3049   -5.7935    1.0000];%%%xiao

% H = [ 0.9853         0         0
%     0.0163    0.9880         0
%    70.0556   -8.0000    1.0000];%%%hu

% H = [ 0.9853         0         0
%     0.0163    1.0000         0
%    70.0556   -8.0000    1.0000];%%%hu_yuanshi
% 
H = [    0.9590    0.0137         0
   -0.0318    0.9750         0
  145.9796   -6.8725    1.0000];%% jian 12修改仿射矩阵
% H = [    0.9590    0.0137         0
%    -0.0318    0.9686         0
%   145.9796   -4.1725    1.0000];%% jian 12原始仿射矩阵
% H = [0.9549    0.0105         0
%    -0.0377    0.9623         0
%    73.7390   -1.4277    1.0000];%%scale 12仿射矩阵

% H = [ 1.0140    0.0074         0
%    -0.0295    1.0255         0
%    49.9593  -84.7181    1.0000];%%%dweihe原始参数
% H = [ 1.0140    0.0074         0
%    -0.0295    1.0300         0
%    52.5593  -85.7181    1.0000];%%%dweihe修改参数

 
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

ref_overlap = ref_img1 + ref_img21;%%%参考图像，2代表重合区域，1代表有像素的区域
ref_overlap = uint8(ref_overlap);%%%%%%%%%%%参考图像%%%%%%%%%%%%%%%%
%figure,imshow(ref_overlap),title('重合区域坐标参考');

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

edge_add = add+edge1;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界

%%figure,imshow(edge_add),title('区域分割');
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

%%在edge_add图中，平滑区域减去k1倍的极差值，混乱区域加上k2倍的熵值(论文中将k1，k2颠倒了)
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
            case(1.5)
                masking_img1(i,j) = -20; %%%%强边界给更小的权值               
            otherwise
                masking_img1(i,j) = 0;
        end
            
    end
 
end

masking_img1 = Normalize(masking_img1+30);
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

% figure,
% imshow(masking),title('掩蔽特性');




%%%%%%%%%%%%%%%%%%求其余映射图%%%%%%%%%%%%%%%%

%%%%%%%%%亮度差异映射图%%%%%%%%
 gray_light1= zeros(m,n);
 gray_light2= zeros(m,n);
gray_light1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = gray1;
gray_light2(up:up+M3-1,left:left+N3-1,:) = gray2;

%%均值滤波求平均亮度
gray_light1 = double(filter2(fspecial('average',3),gray_light1));
gray_light2 = double(filter2(fspecial('average',3),gray_light2));
% gray_light1 = medfilt2(double(gray_light1));
% gray_light2 = medfilt2(double(gray_light2));

light_difference = gray_light1 - gray_light2;
light_difference(find(ref_overlap == 1)) = 0;

light_difference = abs(light_difference );
% %%%%改动
light_difference = Normalize(light_difference + 10);
% %%%

%light_difference = Normalize(light_difference + 1);
light_difference = 255 - light_difference;
%light_difference_overlap = light_difference(find(ref_overlap == 2));
light_difference(find(ref_overlap == 1)) = 0;
light_difference(find(ref_overlap == 0)) = 0;
light_difference = uint8(light_difference);%%%%%light_difference  值越大，差异越小

%%%%%%%%%%%非线性映射图求解%%%%
average = mean(gray1(:));

nonlinear = log(double_gray1+1)+log(average);%%log（average）是公式S = K*LnL + K0中的K0，与图像的平均亮度有关
Nor_nonlinear = uint8(Normalize(nonlinear));%%非线性映射图
%nonlinear = Normalize(nonlinear);

Nor_nonlinear_map = zeros(m,n);

Nor_nonlinear_map(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1) = Nor_nonlinear;
Nor_nonlinear_map = uint8(Nor_nonlinear_map);
% figure,
% imshow(Nor_nonlinear_map),title('灰度对数归一化');



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
% figure,
% imshow(saliency),title('全图显著性');

%%%
masking(find(ref_overlap == 0)) =0;
masking(find(ref_overlap == 1)) =0;%%不重合区域为0

Nor_nonlinear_map(find(ref_overlap == 0)) =0;
Nor_nonlinear_map(find(ref_overlap == 1)) =0;%%不重合区域为0

light_difference(find(ref_overlap == 0)) =0;
light_difference(find(ref_overlap == 1)) =0;%%不重合区域为0


imwrite(Nor_nonlinear_map,[f '_VN.' ext]);
imwrite(light_difference,[f '_LD.' ext]);
imwrite(masking,[f '_VM.' ext]);
imwrite(saliency,[f '_VS.' ext]);

%%%%%%%%%%%%%%%%%缝合路径权值图%%%%%%%%%%%%%%%%
cost_map = 0.53*(255-saliency) + 0.17*masking + 0.1*Nor_nonlinear_map + 0.2*light_difference;%%（公式1）
%cost_map = 0.5*(255-saliency) + 0.2*masking + 0.1*Nor_nonlinear_map + 0.2*light_difference;%%（公式2）
%cost_map = 0.55*(255-saliency) + 0.25*masking + 0.1*Nor_nonlinear_map + 0.1*light_difference;%%（公式3）
%cost_map = 255-saliency;%% VS
%cost_map = masking;%%VM
%cost_map = Nor_nonlinear_map;%%NL
%cost_map =light_difference;%%LD



cost_map = uint8(cost_map);

cost_map(find(ref_overlap == 0)) =0;
cost_map(find(ref_overlap == 1)) =0;%%不重合区域为0


imwrite(cost_map,[f '_Weight_map.' ext]);
figure,
imshow(cost_map),title('权值图');



%%%%%%%%%%%%%%%%%边缘信息参考%%%%%%%%%%%%%%%%

ref_edge1 = zeros(m,n);
ref_edge2 = zeros(m,n);


ref_edge1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = edge1;
ref_edge2(up:up+M3-1,left:left+N3-1,:) = edge2;

%%若重合边界过少，可以做很小范围的膨胀
% edge_pengzhang = strel('disk',1);
% ref_edge1 = imdilate(ref_edge1,edge_pengzhang);%%%peng zhang
%%%%%%%

ref_edge = ref_edge1 + ref_edge2;

ref_edge = uint8(ref_edge);%%2代表重合的边界，可以穿过； 1表示不重合的边界

ref_edge(find(ref_overlap == 0)) =0;
ref_edge(find(ref_overlap == 1)) =0;%%不重合区域为0
% figure,
% imshow(ref_edge*100),title('边缘信息参考');

% figure,
% subplot(121);imshow(cost_map);title('权值图');
% subplot(122);imshow(ref_edge*100);title('边缘信息参考');

%%edge_add中0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界
%%ref_overlap参考图像，2代表重合区域，1代表有像素的区域
%%cost_map权值图，值越大，权值越大
%%ref_edge边界参考图，2代表重合的边界，可以穿过； 1表示不重合的边界



%%求起始点，即图2旋转后与图1边界的交点坐标
%%拼接图像中，图1 的边界（Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1）

cross_up = min(find(ref_overlap(Yoffset+1,:)==2));%%(Yoffset+1,cross_up)    上
cross_down = min(find(ref_overlap(Yoffset+M1,:)==2));%%(Yoffset+M1,cross_down)下
cross_left = max(find(ref_overlap(:,Xoffset+1)==2));%%(cross_left,Xoffset+1)    左
cross_right = max(find(ref_overlap(:,Xoffset+N1)==2));%%(cross_right,Xoffset+N1)右

%%%
if isempty(cross_up) == 0   %%上部相交
    begin = [Yoffset+1,cross_up];
else
    begin = [Yoffset+M1,cross_down];
end

%%%
if isempty(cross_right) == 0   %%右部相交
    terminal = [cross_right,Xoffset+N1];
else
    terminal = [cross_left,Xoffset+1];
end

%%begin起点，terminal终点
step = terminal - begin;%%移动的步数（y_step,x_step）

moasic_line = imgout;%%在此图上显示融合路径点


%%%%%%求图2旋转后的强弱边界
I2_img21 = imdilate(edge2,se1);
I3_img21 = imdilate(edge2,se2);
I4_img21 = imdilate(edge2,se3);
%I5 = imdilate(edge,se4);

%腐蚀
area1_img21 = imerode(I4_img21,se3);
area2_img21 = imerode(area1_img21,se3);
area3_img21 = imerode(area2_img21,se2);

add_img21 = (I4_img21+area3_img21)/2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域
%figure,imshow(add);

edge_add2 = add_img21+edge2;%%0代表平滑区域，1代表混乱区域，0.5为包含强边界的区域，1.5为该区域中的强边界,2为弱边界


edge_strong1 = zeros(m,n);
edge_strong2 = zeros(m,n);


edge_strong1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = edge_add;
edge_strong2(up:up+M3-1,left:left+N3-1,:) = edge_add2;%%%1.5为该区域中的强边界,2为弱边界

edge_strong1(find(ref_overlap == 0)) =0;
edge_strong1(find(ref_overlap == 1)) =0;%%不重合区域为0


edge_strong2(find(ref_overlap == 0)) =0;
edge_strong2(find(ref_overlap == 1)) =0;%%不重合区域为0

% figure,imshow(edge_strong1),title('1');
% figure,imshow(edge_strong2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%不考虑点集%%%%%%%%%%%%%%%%%%%%%%

%%%不重合区域坐标集合%%%%
[row,col] = find(ref_overlap ~= 2);
misalignment_area = [row,col];

%%%重合区域边界坐标%%%
edge_overlap = edge(ref_overlap);
[row,col] = find(edge_overlap == 1);
boundary_overlap = [row,col];

%%%不重合区域坐标集合%%%%
[row1,col1] = find(edge_strong1 == 1.5);
[row2,col2] = find(edge_strong2 == 1.5);
strong_edge1 = [row1,col1];
strong_edge2 = [row2,col2];



strong_edge= union(strong_edge1,strong_edge2,'rows');%%%重合区域两图强边界的并集，每行为像素坐标
%%%%%%
[row,col] = find(ref_edge == 2);
safe_edge = [row,col];%%%重合边界的坐标集，包含纹理区（弱边界）


safe_strongedge = intersect(strong_edge,safe_edge,'rows');%%%%安全的强边界坐标
 
[row,col] = find(ref_edge == 1);
mis_edge = [row,col];%%%不重合边界的坐标集，包含纹理区（弱边界）

misalignment_strongedge =  intersect(strong_edge,mis_edge,'rows'); %%%%不重合的强边界坐标
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%此处待加小范围的强边界区域%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%不考虑像素点初始化坐标%%%%%%
un_candients = union(misalignment_area,misalignment_strongedge,'rows');
un_candients = union(un_candients,boundary_overlap,'rows');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%选择拼接点%%%%%%%%%%%
%%%%%%%%权值图cost_map
%%%%%%%%起点begin
%%%%%%%%终点terminal
%%%%%%%%位移量step
%%%%%%%%不考虑点集un_candients
%%%%%%%%安全强边界safe_strongedge
%%%%%%%%显示融合路径点图像moasic_line

a = ones(3,3);
i1 = begin(1,1);
j1 = begin(1,2);

pos_moasic_points = zeros(m,n);

un_candients = [un_candients;[i1,j1]];
moasic_points = begin;
imax = max(terminal(1,1),up+M3-1);
jmax = max(terminal(1,2),left+N3-1);

while(i1 <=imax && j1 <= jmax) %%%%%条件需要完善，到达边界后程序停止

%         dis_x = terminal(1:1)-i1;
%         dis_y = terminal(1:2)-j1;
%         window = [cost_map(i1-1,j1-1)/(dis_x),  cost_map(i1-1,j1),  cost_map(i1-1,j1+1);
%                   cost_map(i1,j1-1),    cost_map(i1,j1),    cost_map(i1,j1+1);
%                   cost_map(i1+1,j1-1),  cost_map(i1+1,j1),  cost_map(i1+1,j1+1)];  

        pos_moasic_points(i1,j1) = 1; 
        
        window = [cost_map(i1-1,j1-1),  cost_map(i1-1,j1),  cost_map(i1-1,j1+1);
                  cost_map(i1,j1-1),    cost_map(i1,j1),    cost_map(i1,j1+1);
                  cost_map(i1+1,j1-1),  cost_map(i1+1,j1),  cost_map(i1+1,j1+1)];
        pos_window = [i1-1,j1-1; i1-1,j1; i1-1,j1+1;  i1,j1-1; i1,j1; i1,j1+1; i1+1,j1-1; i1+1,j1; i1+1,j1+1];
        if ismember(terminal,pos_window,'rows')== 1
%             i1 = terminal(1,1);
%             j1 = terminal(1,2);
            moasic_points = [moasic_points; terminal];
             break
        else
        ref_candients = ismember(pos_window,un_candients,'rows');
        
        ref_window = uint8(a-reshape(ref_candients,3,3)');%%%与window矩阵对应，该矩阵为0代表不考虑该坐标
        window = double(window.*ref_window);%%将窗口内不考虑的点权值置1
        
        window_length = [ norm([i1-1,j1-1]-terminal),  norm([i1-1,j1]-terminal),   norm([i1-1,j1+1]-terminal);
                          norm([i1,j1-1]-terminal),    norm([i1,j1]-terminal),     norm([i1,j1+1]-terminal);
                          norm([i1+1,j1-1]-terminal),  norm([i1+1,j1]-terminal),   norm([i1+1,j1+1]-terminal)];
        ref_length = 1./(window_length); %%距离对数的倒数   距离远时影响较小，近时影响较大   

        window = ref_length + window;%%%cost+距离对数的倒数
        [i2,j2] = find(window == max(max(window)), 1, 'last');
        h = i2 - 2;%%hang
        l = j2 - 2;%%lie
        i1 = i1 + h;
        j1 = j1 + l;
        moasic_points = [moasic_points; [i1,j1]];%%%
        %%un_candients = [un_candients;[i1,j1];[i1-1,j1-1];[i1,j1-1];[i1+1,j1]];%%窗口上侧及中心点加入非候选点
        
        un_candients = [un_candients;[i1,j1];[i1-h,j1-l];[i1-h-1,j1-l];[i1-h+1,j1-l];[i1-h,j1-l-1];[i1-h,j1-l+1];];%%根据选择点的方向得到十字非候选点
        
        moasic_line(i1,j1,:) = 255;
        end
        
end

figure,imshow(moasic_line);
%figure,imshow(pos_moasic_points);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disk1 = strel('disk',2);
disk2 = strel('disk',1);
%%膨胀
pos_moasic_points = imdilate(pos_moasic_points,disk1);
%%腐蚀
pos_moasic_points = imerode(pos_moasic_points,disk1);%%融合路径点得到的区域
%pos_moasic_points = imerode(pos_moasic_points,disk2);
edge_pos_moasic_points = edge(pos_moasic_points,'canny');
edge_pos_moasic_points = [edge_pos_moasic_points(:,end) edge_pos_moasic_points(:,1:end-1)];%%融合路径
% imshow(edge_pos_moasic_points)

%%%%
im = pos_moasic_points;
im(1:begin(1,1)-1,:) = 1;%%%染白一些区域
im(:,terminal(1,2)) = 1;
im2=imfill(im,'holes');             %填充
%im3=bwperim(im2);                   %轮廓提取
im2 = [im2(:,end) im2(:,1:end-1)];

im2(:,terminal(1,2):end) = 1;


figure,imshow(im2); title('重合区域划分')             %显示%%0表示图1， 1表示图2


%%%%%%%%%%%%%%%%%%%%%根据im2计算拼接图%%%%%%%%%%%%%%%%%%%%

ref_img1_moasic = im2(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1);
ref_img2_moasic = im2(up:up+M3-1,left:left+N3-1);
% figure,imshow(ref_img1_moasic);
% figure,imshow(ref_img2_moasic);


%%%
img_moasic1 = zeros(m,n,3);
img_moasic2 = zeros(m,n,3);
img_moasic1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;
img_moasic2(up:up+M3-1,left:left+N3-1,:) = img21;
%%%%%
% im2(:,:,1) = im2;
% im2(:,:,2) = im2;
% im2(:,:,3) = im2;
img_moasic_direct1(:,:,1) = img_moasic1(:,:,1).*~im2;
img_moasic_direct1(:,:,2) = img_moasic1(:,:,2).*~im2;
img_moasic_direct1(:,:,3) = img_moasic1(:,:,3).*~im2;
img_moasic_direct1 = uint8(img_moasic_direct1);
%%%
img_moasic_direct2(:,:,1) = img_moasic2(:,:,1).*im2;
img_moasic_direct2(:,:,2) = img_moasic2(:,:,2).*im2;
img_moasic_direct2(:,:,3) = img_moasic2(:,:,3).*im2;
img_moasic_direct2 = uint8(img_moasic_direct2);


img_moasic_direct = img_moasic_direct1 +img_moasic_direct2;
figure,imshow(img_moasic_direct),title('未加多尺度融合');


%%%%%显示融合线%%%%%%
img_moasic_line = img_moasic_direct;
seam_line = edge(im2,'canny');
seam_line(find(ref_overlap == 0)) = 0;
seam_line(find(ref_overlap == 1)) = 0;%%%%seam_line 二值图像，1表示拼接线

img_moasic_line_r = img_moasic_line(:,:,1);
img_moasic_line_r(find(seam_line == 1)) = 200;

img_moasic_line_g = img_moasic_line(:,:,2);
img_moasic_line_g(find(seam_line == 1)) = 0;

img_moasic_line_b = img_moasic_line(:,:,3);
img_moasic_line_b(find(seam_line == 1)) = 0;

img_moasic_line(:,:,1) = img_moasic_line_r;
img_moasic_line(:,:,2) = img_moasic_line_g;
img_moasic_line(:,:,3) = img_moasic_line_b;

figure,imshow(img_moasic_line),title('seam line');%%%%%显示融合路径
% 

%%%%%%%以下为多尺度融合%%%%%%%
% im1 = imread('huaji.jpg');  im1 = double(rgb2gray(im1));
% im2 = imread('hand.bmp'); im2 = double(rgb2gray(im2));
% im3 = imread('mask.bmp'); im3 = double(im3);
% C = BlendArbitrary(im1, im2, im3/255, 4);


% C = BlendArbitrary(im1, im2, im2, 4);
% boundary_pengzhang = strel('disk',1);
% edge_overlap_pengzhang = imdilate(edge_overlap,boundary_pengzhang);%%%peng zhang
% 
% ref_overlap_move = ref_overlap;
% ref_overlap_move(edge_overlap_pengzhang == 1) = 1;

extend_moasic1 = img_moasic1(:,:,1);
extend_moasic2 = img_moasic1(:,:,2);
extend_moasic3 = img_moasic1(:,:,3);

extend_moasic1(find(ref_overlap == 2)) = 0;
extend_moasic2(find(ref_overlap == 2)) = 0;
extend_moasic3(find(ref_overlap == 2)) = 0;

extend_moaic(:,:,1) = extend_moasic1;
extend_moaic(:,:,2) = extend_moasic2;
extend_moaic(:,:,3) = extend_moasic3;
img_moasic2 = img_moasic2 + extend_moaic;

img_moasic1(up:up+M3-1,left:left+N3-1,:) = img21;
img_moasic1(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = img1;



%%%%%%%%%%%%%%%%%%%%%%去除旋转图像黑边%%%
img_moasic1_r = img_moasic1(:,:,1);
img_moasic1_g = img_moasic1(:,:,2);
img_moasic1_b = img_moasic1(:,:,3);


img_moasic2_r = img_moasic2(:,:,1);
img_moasic2_g = img_moasic2(:,:,2);
img_moasic2_b = img_moasic2(:,:,3);

edge_overlap(Yoffset+1,:) = 0;
edge_overlap(:,Xoffset+N1+1) = 0;


img_moasic2_r(find(edge_overlap == 1)) = img_moasic1_r(find(edge_overlap == 1));
img_moasic2_g(find(edge_overlap == 1)) = img_moasic1_g(find(edge_overlap == 1));
img_moasic2_b(find(edge_overlap == 1)) = img_moasic1_b(find(edge_overlap == 1));

img_moasic2(:,:,1) = img_moasic2_r;
img_moasic2(:,:,2) = img_moasic2_g;
img_moasic2(:,:,3) = img_moasic2_b;
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%
x = 64;%%%16x为降采样的图像大小
img_moasic_blend1 = zeros(16*x,16*x,3);
img_moasic_blend2 = zeros(16*x,16*x,3);
img_moasic_blend  = zeros(16*x,16*x,3);
c1 = zeros(16*x,16*x);%%%本方法拼接缝参考
boundary = ones(16*x,16*x);%%%图像边界为拼接缝参考
boundary(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = 0;
%%%
a1 = double((img_moasic1(:,:,1)));
img_moasic_blend1(1:m,1:n,1) = a1;

a2 = double((img_moasic1(:,:,2)));
img_moasic_blend1(1:m,1:n,2) = a2;

a3 = double((img_moasic1(:,:,3)));
img_moasic_blend1(1:m,1:n,3) = a3;

%%%
b1 = double((img_moasic2(:,:,1)));
img_moasic_blend2(1:m,1:n,1) = b1;

b2 = double((img_moasic2(:,:,2)));
img_moasic_blend2(1:m,1:n,2) = b2;

b3 = double((img_moasic2(:,:,3)));
img_moasic_blend2(1:m,1:n,3) = b3;

%%%
c1(1:m,1:n) = double(im2);%%融合参考
img_moasic_blend1 = double(img_moasic_blend1);
img_moasic_blend2 = double(img_moasic_blend2);

C1 = BlendArbitrary(img_moasic_blend2(:,:,1), img_moasic_blend1(:,:,1), c1, 4);
C2 = BlendArbitrary(img_moasic_blend2(:,:,2), img_moasic_blend1(:,:,2), c1, 4);
C3 = BlendArbitrary(img_moasic_blend2(:,:,3), img_moasic_blend1(:,:,3), c1, 4);


img_moasic_blend(:,:,1) = C1;
img_moasic_blend(:,:,2) = C2;
img_moasic_blend(:,:,3) = C3;
img_moasic_blend = img_moasic_blend(1:m,1:n,:);

figure;imshow(uint8(img_moasic_blend));title('最终结果');

%%%
boundary = double(boundary);%%融合参考

D1 = BlendArbitrary(img_moasic_blend2(:,:,1), img_moasic_blend1(:,:,1), boundary, 4);
D2 = BlendArbitrary(img_moasic_blend2(:,:,2), img_moasic_blend1(:,:,2), boundary, 4);
D3 = BlendArbitrary(img_moasic_blend2(:,:,3), img_moasic_blend1(:,:,3), boundary, 4);


img_direct_blend(:,:,1) = D1;
img_direct_blend(:,:,2) = D2;
img_direct_blend(:,:,3) = D3;
img_direct_blend = img_direct_blend(1:m,1:n,:);
img_direct_blend = uint8(img_direct_blend);
figure;imshow(img_direct_blend);title('直接平滑融合');

%%显示%%%
img_direct_blend = uint8(img_direct_blend);
img_moasic_blend = uint8(img_moasic_blend);
subplot(221);imshow(imgout);title('直接融合');
subplot(222);imshow(img_direct_blend);title('直接平滑融合');
subplot(223);imshow(img_moasic_direct);title('优化缝合线');
subplot(224);imshow(img_moasic_blend);title('最终结果');


%%%%%保存%%%%
imwrite(imgout,'1.jpg');
imwrite(img_direct_blend,'2.jpg');
imwrite(img_moasic_direct,'3.jpg');
imwrite(img_moasic_blend,'4.jpg');

imwrite(imgout,[f '_mosaic1.' ext]);
imwrite(img_direct_blend,[f '_mosaic2.' ext]);
imwrite(img_moasic_direct,[f '_mosaic3.' ext]);
imwrite(img_moasic_blend,[f '_mosaic4.' ext]);

imwrite(img_moasic_line,[f '_seam line.' ext]);

imwrite(saliency,[f '_saliency.' ext]);
%%%%%%%%%%%%%%%%%%评价图像质量%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%指标1：拼接缝上能检测出的边界%%%%  %%% edge_strong1  %%%1.5为该区域中的强边界,2为弱边界

%%%基于优化缝合线%%%
edge_moasic_direct = edge(rgb2gray(img_moasic_direct),'canny',0.01);%
seam_line_edge = seam_line + edge_moasic_direct;%%%2代表被检测出的缝合线像素
seam_line_edge(find(edge_strong1 == 2)) = 0;%%%%找到融合路径直接拼，缝合线能检测出的边界点。2表示被检测出的缝合线像素
%figure,imshow(uint8(100*seam_line_edge));

edge_moasic_blend = edge(rgb2gray(img_moasic_blend),'canny',0.01);%
seam_blend_edge = seam_line + edge_moasic_blend;%%%2代表被检测出的缝合线像素
seam_blend_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*seam_blend_edge));

%%%基于图形边界作为缝合线%%%
boundary_seam = ones(m,n);%%%图像边界为拼接缝参考
boundary_seam(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = 0;

boundary_seam = edge(boundary_seam,'canny');


edge_imgout = edge(rgb2gray(imgout),'canny',0.01);%
boundary_direct_edge = boundary_seam + edge_imgout;%%%2代表被检测出的缝合线像素
boundary_direct_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*boundary_direct_edge));

edge_direct_blend = edge(rgb2gray(img_direct_blend),'canny',0.01);%
boundary_blend_edge = boundary_seam + edge_direct_blend;%%%2代表被检测出的缝合线像素
boundary_blend_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*boundary_blend_edge));

figure,
subplot(221);imshow(edge_imgout);title('直接融合');
subplot(222);imshow(edge_direct_blend);title('直接平滑融合');
subplot(223);imshow(edge_moasic_direct);title('优化缝合线');
subplot(224);imshow(edge_moasic_blend);title('最终结果');
%%%%拼接线上能检测出的边界像素个数（除去弱边界）
num1=sum(sum(boundary_direct_edge == 2))%%%直接
num2=sum(sum(boundary_blend_edge == 2))%%%直接+融合
num3=sum(sum(seam_line_edge == 2))%%%缝合线
num4=sum(sum(seam_blend_edge == 2))%%%缝合线+融合

%%%%%%%%%指标2：检测到的像素个数与缝合线像素个数的比值%%%
num_dirct = abs(begin(1,1) - terminal(1,1)) + abs(begin(1,2) - terminal(1,2))%%%直接融合像素个数

num_seam_line = sum(sum(seam_line == 1))

ratio1 = num1/num_dirct
ratio2 = num2/num_dirct

ratio3 = num3/num_seam_line
ratio4 = num4/num_seam_line



%%%%%%%默认canny

%%%基于优化缝合线%%%
edge_moasic_direct = edge(rgb2gray(img_moasic_direct),'canny');%
seam_line_edge = seam_line + edge_moasic_direct;%%%2代表被检测出的缝合线像素
seam_line_edge(find(edge_strong1 == 2)) = 0;%%%%找到融合路径直接拼，缝合线能检测出的边界点。2表示被检测出的缝合线像素
%figure,imshow(uint8(100*seam_line_edge));

edge_moasic_blend = edge(rgb2gray(img_moasic_blend),'canny');%
seam_blend_edge = seam_line + edge_moasic_blend;%%%2代表被检测出的缝合线像素
seam_blend_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*seam_blend_edge));

%%%基于图形边界作为缝合线%%%
boundary_seam = ones(m,n);%%%图像边界为拼接缝参考
boundary_seam(Yoffset+1:Yoffset+M1,Xoffset+1:Xoffset+N1,:) = 0;

boundary_seam = edge(boundary_seam,'canny');


edge_imgout = edge(rgb2gray(imgout),'canny');%
boundary_direct_edge = boundary_seam + edge_imgout;%%%2代表被检测出的缝合线像素
boundary_direct_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*boundary_direct_edge));

edge_direct_blend = edge(rgb2gray(img_direct_blend),'canny');%
boundary_blend_edge = boundary_seam + edge_direct_blend;%%%2代表被检测出的缝合线像素
boundary_blend_edge(find(edge_strong1 == 2)) = 0;%%%多尺度融合后检测到的像素点
%figure,imshow(uint8(100*boundary_blend_edge));

figure,
subplot(221);imshow(edge_imgout);title('直接融合');
subplot(222);imshow(edge_direct_blend);title('直接平滑融合');
subplot(223);imshow(edge_moasic_direct);title('优化缝合线');
subplot(224);imshow(edge_moasic_blend);title('最终结果');
%%%%拼接线上能检测出的边界像素个数（除去弱边界）
num1=sum(sum(boundary_direct_edge == 2))%%%直接
num2=sum(sum(boundary_blend_edge == 2))%%%直接+融合
num3=sum(sum(seam_line_edge == 2))%%%缝合线
num4=sum(sum(seam_blend_edge == 2))%%%缝合线+融合

%%%%%%%%%指标2：检测到的像素个数与缝合线像素个数的比值%%%
num_dirct = abs(begin(1,1) - terminal(1,1)) + abs(begin(1,2) - terminal(1,2))%%%直接融合像素个数

num_seam_line = sum(sum(seam_line == 1))

ratio1 = num1/num_dirct
ratio2 = num2/num_dirct

ratio3 = num3/num_seam_line
ratio4 = num4/num_seam_line





% %%%%指标3：拼接缝造成的强边界错位
% ref_edge = ref_edge1 + ref_edge2;
% 
% ref_edge = uint8(ref_edge);%%2代表重合的边界，可以穿过； 1表示不重合的边界
% 
% ref_dis_edge = ref_edge;
% ref_dis_edge(find(ref_dis_edge == 2)) =0;%%%去掉重合边界像素，剩下全部为断裂的边界
% 
% %%%boundary%%
% dis_edge_boundary = double(ref_dis_edge) + double(boundary_seam);%%%2为断裂的像素点
% dis_edge_boundary(find(edge_strong1 == 2)) = 0;
% dis_edge_boundary(find(edge_strong2 == 2)) = 0;
% figure,imshow(uint8(100*dis_edge_boundary));
% 
% %%%seam line%%
% dis_edge_seam = double(ref_dis_edge) + double(seam_line);%%%2为断裂的像素点
% dis_edge_seam(find(edge_strong1 == 2)) = 0;
% dis_edge_seam(find(edge_strong2 == 2)) = 0;
% figure,imshow(uint8(100*dis_edge_seam));
% 
% %%计数%%%
% num_boundary = sum(sum(dis_edge_boundary == 2))%%%直接
% num_seam = sum(sum(dis_edge_seam == 2))%%%直接+融合