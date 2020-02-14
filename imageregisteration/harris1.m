%%%Prewitt Operator Corner Detection.m
%%%时间优化--相邻像素用取差的方法
function lc=harris1(Image,hsize)   %C是角点矩阵，count是角点数量，二者都是输出
 if nargin == 0
	Image = imread('feiji1.jpg');
    hsize=5;
 end
[x,y,z]=size(Image);
if z==3
%Image = imread( 'feiji.jpg');                 % 读取图像
Image = im2uint8(rgb2gray(Image));   
end;

dx = [-1 0 1;-1 0 1;-1 0 1];  %dx：横向Prewitt差分模版
Ix2 = filter2(dx,Image).^2;   
Iy2 = filter2(dx',Image).^2;                                         
Ixy = filter2(dx,Image).*filter2(dx',Image);


%生成 5*5 或 7*7 或 9*9 的高斯窗口。窗口越大，探测到的角点越少。
if hsize==5
    h= fspecial('gaussian',5,2);  
elseif hsize==7
    h= fspecial('gaussian',7,2);
elseif hsize==9
    h= fspecial('gaussian',9,2);  
else
    h= fspecial('gaussian',5,2);  
end
A = filter2(h,Ix2);       % 用高斯窗口差分Ix2得到A 
B = filter2(h,Iy2);                                 
C = filter2(h,Ixy);                                  
nrow = size(Image,1);                            
ncol = size(Image,2); 

det1=zeros(nrow,ncol);
lam1=zeros(nrow,ncol);
Corner = zeros(nrow,ncol); %矩阵Corner用来保存候选角点位置,初值全零，值为1的点是角点
                           %真正的角点在137和138行由(row_ave,column_ave)得到
%参数t:点(i,j)八邻域的“相似度”参数，只有中心点与邻域其他八个点的像素值之差在
%（-t,+t）之间，才确认它们为相似点，相似点不在候选角点之列
% -t<Image(i,j)-Image(i±1,j±1)<+t

%我并没有全部检测图像每个点，而是除去了边界上boundary个像素，
%因为我们感兴趣的角点并不出现在边界上
t =20;
boundary=8;
Image=double(Image);
for i=boundary:nrow-boundary+1 
    for j=boundary:ncol-boundary+1
        nlike=0; %相似点个数
        if Image(i-1,j-1)>Image(i,j)-t && Image(i-1,j-1)<Image(i,j)+t 
            nlike=nlike+1;
        end
        if Image(i-1,j)>Image(i,j)-t && Image(i-1,j)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if Image(i-1,j+1)>Image(i,j)-t && Image(i-1,j+1)<Image(i,j)+t  
            nlike=nlike+1;
        end  
        if Image(i,j-1)>Image(i,j)-t && Image(i,j-1)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if Image(i,j+1)>Image(i,j)-t && Image(i,j+1)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if Image(i+1,j-1)>Image(i,j)-t && Image(i+1,j-1)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if Image(i+1,j)>Image(i,j)-t && Image(i+1,j)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if Image(i+1,j+1)>Image(i,j)-t && Image(i+1,j+1)<Image(i,j)+t  
            nlike=nlike+1;
        end
        if nlike>=2 && nlike<=6
            Corner(i,j)=1;%如果周围有0，1，7，8个与中心的（i,j）相似
                          %那(i,j)就不是角点，所以，直接忽略
        end;
    end;
end;

CRF = zeros(nrow,ncol);    % CRF用来保存角点响应函数值,初值全零
CRFmax = 0;                % 图像中角点响应函数的最大值，作阈值之用 
t=0.05;   
% 计算CRF
%工程上常用CRF(i,j) =det(M)/trace(M)计算CRF，那么此时应该将下面第103行的
%比例系数t设置大一些，t=0.1对采集的这几幅图像来说是一个比较合理的经验值
for i = boundary:nrow-boundary+1 
for j = boundary:ncol-boundary+1
    if Corner(i,j)==1  %只关注候选点
        M = [A(i,j) C(i,j);
             C(i,j) B(i,j)];      
         CRF(i,j) = det(M)-t*(trace(M))^2;
         det1(i,j)=det(M);
         lam1(i,j)=trace(M);
        if CRF(i,j) > CRFmax 
            CRFmax = CRF(i,j);
        end;            
    end
end;             
end;  

count = 0;       % 用来记录角点的个数
t=0.01;         
% 下面通过一个3*3的窗口来判断当前位置是否为角点
for i = boundary:nrow-boundary+1 
for j = boundary:ncol-boundary+1
        if Corner(i,j)==1  %只关注候选点的八邻域
            if CRF(i,j) > t*CRFmax && CRF(i,j) >CRF(i-1,j-1) ......
               && CRF(i,j) > CRF(i-1,j) && CRF(i,j) > CRF(i-1,j+1) ......
               && CRF(i,j) > CRF(i,j-1) && CRF(i,j) > CRF(i,j+1) ......
               && CRF(i,j) > CRF(i+1,j-1) && CRF(i,j) > CRF(i+1,j)......
               && CRF(i,j) > CRF(i+1,j+1) 
          % t11=t*CRFmax;
          % t22=CRFmax;
            count=count+1;%这个是角点，count加1
            else % 如果当前位置（i,j）不是角点，则在Corner(i,j)中删除对该候选角点的记录
                Corner(i,j) = 0;     
            end;
        end; 
end; 
end; 
[x,y]=find(Corner==1);
lc=[x,y];
%pause; %不加参数，程序暂停，直至按任意一个按键； 加参数，程序暂停x秒。
%disp('角点个数');
%disp(count)
%pause;
%imshow(Image);      % display Intensity Image 
% toc(t1)
fid=fopen('harris.txt','w');
fprintf(fid,'harris角点检测运行仿真报告\r\n');
fprintf(fid,'== %s  %s == \r\n',datestr(date,26),datestr(now,13));
fprintf(fid,'--------------------------\r\n');
fprintf(fid,'输入图像的纵高为：%d\r\n',x);
fprintf(fid,'输入图像的横宽为：%d\r\n',y);
fprintf(fid,'\r\n');
fprintf(fid,'高斯滤波窗口大小为：%d*%d\r\n',hsize,hsize);
fprintf(fid,'\r\n');
fprintf(fid,'响应函数极大值CRFmax = %f\r\n',CRFmax);
fprintf(fid,'\r\n');
fprintf(fid,'角点个数为：%d\r\n',count);
fprintf(fid,'\r\n');
fprintf(fid,'角点坐标以二进制的形式存放在corner.txt文件中\r\n');
C=Corner;
        