function [H corrPtIdx] = findHomography1(pts1,pts2)
% [H corrPtIdx] = findHomography(pts1,pts2)
%	Find the homography between two planes using a set of corresponding
%	points. PTS1 = [x1,x2,...;y1,y2,...]. RANSAC method is used.
%	corrPtIdx is the indices of inliers.
%功能：根据两个面的特征点确定这两个平面的转换关系，在此当中使用了ransac算法
%输入：pts1--平面1特征点的坐标
%输出： H-两个平面的转换矩阵 corrPtIdx--特征点索引
minPtNum = 3;
iterNum = 200;
thInlrRatio = .1;
thDist =20;
ptNum = size(pts1,2);
thInlr = round(thInlrRatio*ptNum);

roundnum=zeros(iterNum,3);
f=zeros(3,3);
inliers_last=0;
dist_sum_last=0;
dist_sum=0;
q=0;
q1=0;
q2=0;  
q3=0;%for testing
for p = 1:iterNum
	% 1. fit using  random points
%随机选择样本
if minPtNum > ptNum
	errordlg('所提供的角点对数过少，不能计算','出错');
    return
end
sampleIdx = zeros(1,minPtNum);
available = 1:ptNum;
rs = ceil(rand(1,minPtNum).*(ptNum:-1:ptNum-minPtNum+1));
fid=fopen('ransac_roundnumber.txt','rb');

for p1 = 1:minPtNum
    inliers=0;
    dist_sum=0;
	while rs(p1) == 0
		rs(p1) = ceil(rand(1)*(ptNum-p1+1));
	end
	sampleIdx(p1) = available(rs(p1));
	available(rs(p1)) = [];
end
roundnum(p,1:3)=sampleIdx;
   A=[pts1(:,sampleIdx);ones(1,3)];
   B=[pts2(:,sampleIdx);ones(1,3)];
   
   f1=solveH(A',B');
	
	% 2. count the inliers, if more than thInlr, refit; else iterate
	
    n = size(pts1,2);
    pts22=[pts2;ones(1,n)]';
    pts3 =pts22*f1;
    pts12=[pts1;ones(1,n)]';
    dist = sum((pts3-pts12).^2,2);
    for i= 1:ptNum
	if dist(i) < thDist;
	inliers = inliers+1;
    dist_sum=dist(i)+dist_sum;
    end
    end
   
    if inliers<thInlr
        q1=q1+1;
        continue; 
    end
	if inliers>inliers_last
		inliers_last=inliers;
		dist_sum_last=dist_sum;
	    f=f1;
        q2=q2+1;
	else if inliers==inliers_last
            q3=q3+1;
			if dist_sum <=dist_sum_last              
                	inliers_last=inliers;
					dist_sum_last=dist_sum;   
                    f=f1;
                     q=q+1;
            end
        end
       
    end
    
 end
%[~,idx] = max(inlrNum);
%[~,idx] = min(dist_sum);
%f = fLib{idx};
n = size(pts1,2);
pts4 = [pts2;ones(1,n)]'*f;

dist1 = sum((pts12-pts4).^2,2);

H=f;
corrPtIdx = find(dist1< thDist);
fid= fopen('ransac_roundnum.txt','wb');
%roundnum=uint(roundnum');
fwrite(fid,roundnum);
q
q1
q2
q3
H
end