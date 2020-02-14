
function [matchLoc1, matchLoc2] = findCorr(img1,img2,locs1, locs2)

if size(img1,3)==3
img1=rgb2gray(img1);
end
if size(img2,3)==3
img2=rgb2gray(img2);
end

matchTable = zeros(1,size(locs1,1));
ncc=zeros(1,size(locs2,1));
counter=zeros(1,size(locs2,1));%to make one one pair,because I find different points from locs1,similar to the same points in locs2.
i1=zeros(5,5*size(locs1,1));
i2=zeros(5,5*size(locs2,1));
 for i=1:size(locs1,1)
  
         i1(:,5*i-4)=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)-2);
         i1(:,5*i-3)=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)-1);
         i1(:,5*i-2)=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)-0);
        i1(:,5*i-1)=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)+1);
         i1(:,5*i-0)=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)+2);
         
    
 end
  for i=1:size(locs2,1)
        i2(:,5*i-4)=img2(locs2(i,1)-2:locs2(i,1)+2,locs2(i,2)-2);
         i2(:,5*i-3)=img2(locs2(i,1)-2:locs2(i,1)+2,locs2(i,2)-1);
         i2(:,5*i-2)=img2(locs2(i,1)-2:locs2(i,1)+2,locs2(i,2)-0);
        i2(:,5*i-1)=img2(locs2(i,1)-2:locs2(i,1)+2,locs2(i,2)+1);
         i2(:,5*i-0)=img2(locs2(i,1)-2:locs2(i,1)+2,locs2(i,2)+2);
  end
 for i=1:size(locs1,1)
     for j=1:size(locs2,1)

% For each descriptor in the first image, select its match to second image.



    i1Patch1=img1(locs1(i,1)-2:locs1(i,1)+2,locs1(i,2)-2: locs1(i,2)+2);
    i2Patch1=img2(locs2(j,1)-2:locs2(j,1)+2,locs2(j,2)-2: locs2(j,2)+2);
   
    
    i1PatchMean=mean(mean(i1Patch1));
    i2PatchMean=mean(mean(i2Patch1));
    i1Patch1=double(i1Patch1)-i1PatchMean;
    i2Patch1=double(i2Patch1)-i2PatchMean;%sustract the mean first 
    i1PSumSq=sum(sum(i1Patch1.^2))^0.5;
    i2PSumSq=sum(sum(i2Patch1.^2))^0.5;
    i1PatchNorml=i1Patch1/i1PSumSq;
    i2PatchNorml=i2Patch1/i2PSumSq;
    ncc(j)= sum(sum(i1PatchNorml.*i2PatchNorml));
    
    % Computes vector of dot products
    end
 [vals,index]=sort(ncc)  ;
 

   % Check if nearest neighbor has angle less than distRatio times 2nd.
   
   if (vals(end)>0.8)&&(counter(index(end))==0)
      matchTable(i) = index(end);
      counter(index(end))=1;
   else
      matchTable(i) = 0;
   end
     
 end
% save matchdata matchTable
%}

% Create a new image showing the two images side by side.
img3 = appendimages(img1,img2);

% Show a figure with lines joining the accepted matches.
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
cols1 = size(img1,2);
for i = 1: size(locs1,1)
  if (matchTable(i) > 0)
    line([locs1(i,2) locs2(matchTable(i),2)+cols1], ...
         [locs1(i,1) locs2(matchTable(i),1)], 'Color', 'c');
  end
end
hold off;
num = sum(matchTable > 0);
fprintf('NCC Found %d matches.\n', num);

idx1 = find(matchTable);
idx2 = matchTable(idx1);
x1 = locs1(idx1,2);
x2 = locs2(idx2,2);
y1 = locs1(idx1,1);
y2 = locs2(idx2,1);

matchLoc1 = [x1,y1];
matchLoc2 = [x2,y2];

end