% Harris detector
% 2015.12.24 by Wang Jingbo
% using for image stitching
function [locs] = Harrisv2(frame)
if size(frame,3)==3
I=rgb2gray(frame);
else
    I=frame;
end
 I =double(I);
 sigma=2.5;
 a=size(I,1);
 b=size(I,2);
 num=0;

 dx = [-1 0 1; -1 0 1; -1 0 1]; % The Mask 
    dy = dx';
    Ix = conv2(I, dx, 'same');   
    Iy = conv2(I, dy, 'same');
    g = fspecial('gaussian',5, sigma);
    %gaussian filter
    Ixx= conv2(Ix.^2, g, 'same');  
    Iyy = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g,'same');
    I11=zeros(a,b);
    for i=2:a-7
        for j=2:b-7
            t1=I(i+3,j+3)-20;
           t2=I(i+3,j+3)+20;
            N=[0,0,0,0,0,0,0,0];
            if I(i+2,j+2)>t1&& I(i+2,j+2)<t2
              N(1)=N(1)+1;
            end
            if I(i+2,j+3)>t1&& I(i+2,j+3)<t2
              N(2)=N(2)+1;
            end
             if I(i+2,j+4)>t1&& I(i+2,j+4)<t2
              N(3)=N(3)+1;
             end
             if I(i+3,j+2)>t1&& I(i+3,j+2)<t2
              N(4)=N(4)+1;
             end
             if I(i+3,j+4)>t1&& I(i+3,j+4)<t2
              N(5)=N(5)+1;
             end
            if I(i+4,j+2)>t1&& I(i+4,j+2)<t2
              N(6)=N(6)+1;
            end
             if I(i+4,j+3)>t1&& I(i+4,j+3)<t2
              N(7)=N(7)+1;
             end
             if I(i+4,j+4)>t1&& I(i+4,j+4)<t2
              N(8)=N(8)+1;
             end
             S=sum(N);
            if S>=2&&S<=6
                I11(i+3,j+3)=1;
                num=num+1;
            end
        end
    end
            [r1,c1]=find(I11);
            PI=[r1,c1];
     CRF=zeros(3,3,num); 
     in=zeros(1,num);
     for  i=1:num
        Ixxpatch=Ixx(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
        Iyypatch=Iyy(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
        Ixypatch=Ixy(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
%         Ixxpatch=Ixx(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
%         Iyypatch=Iyy(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
%         Ixypatch=Ixy(PI(i,1)-1:PI(i,1)+1,PI(i,2)-1:PI(i,2)+1);
     k = 0.0498;
    CRF(:,:,i) = (Ixxpatch.*Iyypatch) - Ixypatch.^2 - k*(Ixxpatch + Iyypatch).^2;
    in(1,i)=CRF(2,2,i);
     end
    
    CRFmax=max(CRF(2,2,:));
    for i=1:num
        MAX=max(max(CRF(:,:,i)));
        if (CRF(2,2,i)==MAX)&&(CRF(2,2,i)>0.01*CRFmax)
            I11(PI(i,1),PI(i,2))=1;
        else 
            I11(PI(i,1),PI(i,2))=0;
        end
    end
       [r2,c2]=find(I11);
           locs=[r2,c2]; 
           I=uint8(I);
           imshow(I);
           hold on
           plot(c2,r2,'r+');
            
        
        
    
     
                
