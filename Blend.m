function C = Blend(A, B, level, boundary)

% function blends two images using the multiband blending technique 
% presented in "A Multiresolution Spline With Application to image
% mosaics" by Burt and Adelson.
% The blending is performed at a blending boundary specified by the
% "boundary" parameter. 
% A and B are the two input images and level is the max. level of the
% laplacian pyramid

[ma, na] = size(A);

[LA, GA]= LaplacianPyramid(A,level);
[LB, GB]= LaplacianPyramid(B,level);

LC(ma,na,level) = 0;
for i = 1:level
    s = 1/power(2,i-1);
    LC(:,1:boundary*s,i) = LA(:,1:boundary*s,i);
    LC(:,boundary*s+1:end,i) = LB(:,boundary*s+1:end,i);   
end

C = reconstruct(LC);

figure;imshow(uint8(A));title('image 1');
figure;imshow(uint8(B));title('image 2');
figure;imshow(uint8(C));title('result');

D(:,1:boundary) = A(:,1:boundary);
D(:,boundary+1:na) = B(:,boundary+1:na);
figure;imshow(uint8(D));title('D');