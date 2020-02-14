function G = GaussianPyramid(input, level)

% function constructs the so called Gaussian Pyramid;
% it is the representation of an image with different resolution levels
% The pyramid is formed by successively bluring and sampling of the
% input image. 
% The output is an [m x n x level] matrix. At each level the size of the
% image is reduced by 1/4 th of its original value by sampling the x and y
% directions with 2. The 0th level is equal to the original image and to
% get the level Nth image use, G(1:m/2^(N-1),1:n/2^(N-1),N)


[m,n] =size(input);

G(:,:,1) = input;
g = input;
for i=2:level
    s = 1/power(2,i-1);
    g = reduce(g);
    G(1:m*s,1:n*s,i) = g;
end