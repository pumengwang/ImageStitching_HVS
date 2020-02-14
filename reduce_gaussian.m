function [G, L, R] = reduce_gaussian(input, sigma ) 

% convolves the input image with a gaussian kernel of size 5x5 with
% standard deviation sigma and then samples the image to the 1/4 th of the
% original image size

% simply, blur and sample

hg = fspecial('gaussian',5, sigma);
out = conv2(input,hg,'same');

G = out;
L = input - G;
R = out(1:2:end,1:2:end);