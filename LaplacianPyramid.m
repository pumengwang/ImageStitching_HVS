function [L, G] = LaplacianPyramid( input, level )

% function constructs the so called laplacian pyramid of an image. Each
% level is equal to a different band in the freq. domain. It is found by
% taking the difference of the expanded higher levels of the gaussian
% pyramid from a succesive lower level. 

[m,n] =size(input);

G = GaussianPyramid(input,level);

for i = 1:level-1
    
    s = 1/power(2,i-1);
    L(1:m*s,1:n*s,i) = G(1:m*s,1:n*s,i) - expand(G(1:m*s/2,1:n*s/2,i+1));
    
end

s = 1/power(2,level-1);
L(1:m*s,1:n*s,level) = G(1:m*s,1:n*s,level);