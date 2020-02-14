function out = expand(input)

[m,n] = size(input);

out(2*m,2*n) = 0;
out(1:2:2*m, 1:2:2*n) = input;

a = 0.4;
b = 0.25;
c = 0.05;
h  = [c b a b c];
hp = conv2(h,h');

out = 4*conv2(out,hp,'same');