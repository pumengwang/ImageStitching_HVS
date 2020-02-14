function out = reconstruct(input_pyramid)

[m,n, level] = size(input_pyramid);

s = 1/power(2,level-1);

out = input_pyramid(1:m*s,1:n*s,level);

for i = level-1:-1:1
    s = 1/power(2,i-1);
    out = expand(out) + input_pyramid(1:m*s,1:n*s,i);      
end