function out = reduce(input)

% smoothing filter
a = 0.4;
b = 0.25;
c = 0.05;
h  = [c b a b c];
hp = conv2(h,h');

[m,n] = size( input );
mo = m/2;
no = n/2;

sym_input(m+4,n+4) = 0;

sym_input(3:m+2,3:n+2) = input;
sym_input(1,3:n+2) = 2*input(1,:)-input(3,:);
sym_input(2,3:n+2) = 2*input(1,:)-input(2,:);
sym_input(m+4,3:n+2) = 2*input(m,:)-input(m-2,:);
sym_input(m+3,3:n+2) = 2*input(m,:)-input(m-1,:);

sym_input(3:m+2,1) = 2*input(:,1)-input(:,3);
sym_input(3:m+2,2) = 2*input(:,1)-input(:,2);
sym_input(3:m+2,n+4) = 2*input(:,n)-input(:,n-2);
sym_input(3:m+2,n+3) = 2*input(:,n)-input(:,n-1);


out = conv2(sym_input,hp,'same');
out = out(3:2:end-2,3:2:end-2);