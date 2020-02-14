function R = DrawRegion(im)

f1 = figure;
imshow(uint8(im)); hold on;

[m,n] = size(im);
R = zeros(m,n);

button = 1;
while (button == 1)
    figure(f1);
    
    hold on
    [X1,Y1,button] = GINPUT(1);
    plot(X1,Y1,'r');
    [X2,Y2,button] = GINPUT(1);
    plot(X2,Y2,'b');
  
    if ( button == 1 )
        rh = rectangle('Position',[X1,Y1,X2-X1,Y2-Y1]);
        set(rh,'EdgeColor', [1 0 0]  ,'LineWidth',2);
        R(Y1:Y2,X1:X2) = 1;
    end
end

figure; imshow(uint8(stretchImage(R)));

