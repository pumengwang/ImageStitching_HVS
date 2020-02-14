function saveMatrix(str,A)
fn = fopen(str,'w');
fprintf(fn,'%10.4e ',A);
fclose(fn);