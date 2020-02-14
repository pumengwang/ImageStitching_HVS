%MultiresolutionImages

sN = 0;
eN = 90;
folderPath = 'c:\core\data\tiger\images\';
folderPath2 = 'c:\temp\tiger\';

sigma = 2;

counter = 1;
for i=sN:eN
    
    if     i < 10
        fileName = strcat(folderPath, 'frame000', int2str(i), '.png');
        saveHName = strcat(folderPath2, 'high000', int2str(i),'.txt');
        saveLName = strcat(folderPath2, 'low000', int2str(i),'.txt');
    elseif i < 100
        fileName = strcat(folderPath, 'frame00' , int2str(i), '.png');
        saveHName = strcat(folderPath2, 'high00', int2str(i),'.txt');
        saveLName = strcat(folderPath2, 'low00', int2str(i),'.txt');    
    elseif i < 1000
        fileName = strcat(folderPath, 'frame0'  , int2str(i), '.png');
        saveHName = strcat(folderPath2, 'high0', int2str(i),'.txt');
        saveLName = strcat(folderPath2, 'low0', int2str(i),'.txt');
    end
    
    im = imread(fileName); 
    im = double(rgb2gray(im));
    
    [G, L] = reduce(im, sigma );
    
    saveMatrix(saveHName,L');
    saveMatrix(saveLName,G');
end