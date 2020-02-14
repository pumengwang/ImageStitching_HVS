function out = stretchImage(indexImage)

minI = min(min(indexImage));
maxI = max(max(indexImage));

out = ( indexImage - minI ) / (maxI - minI  ) * 255;

