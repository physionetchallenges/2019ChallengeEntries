function res=normalize015(x,minv,maxv)


res= 1+4*(x-repmat(minv,[size(x,1),1]))./repmat(maxv-minv,[size(x,1),1]);

